"""
tests/test_vits2_tts_plugin.py — VITS2 TTS lifecycle tests (host-side).

The point of these: the VITS2 model is a 60 MB download plus three TensorRT
engines plus a warmup pass, while Agent Core gives a processor tools/call 60 s
(agent-core/src/mcp_client.py). So `start` must never wait for the model, and
every path that abandons an utterance must still release its ACP action.

ROS stubs come from vision_stubs (installed by conftest before collection).
Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import threading
import time

import pytest

from vision_stubs import (  # noqa: F401
    _FakeExecutor,
    _FakeNode,
    _wait_until,
)

import plugins.vits2_tts_trt.plugin as vits2  # noqa: E402
from plugins.vits2_tts_trt.adapter import Vits2TensorRTAdapter  # noqa: E402


class _FakeAdapter(Vits2TensorRTAdapter):
    """A Vits2TensorRTAdapter that synthesizes without TensorRT.

    Subclassed rather than duck-typed because the plugin asserts the adapter is
    a Vits2TensorRTAdapter before installing it — that isinstance check is the
    guard against silently running some other backend.
    """

    def __init__(self):
        self.speed = 1.0
        self.spoken = []
        self.warmups = 0

    def set_speed(self, speed: float) -> None:
        self.speed = float(speed)

    def warmup(self) -> int:
        self.warmups += 1
        return 3200

    def synthesize(self, text: str) -> bytes:
        return b"".join(self.synthesize_stream(text))

    def synthesize_stream(self, text: str):
        self.spoken.append(text)
        yield b"\x00\x01" * 8


@pytest.fixture(autouse=True)
def _fast_audio_gate(monkeypatch):
    """Drop the DDS subscriber-settle wait; the fake publisher is always matched."""
    monkeypatch.setattr(vits2, "SUBSCRIBER_SETTLE_MS", 0)
    monkeypatch.setattr(vits2, "SUBSCRIBER_WAIT_MS", 100)
    monkeypatch.setattr(vits2, "FRAME_INTERVAL_MS", 0)


@pytest.fixture
def completions(monkeypatch):
    """Capture ACP completion callbacks instead of POSTing to Agent Core."""
    seen = []
    monkeypatch.setattr(
        vits2, "_complete_action",
        lambda action_id, text, frames, interrupted: seen.append(
            (action_id, text, frames, interrupted)
        ),
    )
    return seen


def _plugin(monkeypatch, *, load_delay=0.0, fail=None, adapter=None):
    """Build a plugin whose loader is instrumented instead of touching a GPU."""
    adapter = adapter or _FakeAdapter()
    state = {"loads": 0, "engine_dirs": []}

    def fake_ensure(model_dir, family=None):
        state["loads"] += 1
        if load_delay:
            time.sleep(load_delay)
        if fail is not None and state["loads"] <= fail:
            raise RuntimeError("release download failed")
        return f"{model_dir}/engines/jp61"

    def fake_build(cfg):
        state["engine_dirs"].append(cfg.get("engine_dir"))
        adapter.set_speed(float(cfg.get("speed", 1.0)))
        return adapter

    import utils.model_downloader as md
    monkeypatch.setattr(md, "ensure_vits2_model", fake_ensure, raising=False)
    monkeypatch.setattr(vits2, "build_adapter", fake_build)

    executor = _FakeExecutor()
    plugin = vits2.TTSPlugin(
        {"model_dir": "/models/vits2", "backend": "trt", "speed": 1.0}, executor
    )
    return plugin, executor, adapter, state


# ── start never blocks on the model ──────────────────────────────────────────

def test_start_returns_loading_without_waiting_for_the_model(monkeypatch):
    plugin, executor, _, state = _plugin(monkeypatch, load_delay=0.4)

    began = time.monotonic()
    result = plugin.dispatch("tts", {"action": "start", "input_topic": "/say",
                                     "instance_id": "a"})
    elapsed = time.monotonic() - began

    assert result["state"] == "loading"
    assert elapsed < 0.2, f"start blocked for {elapsed:.2f}s"
    # ...and the instance comes up on its own once the load finishes.
    assert _wait_until(lambda: executor.nodes and executor.nodes[0].state == "running")
    assert state["loads"] == 1
    assert state["engine_dirs"] == ["/models/vits2/engines/jp61"]


def test_concurrent_starts_load_once_and_yield_one_node_each(monkeypatch):
    plugin, executor, _, state = _plugin(monkeypatch, load_delay=0.3)

    results = []
    threads = [
        threading.Thread(
            target=lambda i=i: results.append(
                plugin.dispatch("tts", {"action": "start",
                                        "input_topic": f"/say{i}",
                                        "instance_id": f"i{i}"})
            )
        )
        for i in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r["state"] == "loading" for r in results)
    assert _wait_until(lambda: len(executor.nodes) == 6)
    assert state["loads"] == 1, f"downloaded {state['loads']} times"


def test_info_reports_loading_for_instance_and_aggregate(monkeypatch):
    plugin, _, _, _ = _plugin(monkeypatch, load_delay=0.4)
    plugin.dispatch("tts", {"action": "start", "input_topic": "/say",
                            "instance_id": "a"})

    # Regression: the instance branch used to answer "idle" mid-download, so the
    # dashboard showed an idle device that refused to speak.
    per_instance = plugin.dispatch("tts", {"action": "info", "instance_id": "a"})
    aggregate = plugin.dispatch("tts", {"action": "info"})
    assert per_instance["state"] == "loading"
    assert aggregate["state"] == "loading"
    assert "initializing" in per_instance["desc"]

    assert _wait_until(
        lambda: plugin.dispatch("tts", {"action": "info"})["state"] == "running"
    )


def test_info_never_triggers_a_load(monkeypatch):
    plugin, _, _, state = _plugin(monkeypatch)
    for _ in range(3):
        assert plugin.dispatch("tts", {"action": "info"})["state"] == "idle"
    assert state["loads"] == 0


# ── failure and retry ────────────────────────────────────────────────────────

def test_load_failure_surfaces_then_next_start_retries(monkeypatch):
    plugin, executor, _, state = _plugin(monkeypatch, fail=1)

    first = plugin.dispatch("tts", {"action": "start", "input_topic": "/say"})
    assert first["state"] == "loading"
    assert _wait_until(
        lambda: plugin.dispatch("tts", {"action": "info"})["state"] == "error"
    )
    info = plugin.dispatch("tts", {"action": "info"})
    assert "release download failed" in info["error"]

    # A retry is not blocked by the sticky error, and it succeeds.
    plugin.dispatch("tts", {"action": "start", "input_topic": "/say"})
    assert _wait_until(lambda: executor.nodes and executor.nodes[0].state == "running")
    assert state["loads"] == 2


def test_load_failure_releases_queued_speak_actions(monkeypatch, completions):
    plugin, _, _, _ = _plugin(monkeypatch, fail=1, load_delay=0.2)

    queued = plugin.dispatch("tts", {"action": "speak", "text": "你好"})
    assert queued["status"] == "queued"
    action_id = queued["action_id"]

    # The ACP barrier waits for this action; a failed load must still end it.
    assert _wait_until(lambda: any(c[0] == action_id for c in completions))
    assert [c for c in completions if c[0] == action_id][0][3] is True


# ── stop / interrupt while still loading ─────────────────────────────────────

def test_stop_during_loading_cancels_the_pending_instance(monkeypatch):
    plugin, executor, _, _ = _plugin(monkeypatch, load_delay=0.3)

    plugin.dispatch("tts", {"action": "start", "input_topic": "/say",
                            "instance_id": "a"})
    assert plugin.dispatch("tts", {"action": "stop", "instance_id": "a"}) == {
        "state": "idle"
    }

    time.sleep(0.5)  # let the loader finish and try to bring instances up
    assert executor.nodes == [], "cancelled instance was started anyway"
    assert plugin.dispatch("tts", {"action": "info"})["state"] == "idle"


def test_stop_during_loading_releases_queued_speak(monkeypatch, completions):
    plugin, executor, _, _ = _plugin(monkeypatch, load_delay=0.3)

    queued = plugin.dispatch("tts", {"action": "speak", "text": "取消我"})
    plugin.dispatch("tts", {"action": "stop"})

    assert _wait_until(lambda: any(c[0] == queued["action_id"] for c in completions))
    assert [c for c in completions if c[0] == queued["action_id"]][0][3] is True
    time.sleep(0.4)
    assert executor.nodes == []


# ── speak queued before the model is resident ────────────────────────────────

def test_speak_before_ready_plays_once_the_model_lands(monkeypatch, completions):
    plugin, executor, adapter, _ = _plugin(monkeypatch, load_delay=0.2)

    queued = plugin.dispatch("tts", {"action": "speak", "text": "延迟播报"})
    assert queued["action_id"].startswith("speak-")

    assert _wait_until(lambda: adapter.spoken == ["延迟播报"])
    assert _wait_until(lambda: any(c[0] == queued["action_id"] for c in completions))
    # Completed, not cancelled, and the utterance reached a real node.
    assert [c for c in completions if c[0] == queued["action_id"]][0][3] is False
    assert executor.nodes and executor.nodes[0].state == "running"


def test_speak_after_ready_reuses_the_running_node(monkeypatch):
    plugin, executor, adapter, state = _plugin(monkeypatch)

    plugin.dispatch("tts", {"action": "start", "input_topic": "/say"})
    assert _wait_until(lambda: executor.nodes and executor.nodes[0].state == "running")
    plugin.dispatch("tts", {"action": "speak", "text": "第一句"})
    plugin.dispatch("tts", {"action": "speak", "text": "第二句"})

    assert _wait_until(lambda: adapter.spoken == ["第一句", "第二句"])
    assert len(executor.nodes) == 1
    assert state["loads"] == 1


# ── config ───────────────────────────────────────────────────────────────────

def test_config_speed_updates_a_resident_model_without_a_reload(monkeypatch):
    plugin, executor, adapter, state = _plugin(monkeypatch)

    plugin.dispatch("tts", {"action": "start", "input_topic": "/say"})
    assert _wait_until(lambda: executor.nodes and executor.nodes[0].state == "running")
    node = executor.nodes[0]

    assert plugin.dispatch("tts", {"action": "config", "speed": 1.4}) == {
        "status": "configured"
    }
    assert adapter.speed == pytest.approx(1.4)
    # A slider change must not cost a 60 MB reload or drop the live node.
    assert state["loads"] == 1
    assert executor.nodes == [node] and node.state == "running"


def test_config_during_loading_discards_the_stale_adapter(monkeypatch):
    plugin, executor, adapter, state = _plugin(monkeypatch, load_delay=0.3)

    plugin.dispatch("tts", {"action": "start", "input_topic": "/say"})
    plugin.dispatch("tts", {"action": "config", "speed": 0.8})

    # The in-flight loader was building from the old config, so its adapter must
    # be dropped — but the pending start survives and a fresh load serves it.
    assert _wait_until(lambda: executor.nodes and executor.nodes[0].state == "running",
                       timeout=5.0)
    assert adapter.speed == pytest.approx(0.8)
    assert state["loads"] == 2
    assert len(executor.nodes) == 1


def test_speaker_id_other_than_zero_is_refused(monkeypatch):
    _plugin(monkeypatch)  # installs the fakes
    with pytest.raises(ValueError):
        vits2.TTSPlugin({"backend": "trt", "speaker_id": 3}, _FakeExecutor())
    with pytest.raises(ValueError):
        vits2.TTSPlugin({"backend": "onnx"}, _FakeExecutor())


# ── the public tool surface ──────────────────────────────────────────────────

def test_tool_is_the_standard_tts_tool_with_an_engine_selector():
    tools = vits2.TOOLS
    assert [t["name"] for t in tools] == ["tts"]
    config = tools[0]["configSchema"]["properties"]
    # The engine has to be visible in the device panel, or switching it means
    # rebuilding the image (see PR #112 review).
    assert config["tts_engine"]["enum"] == ["vits2_trt", "sherpa_onnx"]
    assert vits2.TTSPlugin.PREFIX == "tts"


# ── ACP callback and error reporting (device regressions) ────────────────────

def test_acp_callback_tolerates_the_self_signed_agent_core_cert(monkeypatch):
    """Agent Core serves HTTPS with a self-signed cert.

    Without an unverified context every completion POST raised
    CERTIFICATE_VERIFY_FAILED, so no speak action ever completed and the ACP
    barrier waited out its full timeout on each utterance.
    """
    import ssl

    captured = {}

    def fake_urlopen(request, timeout=0, context=None):
        captured["url"] = request.full_url
        captured["context"] = context
        captured["body"] = request.data
        return _NullResponse()

    class _NullResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("AGENT_CORE_URL", "https://localhost:15678")

    vits2._complete_action("speak-1", "你好", 3, interrupted=False)

    assert captured["url"].endswith("/api/acp/complete")
    ctx = captured["context"]
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.verify_mode == ssl.CERT_NONE and ctx.check_hostname is False
    assert b'"status": "completed"' in captured["body"]


def test_load_error_keeps_the_underlying_cause(monkeypatch):
    """The dashboard must show why TensorRT was unavailable, not just that."""
    plugin, _, _, _ = _plugin(monkeypatch)

    import utils.model_downloader as md

    def explode(model_dir, family=None):
        try:
            raise ImportError("libnvdla_compiler.so: file too short")
        except ImportError as cause:
            raise RuntimeError("TensorRT is not available in this runtime") from cause

    monkeypatch.setattr(md, "ensure_vits2_model", explode, raising=False)
    plugin.dispatch("tts", {"action": "start", "input_topic": "/say"})

    assert _wait_until(
        lambda: plugin.dispatch("tts", {"action": "info"})["state"] == "error"
    )
    error = plugin.dispatch("tts", {"action": "info"})["error"]
    assert "TensorRT is not available" in error
    assert "libnvdla_compiler.so" in error

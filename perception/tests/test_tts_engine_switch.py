"""
tests/test_tts_engine_switch.py — the public tts tool's engine facade.

PR #112 introduced a second TTS implementation but left the engine selectable
only through the image's config.yaml, so the dashboard could neither show nor
change it. These tests pin the behaviour that fixes that: one tool, an engine
field in configSchema, and a switch that disposes the outgoing engine's nodes
before the incoming one publishes on the same topics.

Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import threading
import time

import pytest

from vision_stubs import _FakeExecutor, _wait_until  # noqa: F401

import plugins.tts as tts  # noqa: E402


class _FakeEngine:
    """Stands in for one engine implementation behind the facade."""

    def __init__(self, name, cfg, executor, record, delay=0.0):
        self.name = name
        self.cfg = dict(cfg)
        self.executor = executor
        self.calls = []
        self.stopped = False
        if delay:
            time.sleep(delay)
        record(self)

    def get_tools(self):
        return tts.TOOLS

    def dispatch(self, name, args):
        action = args.get("action")
        self.calls.append(args)
        if action == "stop":
            self.stopped = True
            return {"state": "idle"}
        if action == "info":
            return {"name": "TTS", "model": self.name, "state": "idle"}
        if action == "config":
            return {"status": "configured", "applied": dict(args)}
        return {"state": "running", "engine_seen": self.name}

    def synthesize_raw(self, text):
        return f"{self.name}:{text}".encode()


class _Engines:
    """Per-test record of what the facade built.

    Deliberately not class-level state on _FakeEngine: a switch left in flight
    by an earlier test would append to a shared list after the next test had
    cleared it, and that pollution looked exactly like a double build.
    """

    def __init__(self):
        self.order = []
        self.impls = {}

    def add(self, impl):
        self.order.append(impl.name)
        self.impls[impl.name] = impl

    def __contains__(self, name):
        return name in self.impls

    def __getitem__(self, name):
        return self.impls[name]


@pytest.fixture(autouse=True)
def _fake_engines(monkeypatch):
    """Replace both real engines with fakes; sherpa's is deliberately slow."""
    engines = _Engines()

    def build(self, engine):
        cfg = dict(self._cfg)
        cfg["engine"] = engine
        # sherpa-onnx really does fetch its Matcha model in __init__.
        delay = 0.3 if engine == "sherpa_onnx" else 0.0
        return _FakeEngine(engine, cfg, self._executor, engines.add, delay=delay)

    monkeypatch.setattr(tts.TTSPlugin, "_build", build)
    return engines


def _plugin(**cfg):
    return tts.TTSPlugin({"engine": "vits2_trt", **cfg}, _FakeExecutor())


def test_config_schema_exposes_the_engine_selector():
    props = tts.TOOLS[0]["configSchema"]["properties"]
    assert props["tts_engine"]["enum"] == list(tts.TTS_ENGINES)
    assert props["tts_engine"]["default"] == tts.DEFAULT_TTS_ENGINE
    # One public tool, whatever the engine.
    assert [t["name"] for t in tts.TOOLS] == ["tts"]
    assert tts.TTSPlugin.PREFIX == "tts"


def test_default_engine_is_built_at_startup(_fake_engines):
    plugin = _plugin()
    assert _fake_engines.order == ["vits2_trt"]
    assert plugin.dispatch("tts", {"action": "info"})["engine"] == "vits2_trt"


def test_actions_are_forwarded_to_the_active_engine(_fake_engines):
    plugin = _plugin()
    result = plugin.dispatch("tts", {"action": "start", "input_topic": "/say"})
    assert result["engine_seen"] == "vits2_trt"
    assert _fake_engines["vits2_trt"].calls[-1]["input_topic"] == "/say"


def test_switch_stops_the_old_engine_and_never_blocks(_fake_engines):
    plugin = _plugin()
    outgoing = _fake_engines["vits2_trt"]

    began = time.monotonic()
    result = plugin.dispatch("tts", {"action": "config", "tts_engine": "sherpa_onnx"})
    elapsed = time.monotonic() - began

    assert result["status"] == "configured"
    assert result["state"] == "loading"
    # sherpa-onnx downloads its model in the constructor; the config call must
    # not wait for that (Agent Core gives a processor call 60 s).
    assert elapsed < 0.2, f"config blocked for {elapsed:.2f}s"
    # The outgoing engine is stopped before the new one can publish.
    assert outgoing.stopped is True

    assert _wait_until(
        lambda: plugin.dispatch("tts", {"action": "info"}).get("model") == "sherpa_onnx"
    )
    assert plugin.dispatch("tts", {"action": "info"})["engine"] == "sherpa_onnx"


def test_info_reports_loading_while_the_new_engine_builds(_fake_engines):
    plugin = _plugin()
    plugin.dispatch("tts", {"action": "config", "tts_engine": "sherpa_onnx"})

    info = plugin.dispatch("tts", {"action": "info"})
    assert info["state"] == "loading"
    assert "sherpa_onnx" in info["desc"]
    # Other actions answer loading too, rather than hanging or lying.
    assert plugin.dispatch("tts", {"action": "start"})["state"] == "loading"


def test_switching_back_and_forth_keeps_one_engine_live(_fake_engines):
    plugin = _plugin()
    plugin.dispatch("tts", {"action": "config", "tts_engine": "sherpa_onnx"})
    assert _wait_until(lambda: "sherpa_onnx" in _fake_engines)
    assert _wait_until(
        lambda: plugin.dispatch("tts", {"action": "info"})["state"] != "loading"
    )
    sherpa = _fake_engines["sherpa_onnx"]

    plugin.dispatch("tts", {"action": "config", "tts_engine": "vits2_trt"})
    assert sherpa.stopped is True
    assert _wait_until(
        lambda: plugin.dispatch("tts", {"action": "info"})["state"] != "loading"
    )
    assert plugin.dispatch("tts", {"action": "info"})["engine"] == "vits2_trt"
    assert _fake_engines.order == ["vits2_trt", "sherpa_onnx", "vits2_trt"]


def test_reconfiguring_the_same_engine_does_not_rebuild(_fake_engines):
    plugin = _plugin()
    result = plugin.dispatch("tts", {"action": "config", "tts_engine": "vits2_trt",
                                     "speed": 1.3})
    assert _fake_engines.order == ["vits2_trt"], "same engine was rebuilt"
    # tts_engine is the facade's own field and must not be forwarded as if it
    # were an engine parameter; speed must be.
    applied = result["applied"]
    assert "tts_engine" not in applied and applied["speed"] == 1.3


def test_shared_config_survives_an_engine_switch(_fake_engines):
    plugin = _plugin()
    plugin.dispatch("tts", {"action": "config", "speed": 0.7})
    plugin.dispatch("tts", {"action": "config", "tts_engine": "sherpa_onnx"})
    assert _wait_until(lambda: "sherpa_onnx" in _fake_engines)
    assert _fake_engines["sherpa_onnx"].cfg["speed"] == 0.7


def test_unknown_engine_is_refused_without_touching_the_live_one(_fake_engines):
    plugin = _plugin()
    with pytest.raises(ValueError):
        plugin.dispatch("tts", {"action": "config", "tts_engine": "espeak"})
    assert _fake_engines["vits2_trt"].stopped is False
    assert plugin.dispatch("tts", {"action": "info"})["engine"] == "vits2_trt"


def test_engine_build_failure_is_reported_not_raised(monkeypatch):
    def boom(self, engine):
        raise RuntimeError("no TensorRT here")

    monkeypatch.setattr(tts.TTSPlugin, "_build", boom)
    plugin = _plugin()          # must not raise: main.py keeps the tool listed
    info = plugin.dispatch("tts", {"action": "info"})
    assert info["state"] == "error"
    assert "no TensorRT here" in info["error"]
    assert plugin.dispatch("tts", {"action": "start"})["state"] == "error"
    with pytest.raises(RuntimeError):
        plugin.synthesize_raw("hi")


def test_concurrent_switches_leave_exactly_one_engine_live(_fake_engines):
    plugin = _plugin()
    targets = ["sherpa_onnx", "vits2_trt", "sherpa_onnx", "vits2_trt"]
    threads = [
        threading.Thread(
            target=lambda e=e: plugin.dispatch(
                "tts", {"action": "config", "tts_engine": e}
            )
        )
        for e in targets
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert _wait_until(
        lambda: plugin.dispatch("tts", {"action": "info"}).get("state") != "loading",
        timeout=5.0,
    )
    info = plugin.dispatch("tts", {"action": "info"})
    assert info["engine"] == info["model"]
    # Every engine built but superseded must have been stopped, so no orphan
    # publisher survives on the shared topic.
    live = info["model"]
    for name, impl in _fake_engines.impls.items():
        if name != live:
            assert impl.stopped is True, f"{name} left running"

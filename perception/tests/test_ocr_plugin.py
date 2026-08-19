"""
tests/test_ocr_plugin.py — OCR plugin lifecycle/concurrency tests (host-side).

ROS stubs come from vision_stubs (installed by conftest before collection).
Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from vision_stubs import (  # noqa: F401
    _FakeCompressedImage,
    _FakeExecutor,
    _FakeNode,
    _FakeString,
    _wait_until,
)

from utils.qos import CAMERA_QOS  # noqa: E402
import plugins.ocr as ocr_plugin  # noqa: E402


class _SlowOCRAdapter:
    def __init__(self, delay=0.15):
        self.delay = delay
        self.seen = []
        self.closed = False

    def recognize(self, image_bytes: bytes, language: str = "zh") -> list:
        self.seen.append(image_bytes)
        time.sleep(self.delay)
        return [{"text": image_bytes.decode(), "bbox": [0, 0, 1, 1], "score": 0.99}]

    def close(self):
        self.closed = True


class _BuilderProbe:
    """Counts _build_ocr_adapter calls; optional delay simulates the real
    model download + TensorRT engine build."""

    def __init__(self, delay=0.0):
        self.delay = delay
        self.calls = 0
        self.built = []
        self.lock = threading.Lock()

    def __call__(self, cfg):
        with self.lock:
            self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        adapter = _SlowOCRAdapter(delay=0.01)
        adapter.cfg = dict(cfg)
        self.built.append(adapter)
        return adapter


def _make_plugin(monkeypatch, builder=None, cfg=None):
    builder = builder or _BuilderProbe()
    monkeypatch.setattr(ocr_plugin, "_build_ocr_adapter", builder)
    executor = _FakeExecutor()
    plugin = ocr_plugin.OCRPlugin(dict(cfg or {"provider": "rapidocr"}), executor)
    return plugin, executor, builder


def _start_and_wait(plugin, executor, topic, instance_id="", count=1):
    args = {"action": "start", "input_topic": topic}
    if instance_id:
        args["instance_id"] = instance_id
    plugin.dispatch("ocr", args)
    # Registration now precedes start (README lifecycle rule), so a node can
    # briefly sit in the executor as "idle"; wait for it to actually run.
    assert _wait_until(
        lambda: len(executor.nodes) >= count
        and all(n.state == "running" for n in executor.nodes)
    ), "node never came up"


@pytest.fixture
def ocr(monkeypatch):
    plugin, executor, builder = _make_plugin(monkeypatch)
    yield plugin, executor, builder
    plugin.dispatch("ocr", {"action": "stop"})


def test_ocr_camera_subscription_uses_shared_qos(ocr):
    plugin, executor, _ = ocr
    _start_and_wait(plugin, executor, "/cam/a")
    node = executor.nodes[0]
    assert node.subscriptions[0].qos is CAMERA_QOS


def test_ocr_topic_change_disposes_old_node(ocr):
    plugin, executor, _ = ocr
    _start_and_wait(plugin, executor, "/cam/a", instance_id="x")
    old = executor.nodes[0]
    plugin.dispatch("ocr", {"action": "start", "input_topic": "/cam/b", "instance_id": "x"})
    assert _wait_until(
        lambda: executor.nodes and executor.nodes[-1].subscriptions
        and executor.nodes[-1].subscriptions[0].topic == "/cam/b"
    )
    assert old not in executor.nodes and old.destroyed
    assert len(executor.nodes) == 1


def test_ocr_latest_frame_wins(ocr):
    plugin, executor, builder = ocr
    _start_and_wait(plugin, executor, "/cam/a")
    node = executor.nodes[0]
    adapter = builder.built[0]
    callback = node.subscriptions[0].callback
    callback(_FakeCompressedImage(b"f0"))
    assert _wait_until(lambda: adapter.seen == [b"f0"])
    for index in range(1, 6):
        callback(_FakeCompressedImage(f"f{index}".encode()))
    assert _wait_until(lambda: len(adapter.seen) >= 2)
    time.sleep(0.2)
    assert adapter.seen == [b"f0", b"f5"], adapter.seen
    payload = json.loads(node.publishers[0].messages[-1])
    assert payload["text"] == "f5"


def test_ocr_stop_wakes_worker_promptly(ocr):
    plugin, executor, _ = ocr
    _start_and_wait(plugin, executor, "/cam/a")
    node = executor.nodes[0]
    started = time.monotonic()
    node.stop()
    assert time.monotonic() - started < 1.0  # no 1s poll timeout wait
    assert not node.worker_alive


# ── OCR start/stop/load state machine ────────────────────────────────────────

def test_ocr_start_returns_loading_without_blocking(monkeypatch):
    plugin, executor, builder = _make_plugin(monkeypatch, _BuilderProbe(delay=1.0))
    started = time.monotonic()
    result = plugin.dispatch("ocr", {"action": "start", "input_topic": "/cam/a"})
    elapsed = time.monotonic() - started
    assert elapsed < 0.1, f"start blocked for {elapsed:.3f}s"
    assert result == {"state": "loading", "input": "/cam/a", "output": "/cam/a/ocr"}
    assert _wait_until(lambda: len(executor.nodes) == 1)
    plugin.dispatch("ocr", {"action": "stop"})


def test_ocr_ten_concurrent_starts_single_flight(monkeypatch):
    plugin, executor, builder = _make_plugin(monkeypatch, _BuilderProbe(delay=0.3))
    results = []

    def call(index):
        results.append(plugin.dispatch("ocr", {
            "action": "start",
            "input_topic": f"/cam/{index}",
            "instance_id": f"inst{index}",
        }))

    threads = [threading.Thread(target=call, args=(i,)) for i in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    assert all(res["state"] == "loading" for res in results)
    assert _wait_until(lambda: len(executor.nodes) == 10)
    assert builder.calls == 1, "ten starts must share one model load"
    plugin.dispatch("ocr", {"action": "stop"})


def test_ocr_info_does_not_block_while_loading(monkeypatch):
    plugin, executor, builder = _make_plugin(monkeypatch, _BuilderProbe(delay=0.8))
    plugin.dispatch("ocr", {"action": "start", "input_topic": "/cam/a", "instance_id": "a"})
    started = time.monotonic()
    info = plugin.dispatch("ocr", {"action": "info"})
    elapsed = time.monotonic() - started
    assert elapsed < 0.1, f"info blocked for {elapsed:.3f}s"
    assert info["state"] == "loading"
    assert info["instances"]["a"]["state"] == "loading"
    assert _wait_until(
        lambda: plugin.dispatch("ocr", {"action": "info"})["state"] == "running"
    )
    plugin.dispatch("ocr", {"action": "stop"})


def test_ocr_stop_during_loading_cancels_pending(monkeypatch):
    plugin, executor, builder = _make_plugin(monkeypatch, _BuilderProbe(delay=0.3))
    plugin.dispatch("ocr", {"action": "start", "input_topic": "/cam/a", "instance_id": "a"})
    plugin.dispatch("ocr", {"action": "start", "input_topic": "/cam/b", "instance_id": "b"})
    plugin.dispatch("ocr", {"action": "stop", "instance_id": "a"})
    assert _wait_until(lambda: len(executor.nodes) == 1)
    time.sleep(0.2)  # loader must not resurrect the stopped instance
    assert len(executor.nodes) == 1
    assert executor.nodes[0].subscriptions[0].topic == "/cam/b"
    assert plugin.dispatch(
        "ocr", {"action": "info", "instance_id": "a"}
    )["state"] == "idle"
    plugin.dispatch("ocr", {"action": "stop"})


def test_ocr_config_during_loading_discards_stale_adapter(monkeypatch):
    builder = _BuilderProbe(delay=0.3)
    plugin, executor, _ = _make_plugin(
        monkeypatch, builder, cfg={"provider": "rapidocr", "det_thresh": 0.3}
    )
    plugin.dispatch("ocr", {"action": "start", "input_topic": "/cam/a", "instance_id": "a"})
    plugin.dispatch("ocr", {"action": "config", "det_thresh": 0.9})
    assert _wait_until(lambda: len(executor.nodes) == 1, timeout=4)
    assert builder.calls == 2, "config change during load must trigger a reload"
    assert _wait_until(lambda: len(builder.built) == 2)
    stale = next(a for a in builder.built if a.cfg["det_thresh"] == 0.3)
    fresh = next(a for a in builder.built if a.cfg["det_thresh"] == 0.9)
    assert _wait_until(lambda: stale.closed), "stale adapter must be closed"
    assert plugin._adapter is fresh
    plugin.dispatch("ocr", {"action": "stop"})


def test_ocr_plain_stop_fully_disposes_node(ocr):
    plugin, executor, _ = ocr
    _start_and_wait(plugin, executor, "/cam/a", instance_id="a")
    node = executor.nodes[0]
    plugin.dispatch("ocr", {"action": "stop", "instance_id": "a"})
    assert executor.nodes == []
    assert node.destroyed
    assert not node.worker_alive
    assert plugin._nodes == {}


def test_ocr_repeated_start_stop_does_not_grow(ocr):
    plugin, executor, builder = ocr
    for _ in range(20):
        _start_and_wait(plugin, executor, "/cam/a", instance_id="a")
        plugin.dispatch("ocr", {"action": "stop", "instance_id": "a"})
    assert executor.nodes == []
    assert builder.calls == 1, "adapter must be loaded once and cached"
    assert threading.active_count() < 15
    created = [n for n in _FakeNode.instances if n.subscriptions]
    assert all(node.destroyed for node in created)


def test_ocr_repeated_start_is_idempotent(ocr):
    plugin, executor, _ = ocr
    _start_and_wait(plugin, executor, "/cam/a", instance_id="a")
    first = executor.nodes[0]
    result = plugin.dispatch("ocr", {"action": "start", "input_topic": "/cam/a", "instance_id": "a"})
    assert result["state"] == "running"
    assert executor.nodes == [first]


def test_ocr_stop_during_ready_path_start_leaves_no_orphan(monkeypatch):
    """PR #113's orphan-node class: a stop racing a fast-path start must not
    end with an invisible running node (stop must always see the instance in
    _pending_starts or _nodes)."""
    plugin, executor, builder = _make_plugin(monkeypatch)
    _start_and_wait(plugin, executor, "/cam/a", instance_id="a")  # adapter ready

    real_start = ocr_plugin._OCRNode.start

    def slow_start(self, *a, **k):
        time.sleep(0.2)
        return real_start(self, *a, **k)

    monkeypatch.setattr(ocr_plugin._OCRNode, "start", slow_start)
    thread = threading.Thread(target=plugin.dispatch, args=(
        "ocr", {"action": "start", "input_topic": "/cam/b", "instance_id": "b"}))
    thread.start()
    time.sleep(0.05)                       # start is inside node.start()
    plugin.dispatch("ocr", {"action": "stop", "instance_id": "b"})
    thread.join(timeout=3)
    monkeypatch.setattr(ocr_plugin._OCRNode, "start", real_start)

    assert "b" not in plugin._nodes
    assert plugin.dispatch("ocr", {"action": "info", "instance_id": "b"})["state"] == "idle"
    b_nodes = [n for n in _FakeNode.instances
               if n.subscriptions and n.subscriptions[0].topic == "/cam/b"]
    assert all(node.destroyed for node in b_nodes), "orphan node left running"
    plugin.dispatch("ocr", {"action": "stop"})


def test_ocr_info_desc_explains_loading_and_error(monkeypatch):
    """desc carries the reason while loading / after failure, like ASR (#113)."""
    builder = _BuilderProbe(delay=0.5)
    plugin, executor, _ = _make_plugin(monkeypatch, builder)
    plugin.dispatch("ocr", {"action": "start", "input_topic": "/cam/a", "instance_id": "a"})

    info = plugin.dispatch("ocr", {"action": "info"})
    assert info["state"] == "loading" and "Loading" in info["desc"]
    inst = plugin.dispatch("ocr", {"action": "info", "instance_id": "a"})
    assert "Loading" in inst["desc"]

    assert _wait_until(
        lambda: plugin.dispatch("ocr", {"action": "info"})["state"] == "running",
        timeout=4,
    )
    ready = plugin.dispatch("ocr", {"action": "info"})
    assert ready["state"] == "running" and ready["desc"] == plugin._DESC
    plugin.dispatch("ocr", {"action": "stop"})


def test_ocr_add_node_failure_leaks_nothing(monkeypatch):
    """If executor.add_node raises during bring-up, the node must be
    destroyed and untracked with no started worker (bot P1: a started but
    unregistered worker would be unreachable forever)."""
    plugin, executor, builder = _make_plugin(monkeypatch)

    original_add = executor.add_node
    calls = {"n": 0}

    def flaky_add(node):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("executor shutting down")
        return original_add(node)

    monkeypatch.setattr(executor, "add_node", flaky_add)
    plugin.dispatch("ocr", {"action": "start", "input_topic": "/cam/a", "instance_id": "a"})
    assert _wait_until(lambda: calls["n"] >= 1, timeout=4)
    time.sleep(0.1)

    assert "a" not in plugin._nodes
    assert "a" not in plugin._pending_starts, "failed instance stuck as loading"
    failed = [n for n in _FakeNode.instances
              if n.subscriptions == [] or (n.subscriptions and n.subscriptions[0].topic == "/cam/a")]
    assert all(n.destroyed for n in failed if not n.subscriptions), "unregistered node leaked"
    # worker was never started on the failed node (register-before-start)
    assert all(not n.subscriptions for n in _FakeNode.instances if n.destroyed)

    # the plugin recovers: a later start succeeds
    plugin.dispatch("ocr", {"action": "start", "input_topic": "/cam/a", "instance_id": "a"})
    assert _wait_until(lambda: "a" in plugin._nodes, timeout=4)
    plugin.dispatch("ocr", {"action": "stop"})


def test_ocr_load_failure_reports_error_and_retries(monkeypatch):
    calls = {"n": 0}

    def flaky_builder(cfg):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("download failed")
        return _SlowOCRAdapter(delay=0.01)

    monkeypatch.setattr(ocr_plugin, "_build_ocr_adapter", flaky_builder)
    executor = _FakeExecutor()
    plugin = ocr_plugin.OCRPlugin({"provider": "rapidocr"}, executor)
    plugin.dispatch("ocr", {"action": "start", "input_topic": "/cam/a", "instance_id": "a"})
    assert _wait_until(
        lambda: plugin.dispatch("ocr", {"action": "info"})["state"] == "error"
    )
    info = plugin.dispatch("ocr", {"action": "info", "instance_id": "a"})
    assert info["state"] == "error" and "download failed" in info.get("error", "")
    # a later start retries the load and succeeds
    plugin.dispatch("ocr", {"action": "start", "input_topic": "/cam/a", "instance_id": "a"})
    assert _wait_until(
        lambda: plugin.dispatch("ocr", {"action": "info"})["state"] == "running"
    )
    plugin.dispatch("ocr", {"action": "stop"})




def test_ocr_model_dir_confined_to_models_tree(monkeypatch):
    """MCP-supplied model_dir cannot point the root-run downloader outside
    /models (bot P2): the wrapper rejects out-of-tree paths before download."""
    import utils.model_downloader as md
    with pytest.raises(ValueError):
        md.ensure_ocr_model("/etc/cron.d")
    with pytest.raises(ValueError):
        md.ensure_ocr_model("/models/../root")

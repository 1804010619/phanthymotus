"""
tests/test_visual_depth_plugin.py — depth encoding, region summary, and lifecycle
(host-side, no GPU).

The encoding tests are the load-bearing ones: agent-core's depth renderer
hardcodes a 640x480 canvas and silently drops any frame shorter than that, so
a regression in the resample path shows up as a blank dashboard panel with
nothing in any log.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest perception/tests -q
"""

from __future__ import annotations

import json
import threading
import time
import zlib

import numpy as np
import pytest

from vision_stubs import (  # noqa: F401
    _FakeCompressedImage,
    _FakeExecutor,
    _FakeNode,
    _wait_until,
)

import plugins.visual_depth as depth_plugin  # noqa: E402
from plugins.vision_runtime import LetterboxMeta  # noqa: E402

W, H = depth_plugin.DEPTH_WIDTH, depth_plugin.DEPTH_HEIGHT


class _FakeModel:
    """Stands in for VisionEngineSession: infer(frame) -> (outputs, meta)."""

    def __init__(self, depth=None, shape=(H, W)):
        self.calls = 0
        self._depth = depth if depth is not None else np.full(shape, 2.0, dtype=np.float32)

    @property
    def input_size(self):
        return (self._depth.shape[1], self._depth.shape[0])

    def infer(self, frame):
        self.calls += 1
        # Identity letterbox sized to the engine output, so decode_depth's crop
        # is a no-op and the tests assert on the plugin's own behaviour.
        meta = LetterboxMeta(1.0, 0, 0, self._depth.shape[1], self._depth.shape[0])
        return [self._depth], meta


def _plugin(cfg=None, model=None):
    executor = _FakeExecutor()
    plugin = depth_plugin.VideoDepthPerceptionPlugin(cfg or {}, "testns", executor)
    if model is not None:
        plugin._model = model
    return plugin, executor


# ── encoding: the renderer contract ──────────────────────────────────────────

def test_encoded_depth_decompresses_to_exactly_640x480_uint16():
    depth = np.full((H, W), 1.234, dtype=np.float32)
    raw = zlib.decompress(depth_plugin.encode_depth(depth, max_depth_m=20.0))
    values = np.frombuffer(raw, dtype="<u2")
    # Shorter than this and DepthZlibRenderer returns early, rendering nothing.
    assert values.size == W * H
    assert values[0] == 1234          # metres → millimetres


def test_wrong_shape_is_refused_rather_than_published():
    """Publishing a mis-sized map produces a blank panel and no log line."""
    with pytest.raises(ValueError, match="640x480"):
        depth_plugin.encode_depth(np.zeros((480, 512), dtype=np.float32), max_depth_m=20.0)


def test_out_of_range_becomes_invalid_not_wrapped():
    """A 70 m reading must not come out as an obstacle at arm's length."""
    depth = np.full((H, W), 70.0, dtype=np.float32)
    values = np.frombuffer(zlib.decompress(depth_plugin.encode_depth(depth, max_depth_m=20.0)), dtype="<u2")
    assert set(values.tolist()) == {0}


def test_sub_millimetre_and_nonfinite_become_invalid():
    depth = np.zeros((H, W), dtype=np.float32)
    depth[0, 0] = np.nan
    depth[0, 1] = np.inf
    depth[0, 2] = 0.0001          # rounds below 1 mm
    depth[0, 3] = 5.0
    values = np.frombuffer(zlib.decompress(depth_plugin.encode_depth(depth, max_depth_m=20.0)), dtype="<u2")
    assert values[0] == 0 and values[1] == 0 and values[2] == 0
    assert values[3] == 5000


def test_max_depth_ceiling_is_honoured():
    depth = np.full((H, W), 15.0, dtype=np.float32)
    kept = np.frombuffer(zlib.decompress(depth_plugin.encode_depth(depth, max_depth_m=20.0)), dtype="<u2")
    dropped = np.frombuffer(zlib.decompress(depth_plugin.encode_depth(depth, max_depth_m=10.0)), dtype="<u2")
    assert kept[0] == 15000
    assert dropped[0] == 0


# ── summary ──────────────────────────────────────────────────────────────────

def test_summary_reports_nearest_per_region():
    depth = np.full((H, W), 9.0, dtype=np.float32)
    depth[:, : W // 3] = 1.0            # something close on the left
    summary = depth_plugin.summarize_depth(depth, "metric")
    assert summary["nearest_by_region"]["left"] == pytest.approx(1.0)
    assert summary["nearest_by_region"]["center"] == pytest.approx(9.0)
    assert summary["nearest_by_region"]["right"] == pytest.approx(9.0)


def test_summary_ignores_a_few_outlier_pixels():
    """The 5th percentile is the point: edge artefacts must not invent an obstacle."""
    depth = np.full((H, W), 5.0, dtype=np.float32)
    depth[0, :10] = 0.01               # a handful of edge artefacts
    summary = depth_plugin.summarize_depth(depth, "metric")
    assert summary["nearest_by_region"]["left"] == pytest.approx(5.0)


def test_summary_marks_uncalibrated_output_as_relative():
    """An agent reading relative numbers as metres is the failure to prevent."""
    summary = depth_plugin.summarize_depth(np.full((H, W), 2.0, dtype=np.float32), "relative")
    assert summary["scale"] == "relative"
    assert summary["unit"] == "relative"


def test_summary_handles_an_all_invalid_map():
    summary = depth_plugin.summarize_depth(np.zeros((H, W), dtype=np.float32), "metric")
    assert summary["nearest_by_region"] == {"left": None, "center": None, "right": None}
    assert summary["range"] is None
    assert summary["valid_fraction"] == 0.0


def test_valid_fraction_reflects_partial_coverage():
    depth = np.zeros((H, W), dtype=np.float32)
    depth[: H // 2] = 3.0
    assert depth_plugin.summarize_depth(depth, "metric")["valid_fraction"] == pytest.approx(0.5)


# ── pipeline ─────────────────────────────────────────────────────────────────

def _feed(node, marker=b"640x480"):
    node._image_cb(_FakeCompressedImage(marker))


def test_both_topics_are_published_for_one_frame():
    plugin, executor = _plugin(model=_FakeModel())
    plugin.dispatch("visual_depth", {"action": "start", "input_topic": "/cam/rgb"})
    node = executor.nodes[0]
    _feed(node)

    depth_pub = next(p for p in node.publishers if p.topic.endswith("/depth"))
    summary_pub = next(p for p in node.publishers if p.topic.endswith("/depth_summary"))
    assert _wait_until(lambda: depth_pub.messages and summary_pub.messages)

    values = np.frombuffer(zlib.decompress(depth_pub.messages[0]), dtype="<u2")
    assert values.size == W * H
    assert json.loads(summary_pub.messages[0])["scale"] == "relative"


def test_model_output_is_resampled_to_the_renderer_size():
    """The model runs at its own resolution; the renderer only accepts 640x480."""
    plugin, executor = _plugin(model=_FakeModel(depth=np.full((768, 768), 3.0, dtype=np.float32)))
    plugin.dispatch("visual_depth", {"action": "start", "input_topic": "/cam/rgb"})
    node = executor.nodes[0]
    _feed(node)

    depth_pub = next(p for p in node.publishers if p.topic.endswith("/depth"))
    assert _wait_until(lambda: bool(depth_pub.messages))
    values = np.frombuffer(zlib.decompress(depth_pub.messages[0]), dtype="<u2")
    assert values.size == W * H
    assert values[0] == 3000


def test_depth_scale_is_applied():
    plugin, executor = _plugin(cfg={"depth_scale": 2.0, "calibrated": True},
                               model=_FakeModel(depth=np.full((H, W), 1.0, dtype=np.float32)))
    plugin.dispatch("visual_depth", {"action": "start", "input_topic": "/cam/rgb"})
    node = executor.nodes[0]
    _feed(node)

    depth_pub = next(p for p in node.publishers if p.topic.endswith("/depth"))
    assert _wait_until(lambda: bool(depth_pub.messages))
    values = np.frombuffer(zlib.decompress(depth_pub.messages[0]), dtype="<u2")
    assert values[0] == 2000          # 1.0 * 2.0 m → 2000 mm


# ── lifecycle ────────────────────────────────────────────────────────────────

def test_info_warns_while_uncalibrated():
    plugin, _ = _plugin()
    info = plugin.dispatch("visual_depth", {"action": "info"})
    assert info["scale"] == "relative"
    assert "RELATIVE" in info["warning"]


def test_info_drops_the_warning_once_calibrated():
    plugin, _ = _plugin(cfg={"calibrated": True})
    info = plugin.dispatch("visual_depth", {"action": "info"})
    assert info["scale"] == "metric"
    assert "warning" not in info


def test_info_advertises_both_output_formats():
    plugin, executor = _plugin(model=_FakeModel())
    plugin.dispatch("visual_depth", {"action": "start", "input_topic": "/cam/rgb"})
    info = plugin.dispatch("visual_depth", {"action": "info"})
    formats = {t["format"] for t in info["topic_out"]}
    assert formats == {"image/depth-zlib", "data/json"}


def test_start_then_stop_destroys_the_node():
    plugin, executor = _plugin(model=_FakeModel())
    plugin.dispatch("visual_depth", {"action": "start", "input_topic": "/cam/rgb"})
    node = executor.nodes[0]
    plugin.dispatch("visual_depth", {"action": "stop", "instance_id": "/cam/rgb"})
    assert executor.nodes == []
    assert node.destroyed is True


def test_concurrent_starts_create_exactly_one_node():
    plugin, executor = _plugin(model=_FakeModel())
    barrier = threading.Barrier(8)

    def _start():
        barrier.wait()
        plugin.dispatch("visual_depth", {"action": "start", "input_topic": "/cam/rgb"})

    threads = [threading.Thread(target=_start) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert len(executor.nodes) == 1


def test_config_updates_global_defaults():
    plugin, _ = _plugin()
    plugin.dispatch("visual_depth", {"action": "config", "fps": 7, "calibrated": True, "max_depth_m": 5.0})
    assert plugin._fps == 7
    assert plugin._calibrated is True
    assert plugin._max_depth_m == 5.0


def test_a_frame_that_fails_to_decode_is_skipped_not_fatal():
    # fps high enough that the rate limiter does not swallow the second frame —
    # at the default 2 fps the good frame lands inside the 500 ms window and is
    # dropped, which would make this pass or fail for the wrong reason.
    plugin, executor = _plugin(cfg={"fps": 1000}, model=_FakeModel())
    plugin.dispatch("visual_depth", {"action": "start", "input_topic": "/cam/rgb"})
    node = executor.nodes[0]
    _feed(node, b"not-a-frame")
    time.sleep(0.01)          # clear the 1 ms rate-limit window between frames
    _feed(node, b"640x480")

    depth_pub = next(p for p in node.publishers if p.topic.endswith("/depth"))
    assert _wait_until(lambda: bool(depth_pub.messages))

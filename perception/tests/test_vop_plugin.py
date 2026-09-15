"""
tests/test_vop_plugin.py — vop lifecycle, model-name aliasing, and the
frozen-vocabulary rejection paths (host-side, no GPU).

ROS stubs come from vision_stubs (installed by conftest before collection).
Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest perception/tests -q
"""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest

from vision_stubs import (  # noqa: F401
    _FakeCompressedImage,
    _FakeExecutor,
    _FakeNode,
    _wait_until,
)

import plugins.vop as vop_plugin  # noqa: E402
from plugins.vision_runtime import LetterboxMeta  # noqa: E402


class _FakeModel:
    """Stands in for VisionEngineSession: infer(frame) -> (outputs, meta)."""

    def __init__(self, rows=()):
        self.calls = 0
        # (1, N, 6) of [x1, y1, x2, y2, score, class], the NMS-free head layout.
        self._output = np.array(list(rows), dtype=np.float32).reshape(1, -1, 6)

    @property
    def input_size(self):
        return (640, 640)

    def infer(self, frame):
        self.calls += 1
        h, w = frame.shape[:2]
        # Identity letterbox keeps the coordinate assertions readable.
        return [self._output], LetterboxMeta(1.0, 0, 0, w, h)


def _plugin(cfg=None, model=None):
    executor = _FakeExecutor()
    plugin = vop_plugin.VideoObjectPerceptionPlugin(cfg or {}, "testns", executor)
    if model is not None:
        plugin._model = model
    plugin._vocabulary = ["person", "door", "forklift"]
    return plugin, executor


# ── model name aliasing ──────────────────────────────────────────────────────

@pytest.mark.parametrize("configured,expected", [
    ("yolov8s-worldv2", "yoloe-26s-seg"),
    ("yolov8s-worldv2.pt", "yoloe-26s-seg"),
    ("yolov8s-world", "yoloe-26s-seg"),
    ("yoloe-26s", "yoloe-26s-seg"),
    ("yoloe-26s-seg", "yoloe-26s-seg"),
    ("", "yoloe-26s-seg"),
])
def test_legacy_model_names_still_resolve(configured, expected):
    """A card saved before the model switch must not become state: error."""
    assert vop_plugin.canonical_model_name(configured) == expected


def test_unknown_model_name_is_left_alone():
    """An explicit engine path is a dev-box escape hatch, not an alias miss."""
    assert vop_plugin.canonical_model_name("/models/custom.engine") == "/models/custom"


# ── frozen vocabulary ────────────────────────────────────────────────────────

def test_set_classes_is_refused_with_the_vocabulary_named():
    plugin, _ = _plugin()
    with pytest.raises(ValueError) as excinfo:
        plugin.dispatch("vop", {"action": "set_classes", "classes": ["zebra crossing"]})
    message = str(excinfo.value)
    assert "zebra crossing" in message          # what was asked for
    assert "frozen" in message                  # why it failed
    assert "person" in message                  # what it can do instead


def test_config_with_classes_is_refused_wholesale():
    """Half-applying a config is the silent failure this guards against."""
    plugin, _ = _plugin()
    with pytest.raises(ValueError):
        plugin.dispatch("vop", {"action": "config", "classes": ["forklift"], "fps": 9})
    # fps must NOT have been applied — the call was rejected, not partially done.
    assert plugin._fps == 5


def test_config_without_classes_still_works():
    plugin, _ = _plugin()
    result = plugin.dispatch("vop", {"action": "config", "fps": 9, "confidence": 0.7})
    assert result["status"] == "configured"
    assert plugin._fps == 9
    assert plugin._confidence == 0.7


def test_yaml_configured_classes_surface_in_info():
    """Config that cannot be honoured has to be visible, not swallowed."""
    plugin, _ = _plugin(cfg={"classes": ["forklift", "pallet"]})
    info = plugin.dispatch("vop", {"action": "info"})
    assert info["ignored_config_classes"] == ["forklift", "pallet"]
    assert "frozen" in info["warning"]
    assert info["vocabulary_frozen"] is True


def test_info_reports_the_engine_vocabulary():
    plugin, _ = _plugin()
    info = plugin.dispatch("vop", {"action": "info"})
    assert info["total_classes"] == 3
    assert info["classes"] == ["person", "door", "forklift"]
    assert info["classes_loaded"] is True
    assert "warning" not in info          # nothing was rejected in this one


def test_info_before_the_vocabulary_is_known_does_not_claim_zero():
    """A freshly deployed robot must not show a vop card reading "0 classes".

    The engine loads lazily on first start, so until then the list is unknown —
    which is not the same as empty, and an operator reads 0 as "detects
    nothing" and files a bug.
    """
    plugin, _ = _plugin()
    plugin._vocabulary = []
    info = plugin.dispatch("vop", {"action": "info"})
    assert info["total_classes"] is None
    assert info["classes_loaded"] is False


def test_rejection_message_does_not_say_zero_classes():
    plugin, _ = _plugin()
    plugin._vocabulary = []
    message = plugin._frozen_vocab_error(["forklift"])
    assert "0 classes" not in message
    assert "not loaded yet" in message


# ── vocab.json reading ───────────────────────────────────────────────────────

def test_vocab_json_is_read_from_the_bundle(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"classes": ["a", "b"]}), encoding="utf-8")
    assert vop_plugin.VideoObjectPerceptionPlugin._read_vocab(str(path)) == ["a", "b"]


def test_vocab_json_accepts_a_bare_list(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(["a", "b"]), encoding="utf-8")
    assert vop_plugin.VideoObjectPerceptionPlugin._read_vocab(str(path)) == ["a", "b"]


@pytest.mark.parametrize("content", ["not json at all", json.dumps({"nope": 1})])
def test_unreadable_vocab_is_not_fatal(tmp_path, content):
    """The engine still detects what it detects; only the labels are lost."""
    path = tmp_path / "vocab.json"
    path.write_text(content, encoding="utf-8")
    assert vop_plugin.VideoObjectPerceptionPlugin._read_vocab(str(path)) == []


def test_missing_vocab_is_not_fatal():
    assert vop_plugin.VideoObjectPerceptionPlugin._read_vocab(None) == []
    assert vop_plugin.VideoObjectPerceptionPlugin._read_vocab("/nope/vocab.json") == []


# ── lifecycle / concurrency ──────────────────────────────────────────────────

def test_start_then_stop_retires_the_node():
    plugin, executor = _plugin(model=_FakeModel())
    plugin.dispatch("vop", {"action": "start", "input_topic": "/cam/rgb"})
    assert len(executor.nodes) == 1
    node = executor.nodes[0]

    plugin.dispatch("vop", {"action": "stop", "instance_id": "/cam/rgb"})
    assert executor.nodes == []
    # destroy_node(), not just remove_node() — otherwise the publisher and the
    # ROS node name leak and the next start trips "already registered".
    assert node.destroyed is True
    assert plugin._nodes == {}


def test_concurrent_starts_create_exactly_one_node():
    """Two HTTP threads racing on start must not orphan a live node."""
    plugin, executor = _plugin(model=_FakeModel())
    barrier = threading.Barrier(8)

    def _start():
        barrier.wait()
        plugin.dispatch("vop", {"action": "start", "input_topic": "/cam/rgb"})

    threads = [threading.Thread(target=_start) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert len(executor.nodes) == 1
    assert len(plugin._nodes) == 1


def test_stop_all_retires_every_instance():
    plugin, executor = _plugin(model=_FakeModel())
    for topic in ("/cam/a", "/cam/b", "/cam/c"):
        plugin.dispatch("vop", {"action": "start", "input_topic": topic})
    assert len(executor.nodes) == 3

    result = plugin.dispatch("vop", {"action": "stop"})
    assert sorted(result["stopped_instances"]) == ["/cam/a", "/cam/b", "/cam/c"]
    assert executor.nodes == []


def test_stop_of_an_unknown_instance_is_idle_not_an_error():
    plugin, _ = _plugin(model=_FakeModel())
    assert plugin.dispatch("vop", {"action": "stop", "instance_id": "/nope"}) == {"state": "idle"}


def test_repeated_start_stop_cycles_leave_nothing_behind():
    """The canvas really does config→start→stop→start within seconds."""
    plugin, executor = _plugin(model=_FakeModel())
    for _ in range(10):
        plugin.dispatch("vop", {"action": "start", "input_topic": "/cam/rgb"})
        plugin.dispatch("vop", {"action": "stop", "instance_id": "/cam/rgb"})
    assert executor.nodes == []
    assert plugin._nodes == {}


# ── detection output ─────────────────────────────────────────────────────────

def _node_with(plugin, executor):
    plugin.dispatch("vop", {"action": "start", "input_topic": "/cam/rgb"})
    return executor.nodes[0]


def test_objects_are_published_with_center_relative_coordinates():
    # 200x100 frame, box centered at (150, 25) → x_norm=+0.5, y_norm=-0.5
    plugin, executor = _plugin(model=_FakeModel())
    node = _node_with(plugin, executor)

    boxes = np.array([[100.0, 0.0, 200.0, 50.0]], dtype=np.float32)
    objects = node._extract_objects(boxes, np.array([0.9]), np.array([0]), (100, 200, 3))
    assert objects == [{"name": "person", "position": [0.5, -0.5], "confidence": 0.9}]


def test_class_ids_resolve_through_the_frozen_vocabulary():
    plugin, executor = _plugin(model=_FakeModel())   # vocab: person, door, forklift
    node = _node_with(plugin, executor)
    assert node._class_name(1) == "door"
    assert node._class_name(2) == "forklift"


def test_an_unresolvable_class_id_stays_numeric():
    """Better a useless label than a confidently wrong one."""
    plugin, executor = _plugin(model=_FakeModel())
    node = _node_with(plugin, executor)
    assert node._class_name(99) == "99"


def test_the_full_frame_path_publishes_decoded_objects():
    """End to end through infer → decode_detections → publish."""
    plugin, executor = _plugin(
        model=_FakeModel(rows=[[100.0, 0.0, 200.0, 50.0, 0.9, 1]])
    )
    node = _node_with(plugin, executor)
    node._image_cb(_FakeCompressedImage(b"200x100"))

    pub = node.publishers[0]
    assert _wait_until(lambda: bool(pub.messages))
    payload = json.loads(pub.messages[0])
    assert payload["objects"] == [
        {"name": "door", "position": [0.5, -0.5], "confidence": 0.9}
    ]


def test_low_confidence_detections_are_dropped():
    plugin, executor = _plugin(
        model=_FakeModel(rows=[[10.0, 10.0, 20.0, 20.0, 0.05, 0]])
    )
    node = _node_with(plugin, executor)
    node._image_cb(_FakeCompressedImage(b"200x100"))

    pub = node.publishers[0]
    assert _wait_until(lambda: bool(pub.messages))
    assert json.loads(pub.messages[0])["objects"] == []


# ── discoverability ──────────────────────────────────────────────────────────

def test_description_names_the_vocabulary_once_it_is_known():
    """The dashboard renders `description` and drops the rest of info.

    canvas.js keeps only `topic_out` from an info payload, so the `classes`
    list never reaches the card. Removing the `classes` config field took away
    the last thing the UI showed about what vop detects — the description is
    the one channel left.
    """
    plugin, _ = _plugin()          # vocabulary: person, door, forklift
    description = plugin.get_tools()[0]["description"]
    assert "3 classes" in description
    assert "person" in description
    assert "cannot be changed at runtime" in description


def test_description_falls_back_before_the_vocabulary_arrives():
    plugin, _ = _plugin()
    plugin._vocabulary = []
    assert plugin.get_tools() is vop_plugin.TOOLS


def test_get_tools_does_not_mutate_the_module_level_TOOLS():
    """A per-call rewrite must not leave the shared dict edited."""
    original = vop_plugin.TOOLS[0]["description"]
    plugin, _ = _plugin()
    plugin.get_tools()
    assert vop_plugin.TOOLS[0]["description"] == original

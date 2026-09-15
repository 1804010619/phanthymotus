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


# ── one-shot recognition ─────────────────────────────────────────────────────

def _photo_plugin(tmp_path, rows=(), **cfg):
    """A plugin whose image roots are tmp_path and whose engine is already up."""
    base = {"image_roots": [str(tmp_path)], "max_image_bytes": 1 << 20}
    base.update(cfg)
    plugin, executor = _plugin(cfg=base, model=_FakeModel(rows=rows))
    return plugin, executor


def _write_frame(tmp_path, name="scene.jpg", marker=b"200x100"):
    path = tmp_path / name
    path.write_bytes(marker)
    return str(path)


def test_recognize_by_photo_reports_objects_without_any_instance(tmp_path):
    """The point of the action: answer about a picture with no camera running."""
    plugin, executor = _photo_plugin(
        tmp_path, rows=[[100.0, 0.0, 200.0, 50.0, 0.9, 1]])
    result = plugin.dispatch("vop", {
        "action": "recognize_by_photo", "image_path": _write_frame(tmp_path)})

    assert executor.nodes == []          # nothing was started
    assert result["ok"] is True
    assert result["count"] == 1
    obj = result["objects"][0]
    assert obj["name"] == "door"         # vocab index 1
    assert obj["position"] == [0.5, -0.5]
    assert obj["confidence"] == 0.9
    # The caller cannot see the frame, so the pixel box is the only way back
    # to where in the image this was.
    assert obj["bbox"] == [100.0, 0.0, 200.0, 50.0]
    assert result["image_size"] == [200, 100]


def test_recognize_by_photo_honours_a_per_call_confidence(tmp_path):
    plugin, _ = _photo_plugin(tmp_path, rows=[[10.0, 10.0, 20.0, 20.0, 0.4, 0]])
    photo = _write_frame(tmp_path)

    loose = plugin.dispatch("vop", {"action": "recognize_by_photo",
                                    "image_path": photo, "confidence": 0.2})
    strict = plugin.dispatch("vop", {"action": "recognize_by_photo",
                                     "image_path": photo, "confidence": 0.8})
    assert loose["count"] == 1
    assert strict["count"] == 0
    assert strict["confidence_threshold"] == 0.8


def test_recognize_by_photo_falls_back_to_the_card_confidence(tmp_path):
    plugin, _ = _photo_plugin(tmp_path, rows=[], confidence=0.55)
    result = plugin.dispatch("vop", {"action": "recognize_by_photo",
                                     "image_path": _write_frame(tmp_path)})
    assert result["confidence_threshold"] == 0.55


def test_recognize_by_photo_refuses_a_path_outside_the_roots(tmp_path):
    plugin, _ = _photo_plugin(tmp_path)
    result = plugin.dispatch("vop", {"action": "recognize_by_photo",
                                     "image_path": "/etc/passwd"})
    assert result["ok"] is False
    assert result["reason"] == "bad_input"
    # Must name vop's own action, not "the _by_url action".
    assert "recognize_by_url" in result["detail"]


def test_recognize_by_photo_on_an_undecodable_file(tmp_path):
    plugin, _ = _photo_plugin(tmp_path)
    path = _write_frame(tmp_path, "junk.jpg", b"this is not an image")
    result = plugin.dispatch("vop", {"action": "recognize_by_photo",
                                     "image_path": path})
    assert result["ok"] is False
    assert "decode" in result["detail"]


def test_recognize_by_url_shares_the_photo_path(tmp_path, monkeypatch):
    import plugins.image_input as image_input
    monkeypatch.setattr(image_input, "fetch_url", lambda url, max_bytes: b"200x100")
    plugin, _ = _photo_plugin(tmp_path, rows=[[100.0, 0.0, 200.0, 50.0, 0.9, 0]])
    result = plugin.dispatch("vop", {"action": "recognize_by_url",
                                     "url": "https://example.com/a.jpg"})
    assert result["ok"] is True
    assert result["source"] == "https://example.com/a.jpg"
    assert result["objects"][0]["name"] == "person"


# ── list_recognizable_objects ────────────────────────────────────────────────

def test_list_recognizable_objects_returns_the_frozen_vocabulary():
    plugin, _ = _plugin()
    result = plugin.dispatch("vop", {"action": "list_recognizable_objects"})
    assert result["ok"] is True
    assert result["count"] == 3
    assert result["objects"] == ["person", "door", "forklift"]
    assert result["frozen"] is True
    # The note is the actionable half: an agent that reads "not listed here
    # will never be detected" stops rephrasing and reports the limit.
    assert "cannot be changed at runtime" in result["note"]


def test_list_recognizable_objects_needs_no_engine():
    """It answers from the prefetched vocab.json, not from a loaded engine."""
    plugin, executor = _plugin()      # no model installed
    assert plugin._model is None
    assert plugin.dispatch("vop", {"action": "list_recognizable_objects"})["ok"] is True
    assert executor.nodes == []


def test_list_recognizable_objects_says_so_when_it_does_not_know_yet():
    plugin, _ = _plugin()
    plugin._vocabulary = []
    result = plugin.dispatch("vop", {"action": "list_recognizable_objects"})
    assert result["ok"] is False
    assert result["reason"] == "vocabulary_unavailable"


def test_the_new_actions_are_advertised_with_their_params():
    schema = vop_plugin.TOOLS[0]["inputSchema"]
    actions = set(schema["properties"]["action"]["enum"])
    assert {"recognize_by_photo", "recognize_by_url",
            "list_recognizable_objects"} <= actions
    params = schema["x-action-params"]
    assert params["recognize_by_photo"]["params"] == ["image_path", "confidence"]
    assert params["recognize_by_url"]["params"] == ["url", "confidence"]
    # The card renders a file picker only for format=file + uploadTo=mcp.
    image_path = schema["properties"]["image_path"]
    assert image_path["format"] == "file"
    assert image_path["uploadTo"] == "mcp"


def test_start_without_a_topic_points_at_the_actions_that_need_none():
    """The instinct is to `start` first; the message has to redirect it.

    A bare "input_topic is required" reads as a broken tool to an agent that
    only wanted to look at one picture.
    """
    plugin, _ = _plugin(model=_FakeModel())
    with pytest.raises(ValueError) as excinfo:
        plugin.dispatch("vop", {"action": "start"})
    message = str(excinfo.value)
    assert "recognize_by_photo" in message
    assert "recognize_by_url" in message
    assert "list_recognizable_objects" in message

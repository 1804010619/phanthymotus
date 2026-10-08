"""
tests/test_pose_stgcn.py — the ST-GCN++ skeleton-action backend.

The engine is injected, so everything here runs on a laptop with no TensorRT,
no checkpoint and no GPU. What is covered is the part that fails *silently* if
it is wrong: the normalisation arithmetic, the uniform sampling, the tensor
layout, the probability/logit handling, the NTU-60 label mapping, and the
failure paths that must not turn into a label.

What is NOT covered, and cannot be from here: whether ST-GCN++ actually agrees
with these labels on a real robot. There is no published engine yet. Treat a
green run as "the plumbing is right", not as "the model works".

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest perception/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

import vision_stubs  # noqa: F401

from plugins.pose_action import (  # noqa: E402
    ACTION_PRIORITY,
    PoseActionClassifier,
    PoseFrame,
)
from plugins.pose_stgcn import (  # noqa: E402
    DEFAULT_WINDOW_FRAMES,
    NTU60_TO_ACTION,
    NTU60_TRANSITIONS,
    ActionBackendError,
    HybridActionBackend,
    SkeletonActionBackend,
    build_backend,
    looks_like_probabilities,
    pre_normalize_2d,
    softmax,
    uniform_sample_indices,
)
from plugins.vision_runtime import N_KEYPOINTS  # noqa: E402
from test_pose_action import (  # noqa: E402
    CX, H, TOP, _body, _kp, _lying_body, _lying_box, _standing_box,
)

FRAME = (1280, 720)
N_CLASSES = 60


class _FakeEngine:
    """Stands in for TensorRTEngine: infer(blob) -> per-class scores."""

    def __init__(self, scores=None, *, window_frames=DEFAULT_WINDOW_FRAMES,
                 outputs=None, raise_on_infer=None):
        self.input_shape = (1, 1, window_frames, N_KEYPOINTS, 2)
        self.calls = []
        self._raise = raise_on_infer
        if outputs is not None:
            self._outputs = outputs
        else:
            values = np.zeros(N_CLASSES, dtype=np.float32)
            for index, value in (scores or {}).items():
                values[index] = value
            self._outputs = [values[None]]

    def infer(self, blob):
        if self._raise:
            raise self._raise
        self.calls.append(np.asarray(blob).copy())
        return self._outputs


def _frames(joints_fn, *, n=30, fps=10.0, box=None, image_size=FRAME):
    out = []
    for i in range(n):
        t = i / fps
        out.append(PoseFrame(t, box or _standing_box(), _kp(**joints_fn(t)),
                             0.3, image_size=image_size))
    return out


def _standing(_t):
    return _body()


# ── preprocessing: the part that fails silently ─────────────────────────────

def test_normalisation_maps_the_frame_onto_minus_one_to_one():
    """PYSKL normalises by the FRAME, not by the person's box.

    Where someone stands and how large they appear within the frame is
    information the network trained with; re-centring on the body would discard
    it. Pinned on exact corners because a wrong scale raises nothing — it just
    feeds the model a skeleton from a distribution it never saw.
    """
    corners = np.array([[[0.0, 0.0], [1280.0, 720.0], [640.0, 360.0]]],
                       dtype=np.float32)
    out = pre_normalize_2d(corners, FRAME)
    assert out[0, 0].tolist() == pytest.approx([-1.0, -1.0])
    assert out[0, 1].tolist() == pytest.approx([1.0, 1.0])
    assert out[0, 2].tolist() == pytest.approx([0.0, 0.0])


def test_normalisation_refuses_a_zero_frame_size():
    with pytest.raises(ActionBackendError, match="unusable"):
        pre_normalize_2d(np.zeros((1, 17, 2), dtype=np.float32), (0, 720))


def test_uniform_sampling_covers_the_clip_evenly():
    """PYSKL's UniformSample, made deterministic: bin the clip and take each
    bin's centre. Random offsets are for training; a robot wants the same
    answer twice for the same input."""
    assert uniform_sample_indices(48, 48).tolist() == list(range(48))
    short = uniform_sample_indices(6, 12)
    assert short.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    long = uniform_sample_indices(96, 4)
    assert long.tolist() == [12, 36, 60, 84]


def test_uniform_sampling_never_indexes_past_the_clip():
    for available in (1, 2, 5, 31, 100):
        indices = uniform_sample_indices(available, 48)
        assert indices.min() >= 0 and indices.max() < available


def test_sampling_an_empty_sequence_raises_rather_than_returning_nothing():
    with pytest.raises(ActionBackendError):
        uniform_sample_indices(0, 48)


def test_the_input_tensor_has_pyskls_layout_and_the_engines_window():
    """(N, M, T, V, C) — FormatGCNInput's order. A transposed tensor is a shape
    error at best and a silent reinterpretation at worst."""
    engine = _FakeEngine({42: 9.0}, window_frames=48)
    backend = SkeletonActionBackend(engine=engine)
    backend.classify(_frames(_standing, n=30))
    assert engine.calls, "the engine was never called"
    assert engine.calls[0].shape == (1, 1, 48, N_KEYPOINTS, 2)
    assert engine.calls[0].dtype == np.float32


def test_the_window_comes_from_the_engine_not_from_our_constant():
    """An ST-GCN export has a fixed temporal dimension; ours is only a fallback."""
    engine = _FakeEngine({42: 9.0}, window_frames=100)
    SkeletonActionBackend(engine=engine).classify(_frames(_standing, n=30))
    assert engine.calls[0].shape[2] == 100


def test_the_backend_refuses_frames_with_no_declared_image_size():
    """Guessing the frame from the bounding box would silently misnormalise
    every input, and this model answers a misnormalised skeleton with confident
    nonsense rather than an error."""
    engine = _FakeEngine({42: 9.0})
    frames = _frames(_standing, n=20, image_size=None)
    result = SkeletonActionBackend(engine=engine).classify(frames)
    assert result["backend_error"] is True
    assert "image_size" in result["evidence"]["reason"]
    assert engine.calls == []


def test_the_window_bounds_what_the_model_sees():
    engine = _FakeEngine({42: 9.0})
    frames = _frames(_standing, n=60, fps=10.0)      # 6 s of video
    SkeletonActionBackend(engine=engine, window_s=2.0).classify(frames)
    # 2 s at 10 fps is 21 frames inclusive; they are then resampled to T.
    assert engine.calls[0].shape[2] == DEFAULT_WINDOW_FRAMES


# ── logits vs probabilities ─────────────────────────────────────────────────

def test_a_softmaxed_output_is_detected_and_not_softmaxed_again():
    """Running softmax twice flattens the distribution towards uniform, which
    shows up as every score below threshold and the backend reporting nothing —
    with no error anywhere."""
    probabilities = np.zeros(N_CLASSES, dtype=np.float32)
    probabilities[42] = 0.9
    probabilities[0] = 0.1
    assert looks_like_probabilities(probabilities)
    engine = _FakeEngine(outputs=[probabilities[None]])
    result = SkeletonActionBackend(engine=engine).classify(_frames(_standing))
    assert result["action"] == "fall"
    assert result["action_confidence"] == pytest.approx(0.9, abs=0.01)


def test_raw_logits_are_softmaxed():
    engine = _FakeEngine({42: 12.0})
    result = SkeletonActionBackend(engine=engine).classify(_frames(_standing))
    assert result["action"] == "fall"
    assert 0.9 < result["action_confidence"] <= 1.0


def test_logits_are_not_mistaken_for_probabilities():
    logits = np.full(N_CLASSES, 0.01, dtype=np.float32)   # in [0,1] but sums to 0.6
    assert not looks_like_probabilities(logits)


def test_softmax_is_stable_on_large_logits():
    out = softmax(np.array([1000.0, 1001.0], dtype=np.float32))
    assert np.isfinite(out).all()
    assert out.sum() == pytest.approx(1.0)


# ── label mapping ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("ntu_index,expected", sorted(NTU60_TO_ACTION.items()))
def test_each_mapped_ntu_class_produces_its_label(ntu_index, expected):
    engine = _FakeEngine({ntu_index: 9.0})
    result = SkeletonActionBackend(engine=engine).classify(_frames(_standing))
    assert result["action"] == expected
    assert result["evidence"]["ntu_class"] == ntu_index + 1


def test_every_mapped_label_exists_in_the_shared_priority_order():
    """A label the priority order does not know would crash the merge."""
    for label in NTU60_TO_ACTION.values():
        assert label in ACTION_PRIORITY


def test_mutual_two_person_classes_are_not_mapped():
    """A59/A60 walking-towards/apart are defined on a *pair* of skeletons, and
    this backend is fed one person at a time — a confident prediction there
    would be meaningless. Which is why `walking` stays with the geometry."""
    assert max(NTU60_TO_ACTION) < 49
    assert "walking" not in NTU60_TO_ACTION.values()


def test_an_unmapped_class_winning_means_the_model_has_nothing_to_say():
    """Not `unknown` by way of a wrong label: class 5 is "drop something", for
    which this project has no label at all."""
    engine = _FakeEngine({5: 20.0})
    result = SkeletonActionBackend(engine=engine).classify(_frames(_standing))
    assert result["action"] == "unknown"
    assert result["actions"] == []


def test_transitions_are_flagged_as_transitions():
    """NTU's `sit down` is a transition; our `sitting` is a state. The reply
    says which it was, so a consumer is not told a state was observed."""
    engine = _FakeEngine({7: 9.0})
    result = SkeletonActionBackend(engine=engine).classify(_frames(_standing))
    assert result["action"] == "sitting"
    assert result["evidence"]["transition"] is True
    assert 7 in NTU60_TRANSITIONS


def test_a_score_below_min_score_reports_what_it_nearly_said():
    """So "the model is wrong" can be told from "the threshold is wrong" on a
    robot, which is the only way to tune this without a rebuild."""
    engine = _FakeEngine(outputs=[np.full(N_CLASSES, 1.0 / N_CLASSES,
                                          dtype=np.float32)[None]])
    result = SkeletonActionBackend(engine=engine, min_score=0.4).classify(
        _frames(_standing))
    assert result["action"] == "unknown"
    assert result["evidence"]["min_score"] == 0.4
    assert result["evidence"]["best"]["score"] < 0.4


def test_predict_exposes_the_full_ranking_for_info():
    engine = _FakeEngine({42: 5.0, 22: 4.0})
    prediction = SkeletonActionBackend(engine=engine).predict(_frames(_standing))
    labels = [entry["action"] for entry in prediction["scores"]]
    assert labels[0] == "fall" and labels[1] == "waving"
    assert prediction["frames_used"] > 0


# ── failure paths must not become labels ────────────────────────────────────

def test_an_engine_failure_is_reported_as_an_error_not_as_unknown():
    """`unknown` would be indistinguishable from "nobody is doing anything" and
    would hide a broken engine for weeks."""
    engine = _FakeEngine(raise_on_infer=RuntimeError("deserialize failed"))
    result = SkeletonActionBackend(engine=engine).classify(_frames(_standing))
    assert result["backend_error"] is True
    assert "deserialize failed" in result["evidence"]["reason"]


def test_an_output_with_too_few_classes_is_refused():
    engine = _FakeEngine(outputs=[np.zeros((1, 10), dtype=np.float32)])
    result = SkeletonActionBackend(engine=engine).classify(_frames(_standing))
    assert result["backend_error"] is True
    assert "NTU-60" in result["evidence"]["reason"]


def test_a_wrong_keypoint_count_is_refused_at_the_boundary():
    """PoseFrame owns this invariant, because it indexes fixed COCO slots by
    name — a 13-joint array otherwise fails as an IndexError from deep in the
    geometry, which reads as a bug in the rules rather than the wrong input."""
    with pytest.raises(ValueError, match="COCO-17"):
        PoseFrame(0.0, _standing_box(), np.zeros((13, 3), dtype=np.float32),
                  0.3, image_size=FRAME)


def test_the_backend_also_guards_the_keypoint_count_itself():
    """Defence in depth: a caller that assembled the tensor some other way
    must not reach the engine with the wrong joint count."""
    engine = _FakeEngine({42: 9.0})
    frames = _frames(_standing, n=5)
    # Bypass PoseFrame's own check to exercise the backend's.
    frames[0].keypoints = np.zeros((13, 3), dtype=np.float32)
    result = SkeletonActionBackend(engine=engine).classify(frames)
    assert result["backend_error"] is True
    assert "17" in result["evidence"]["reason"]


def test_no_frames_is_unknown_without_touching_the_engine():
    engine = _FakeEngine({42: 9.0})
    assert SkeletonActionBackend(engine=engine).classify([])["action"] == "unknown"
    assert engine.calls == []


def test_a_single_frame_is_declined_rather_than_padded():
    """Padding one frame to T and calling it an action would produce a
    confident answer from a clip in which nothing moves."""
    engine = _FakeEngine({42: 9.0})
    result = SkeletonActionBackend(engine=engine).classify_frame(
        _frames(_standing, n=1)[0])
    assert result["action"] == "unknown"
    assert result["temporal"] is False
    assert "fall" in result["unavailable_actions"]
    assert engine.calls == []


# ── hybrid ──────────────────────────────────────────────────────────────────

def test_hybrid_takes_postures_from_geometry():
    """NTU-60 has no `standing` class, so a pure swap would lose the postures."""
    engine = _FakeEngine(outputs=[np.zeros((1, N_CLASSES), dtype=np.float32)])
    backend = HybridActionBackend(engine=engine)
    result = backend.classify(_frames(_standing, n=20))
    assert result["action"] == "standing"
    assert result["evidence"]["source"] == "rules"


def test_hybrid_lets_the_model_win_on_its_own_classes():
    engine = _FakeEngine({42: 9.0})
    backend = HybridActionBackend(engine=engine)
    result = backend.classify(_frames(_standing, n=20))
    assert result["action"] == "fall"
    assert result["evidence"]["source"] == "stgcn"


def test_hybrid_merges_a_posture_and_a_learned_action():
    """"Standing while waving" has to survive, as it does in the rules."""
    engine = _FakeEngine({22: 9.0})
    result = HybridActionBackend(engine=engine).classify(_frames(_standing, n=20))
    assert result["action"] == "waving"
    assert "standing" in result["actions"]


def test_hybrid_keeps_the_geometrys_point_direction():
    """The model names the act; only the geometry measures where."""
    joints = _body()
    shoulder = (CX - 0.10 * H, TOP + 0.18 * H)
    wrist = (CX - 0.40 * H, TOP + 0.19 * H)
    joints.update({"left_shoulder": shoulder, "left_wrist": wrist,
                   "left_elbow": ((shoulder[0] + wrist[0]) / 2,
                                  (shoulder[1] + wrist[1]) / 2)})
    box = (CX - 0.45 * H, TOP, CX + 0.15 * H, TOP + H)
    engine = _FakeEngine({30: 9.0})
    result = HybridActionBackend(engine=engine).classify(
        _frames(lambda _t: joints, n=20, box=box))
    assert result["action"] == "pointing"
    assert result["point_direction"][0] == pytest.approx(-1.0, abs=0.05)


def test_hybrid_still_answers_when_the_model_is_broken():
    """A dead action engine must not take the postures down with it, and must
    leave a trace rather than looking like a quiet frame."""
    engine = _FakeEngine(raise_on_infer=RuntimeError("no engine"))
    result = HybridActionBackend(engine=engine).classify(_frames(_standing, n=20))
    assert result["action"] == "standing"
    assert "no engine" in result["evidence"]["stgcn_error"]


def test_hybrid_history_covers_both_backends():
    engine = _FakeEngine({42: 9.0})
    backend = HybridActionBackend(engine=engine, window_s=4.0)
    assert backend.history_s >= 4.0
    assert backend.history_s >= backend.rules.history_s


def test_hybrid_defers_to_geometry_on_ntu_transitions():
    """`sit down` is a transition; the geometry observes the state directly."""
    engine = _FakeEngine({7: 9.0})
    result = HybridActionBackend(engine=engine).classify(_frames(_standing, n=20))
    # The model said "sit down"; the body is plainly standing.
    assert result["action"] == "standing"
    assert result["evidence"]["source"] == "rules"


def test_a_single_frame_through_hybrid_falls_to_the_geometry():
    engine = _FakeEngine({42: 9.0})
    result = HybridActionBackend(engine=engine).classify_frame(
        _frames(_standing, n=1)[0])
    assert result["action"] == "standing"


# ── backend selection ───────────────────────────────────────────────────────

def test_build_backend_returns_each_kind():
    assert isinstance(build_backend("rules"), PoseActionClassifier)
    assert isinstance(build_backend("stgcn", engine=_FakeEngine({42: 1.0})),
                      SkeletonActionBackend)
    assert isinstance(build_backend("hybrid", engine=_FakeEngine({42: 1.0})),
                      HybridActionBackend)


def test_an_unknown_backend_name_raises_rather_than_falling_back():
    """A card silently running geometry while its config says `stgcn` is the
    failure mode this whole exercise is about."""
    with pytest.raises(ActionBackendError, match="unknown action_backend"):
        build_backend("stgcnpp")


def test_every_backend_offers_the_same_interface():
    engine = _FakeEngine({42: 1.0})
    for name in ("rules", "stgcn", "hybrid"):
        backend = build_backend(name, engine=engine)
        assert callable(backend.classify)
        assert callable(backend.classify_frame)
        assert isinstance(backend.history_s, float)

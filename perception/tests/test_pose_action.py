"""
tests/test_pose_action.py — the rules that turn keypoints into an action label.

Entirely synthetic bodies and entirely numpy, which is the design point: the
pose plugin's only hardware-bound call is the engine's `infer()`, so every
decision about what counts as sitting, waving or falling is testable here.

The bodies are built from fractions of a person's height (see `_body`), because
that is what the rules themselves are written in — a test in raw pixels would
pass or fail depending on how far away the imaginary person stood.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest perception/tests -q
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import vision_stubs  # noqa: F401  (installs the cv2 / ROS stubs)

from plugins.pose_action import (  # noqa: E402
    ACTION_LABELS_ZH,
    ACTION_PRIORITY,
    DEFAULT_THRESHOLDS,
    EVENT_ACTIONS,
    PoseActionClassifier,
    PoseFrame,
    PoseTracker,
    action_catalogue,
)
from plugins.vision_runtime import COCO_INDEX, N_KEYPOINTS  # noqa: E402

H = 400.0          # the imaginary person's height in pixels
CX = 320.0         # their horizontal centre
TOP = 100.0        # top of their bounding box


def _kp(**named) -> np.ndarray:
    """COCO-17 array from {joint_name: (x, y)}; everything else invisible."""
    keypoints = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
    for name, xy in named.items():
        index = COCO_INDEX[name]
        keypoints[index, 0] = xy[0]
        keypoints[index, 1] = xy[1]
        keypoints[index, 2] = 0.9
    return keypoints


def _head(cx=CX, top=TOP, h=H) -> dict:
    return {
        "nose":      (cx, top + 0.06 * h),
        "left_eye":  (cx - 0.02 * h, top + 0.05 * h),
        "right_eye": (cx + 0.02 * h, top + 0.05 * h),
        "left_ear":  (cx - 0.04 * h, top + 0.06 * h),
        "right_ear": (cx + 0.04 * h, top + 0.06 * h),
    }


def _mid(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _body(*, cx=CX, top=TOP, h=H, left_ankle_dx=0.0, right_ankle_dx=0.0,
          shoulder_half=0.10) -> dict:
    """A plain standing body. Knees sit midway between hip and ankle, so the
    legs stay straight however the ankles are displaced."""
    hips = {"left_hip": (cx - 0.07 * h, top + 0.52 * h),
            "right_hip": (cx + 0.07 * h, top + 0.52 * h)}
    ankles = {"left_ankle": (cx - 0.07 * h + left_ankle_dx, top + 0.97 * h),
              "right_ankle": (cx + 0.07 * h + right_ankle_dx, top + 0.97 * h)}
    shoulders = {"left_shoulder": (cx - shoulder_half * h, top + 0.18 * h),
                 "right_shoulder": (cx + shoulder_half * h, top + 0.18 * h)}
    arms = {}
    for side in ("left", "right"):
        sign = -1.0 if side == "left" else 1.0
        arms[f"{side}_wrist"] = (cx + sign * 0.12 * h, top + 0.50 * h)
        arms[f"{side}_elbow"] = _mid(shoulders[f"{side}_shoulder"],
                                     arms[f"{side}_wrist"])
    return {
        **_head(cx, top, h), **shoulders, **arms, **hips,
        "left_knee": _mid(hips["left_hip"], ankles["left_ankle"]),
        "right_knee": _mid(hips["right_hip"], ankles["right_ankle"]),
        **ankles,
    }


def _standing_box(cx=CX, top=TOP, h=H, half_w=0.15):
    return (cx - half_w * h, top, cx + half_w * h, top + h)


def _frame(joints: dict, box, t: float = 0.0, min_conf: float = 0.3) -> PoseFrame:
    return PoseFrame(t, box, _kp(**joints), min_conf)


def _sequence(joints_at, box_at, times) -> list:
    return [_frame(joints_at(t), box_at(t), t) for t in times]


def _classify(frames, **config) -> dict:
    return PoseActionClassifier(**config).classify(frames)


def _steady(joints: dict, box, *, duration=1.5, fps=10.0) -> list:
    steps = int(round(duration * fps)) + 1
    return [_frame(joints, box, i / fps) for i in range(steps)]


# ── postures ────────────────────────────────────────────────────────────────

def test_a_standing_body_is_standing():
    result = _classify(_steady(_body(), _standing_box()))
    assert result["action"] == "standing"
    assert "standing" in result["actions"]
    assert result["action_confidence"] >= DEFAULT_THRESHOLDS["min_confidence"]


def test_standing_also_reports_still_but_standing_wins():
    """`still` says less than `standing` does, so it must not be primary."""
    result = _classify(_steady(_body(), _standing_box()))
    assert result["action"] == "standing"
    assert "still" in result["actions"]


def _sitting_body(cx=CX, top=TOP, h=H) -> dict:
    hips = {"left_hip": (cx - 0.07 * h, top + 0.52 * h),
            "right_hip": (cx + 0.07 * h, top + 0.52 * h)}
    # Knees forward at hip height, shins dropping to the floor: the knee angle
    # is a right angle and the hips and knees are level.
    knees = {"left_knee": (cx - 0.07 * h + 0.20 * h, top + 0.52 * h),
             "right_knee": (cx + 0.07 * h + 0.20 * h, top + 0.52 * h)}
    ankles = {"left_ankle": (knees["left_knee"][0], top + 0.75 * h),
              "right_ankle": (knees["right_knee"][0], top + 0.75 * h)}
    shoulders = {"left_shoulder": (cx - 0.10 * h, top + 0.18 * h),
                 "right_shoulder": (cx + 0.10 * h, top + 0.18 * h)}
    return {**_head(cx, top, h), **shoulders, **hips, **knees, **ankles}


def test_a_seated_body_is_sitting_not_standing():
    box = (CX - 0.22 * H, TOP, CX + 0.30 * H, TOP + 0.80 * H)
    result = _classify(_steady(_sitting_body(), box))
    assert result["action"] == "sitting"
    assert "standing" not in result["actions"]


def _crouching_body(cx=CX, top=TOP, h=H) -> dict:
    hips = {"left_hip": (cx - 0.07 * h, top + 0.60 * h),
            "right_hip": (cx + 0.07 * h, top + 0.60 * h)}
    knees = {"left_knee": (cx - 0.07 * h + 0.12 * h, top + 0.70 * h),
             "right_knee": (cx + 0.07 * h + 0.12 * h, top + 0.70 * h)}
    ankles = {"left_ankle": (cx - 0.07 * h, top + 0.75 * h),
              "right_ankle": (cx + 0.07 * h, top + 0.75 * h)}
    shoulders = {"left_shoulder": (cx - 0.10 * h, top + 0.30 * h),
                 "right_shoulder": (cx + 0.10 * h, top + 0.30 * h)}
    return {**_head(cx, top, h), **shoulders, **hips, **knees, **ankles}


def test_a_crouching_body_is_crouching_not_sitting():
    """Folded knees past the sitting range, hips down near the ankles."""
    box = (CX - 0.20 * H, TOP, CX + 0.22 * H, TOP + 0.78 * H)
    result = _classify(_steady(_crouching_body(), box))
    assert result["action"] == "crouching"
    assert "sitting" not in result["actions"]


def _bending_body(cx=CX, top=TOP, h=H) -> dict:
    hips = {"left_hip": (cx - 0.07 * h, top + 0.52 * h),
            "right_hip": (cx + 0.07 * h, top + 0.52 * h)}
    ankles = {"left_ankle": (cx - 0.07 * h, top + 0.97 * h),
              "right_ankle": (cx + 0.07 * h, top + 0.97 * h)}
    shoulders = {"left_shoulder": (cx + 0.25 * h, top + 0.28 * h),
                 "right_shoulder": (cx + 0.25 * h, top + 0.32 * h)}
    return {
        **_head(cx + 0.30 * h, top + 0.20 * h, h), **shoulders, **hips,
        "left_knee": _mid(hips["left_hip"], ankles["left_ankle"]),
        "right_knee": _mid(hips["right_hip"], ankles["right_ankle"]),
        **ankles,
    }


def test_bending_over_is_not_crouching():
    """Straight legs are the whole difference — both put the torso over the
    floor, only one folds the knees."""
    box = (CX - 0.15 * H, TOP + 0.18 * H, CX + 0.40 * H, TOP + H)
    result = _classify(_steady(_bending_body(), box))
    assert result["action"] == "bending"
    assert "crouching" not in result["actions"]
    assert "lying" not in result["actions"]


def _lying_body(cx=CX, floor=TOP + 0.90 * H, h=H) -> dict:
    hips = {"left_hip": (cx + 0.05 * h, floor + 0.01 * h),
            "right_hip": (cx + 0.05 * h, floor + 0.03 * h)}
    shoulders = {"left_shoulder": (cx - 0.30 * h, floor),
                 "right_shoulder": (cx - 0.30 * h, floor + 0.04 * h)}
    knees = {"left_knee": (cx + 0.25 * h, floor + 0.02 * h),
             "right_knee": (cx + 0.25 * h, floor + 0.04 * h)}
    ankles = {"left_ankle": (cx + 0.42 * h, floor + 0.02 * h),
              "right_ankle": (cx + 0.42 * h, floor + 0.04 * h)}
    return {**_head(cx - 0.40 * h, floor - 0.04 * h, h),
            **shoulders, **hips, **knees, **ankles}


def _lying_box(cx=CX, floor=TOP + 0.90 * H, h=H):
    return (cx - 0.45 * h, floor - 0.06 * h, cx + 0.45 * h, floor + 0.08 * h)


def test_a_horizontal_body_is_lying():
    result = _classify(_steady(_lying_body(), _lying_box(), duration=0.5))
    assert "lying" in result["actions"]
    assert result["action"] == "lying"


# ── occlusion ───────────────────────────────────────────────────────────────

def test_a_body_with_no_visible_hips_refuses_to_guess_a_posture():
    """Sitting at a desk and standing behind it have the same torso axis.

    This is the case the plugin exists to not get wrong: inferring a posture
    from the shoulders alone would make "I can't see" indistinguishable from
    "nothing is wrong".
    """
    upper = {**_head(),
             "left_shoulder": (CX - 0.10 * H, TOP + 0.18 * H),
             "right_shoulder": (CX + 0.10 * H, TOP + 0.18 * H)}
    result = _classify(_steady(upper, _standing_box()))
    assert result["action"] == "unknown"
    assert result["actions"] == []
    assert result["evidence"]["reason"] == "occluded"
    assert result["evidence"]["has_torso"] is False


def test_a_nearly_invisible_body_is_unknown():
    result = _classify(_steady({"nose": (CX, TOP)}, _standing_box()))
    assert result["action"] == "unknown"
    assert result["evidence"]["visible_keypoints"] == 1


def test_an_empty_history_is_unknown_rather_than_an_error():
    assert PoseActionClassifier().classify([])["action"] == "unknown"


# ── arms ────────────────────────────────────────────────────────────────────

def _raised_arm(joints: dict, *, side="left", wrist_dx=0.0, cx=CX, top=TOP,
                h=H) -> dict:
    sign = -1.0 if side == "left" else 1.0
    shoulder = (cx + sign * 0.10 * h, top + 0.18 * h)
    wrist = (cx + sign * 0.10 * h + wrist_dx, top + 0.05 * h)
    return {**joints,
            f"{side}_shoulder": shoulder,
            f"{side}_wrist": wrist,
            f"{side}_elbow": _mid(shoulder, wrist)}


def test_a_held_raised_hand_is_raising_hand():
    frames = _steady(_raised_arm(_body()), _standing_box(), duration=1.0)
    result = _classify(frames)
    assert result["action"] == "raising_hand"
    assert result["evidence"]["side"] == "left"
    assert result["evidence"]["held_s"] >= DEFAULT_THRESHOLDS["raise_hold_s"]


def test_a_glimpsed_raised_hand_is_not_raising_hand():
    """An arm passing through shoulder height on one frame is not a raise."""
    plain, raised = _body(), _raised_arm(_body())
    frames = [_frame(plain, _standing_box(), 0.0),
              _frame(plain, _standing_box(), 0.1),
              _frame(raised, _standing_box(), 0.2)]
    assert "raising_hand" not in _classify(frames)["actions"]


def test_a_hand_that_oscillates_while_raised_is_waving():
    """This is the "someone is calling the robot over" case."""
    def joints_at(t):
        return _raised_arm(_body(), wrist_dx=0.15 * H * math.sin(2 * math.pi * 1.5 * t))
    frames = _sequence(joints_at, lambda _t: _standing_box(),
                       [i / 10.0 for i in range(11)])
    result = _classify(frames)
    assert result["action"] == "waving"
    assert "raising_hand" in result["actions"]      # both hold; waving is primary
    evidence = result["evidence"]
    assert evidence["reversals"] >= DEFAULT_THRESHOLDS["wave_reversals"]
    assert (DEFAULT_THRESHOLDS["wave_freq_min_hz"] <= evidence["frequency_hz"]
            <= DEFAULT_THRESHOLDS["wave_freq_max_hz"])


def test_a_still_raised_hand_is_not_waving():
    frames = _steady(_raised_arm(_body()), _standing_box(), duration=1.0)
    assert "waving" not in _classify(frames)["actions"]


def test_an_arm_swinging_below_the_shoulder_is_not_waving():
    """Gating waving on the raise is what keeps a walking arm swing out."""
    def joints_at(t):
        joints = _body()
        shoulder = joints["left_shoulder"]
        wrist = (CX - 0.12 * H + 0.15 * H * math.sin(2 * math.pi * 1.5 * t),
                 TOP + 0.50 * H)
        joints["left_wrist"] = wrist
        joints["left_elbow"] = _mid(shoulder, wrist)
        return joints
    frames = _sequence(joints_at, lambda _t: _standing_box(),
                       [i / 10.0 for i in range(11)])
    actions = _classify(frames)["actions"]
    assert "waving" not in actions and "raising_hand" not in actions


def test_a_straight_horizontal_arm_is_pointing_and_reports_a_direction():
    joints = _body()
    shoulder = (CX - 0.10 * H, TOP + 0.18 * H)
    wrist = (CX - 0.40 * H, TOP + 0.19 * H)
    joints.update({"left_shoulder": shoulder, "left_wrist": wrist,
                   "left_elbow": _mid(shoulder, wrist)})
    box = (CX - 0.45 * H, TOP, CX + 0.15 * H, TOP + H)
    result = _classify(_steady(joints, box))
    assert result["action"] == "pointing"
    direction = result["point_direction"]
    assert direction[0] == pytest.approx(-1.0, abs=0.05)
    assert abs(direction[1]) < 0.2
    assert np.hypot(*direction) == pytest.approx(1.0, abs=1e-2)


def test_two_wrists_across_the_midline_at_chest_height_are_arms_crossed():
    joints = _body()
    joints.update({
        "left_wrist": (CX + 0.08 * H, TOP + 0.40 * H),
        "right_wrist": (CX - 0.08 * H, TOP + 0.40 * H),
        "left_elbow": (CX - 0.18 * H, TOP + 0.35 * H),
        "right_elbow": (CX + 0.18 * H, TOP + 0.35 * H),
    })
    result = _classify(_steady(joints, _standing_box(half_w=0.20)))
    assert result["action"] == "arms_crossed"
    assert "pointing" not in result["actions"]


# ── motion ──────────────────────────────────────────────────────────────────

def test_alternating_ankles_under_an_upright_torso_is_walking():
    def joints_at(t):
        swing = 0.12 * H * math.sin(2 * math.pi * 1.0 * t)
        return _body(left_ankle_dx=swing, right_ankle_dx=-swing)
    frames = _sequence(joints_at, lambda _t: _standing_box(half_w=0.20),
                       [i / 10.0 for i in range(16)])
    result = _classify(frames)
    assert "walking" in result["actions"]
    assert result["action"] == "walking"      # above standing: more informative
    cadence = result["evidence"]["cadence_hz"]
    assert (DEFAULT_THRESHOLDS["walk_cadence_min_hz"] <= cadence
            <= DEFAULT_THRESHOLDS["walk_cadence_max_hz"])


def test_a_seated_body_shuffling_its_feet_is_not_walking():
    """Walking is gated on an upright standing torso for exactly this case."""
    def joints_at(t):
        joints = _sitting_body()
        swing = 0.12 * H * math.sin(2 * math.pi * 1.0 * t)
        joints["left_ankle"] = (joints["left_ankle"][0] + swing,
                                joints["left_ankle"][1])
        joints["right_ankle"] = (joints["right_ankle"][0] - swing,
                                 joints["right_ankle"][1])
        return joints
    box = (CX - 0.30 * H, TOP, CX + 0.40 * H, TOP + 0.80 * H)
    frames = _sequence(joints_at, lambda _t: box, [i / 10.0 for i in range(16)])
    assert "walking" not in _classify(frames)["actions"]


def test_a_narrowing_shoulder_width_is_turning():
    def joints_at(t):
        half = 0.10 - 0.06 * (t / 1.5)
        return _body(shoulder_half=max(half, 0.02))
    frames = _sequence(joints_at, lambda _t: _standing_box(),
                       [i / 10.0 for i in range(16)])
    result = _classify(frames)
    assert "turning" in result["actions"]


# ── fall ────────────────────────────────────────────────────────────────────

def _fall_sequence(*, sit_before_landing=False, fps=10.0,
                   upright_s=1.0, settle_s=1.2) -> list:
    frames = []
    t = 0.0
    step = 1 / fps
    while t < upright_s:
        frames.append(_frame(_body(), _standing_box(), t))
        t += step
    if sit_before_landing:
        frames.append(_frame(_sitting_body(),
                             (CX - 0.22 * H, TOP, CX + 0.30 * H, TOP + 0.80 * H),
                             t))
        t += step
    landed_at = t
    while t < landed_at + settle_s:
        frames.append(_frame(_lying_body(), _lying_box(), t))
        t += step
    return frames


def test_a_fast_drop_into_a_sustained_horizontal_pose_is_a_fall():
    result = _classify(_fall_sequence())
    assert result["action"] == "fall"
    assert result["actions"][0] == "fall"
    evidence = result["evidence"]
    assert evidence["is_fall"] is True
    assert evidence["drop_ratio"] >= DEFAULT_THRESHOLDS["fall_drop_ratio"]
    assert evidence["drop_ms"] <= DEFAULT_THRESHOLDS["fall_drop_window_s"] * 1000
    assert evidence["settle_ms"] >= DEFAULT_THRESHOLDS["fall_settle_s"] * 1000
    assert evidence["had_sitting_phase"] is False


def test_sitting_down_then_lying_down_is_lying_not_a_fall():
    """Sitting on the way down is the signature of a deliberate descent."""
    result = _classify(_fall_sequence(sit_before_landing=True))
    assert result["action"] == "lying"
    assert "fall" not in result["actions"]
    assert result["evidence"]["had_sitting_phase"] is True
    assert result["evidence"]["is_fall"] is False


def test_someone_already_lying_down_is_not_a_fall():
    """Lying on the floor and lying on a sofa are the same terminal state —
    without the drop there is no fall to report."""
    result = _classify(_steady(_lying_body(), _lying_box(), duration=2.0))
    assert result["action"] == "lying"
    assert "fall" not in result["actions"]
    assert result["evidence"]["is_fall"] is False
    assert result["evidence"]["reason"] == "no fast drop into the horizontal pose"


def test_a_fall_is_not_declared_before_the_body_has_settled():
    """Bending down to pick something up is horizontal too, briefly."""
    result = _classify(_fall_sequence(settle_s=0.4))
    assert "fall" not in result["actions"]
    assert result["action"] == "lying"
    assert "is_fall" not in result["evidence"]


def test_the_fall_drop_is_measured_against_the_standing_height():
    """Normalising by the current box would divide the drop by the post-fall
    height — a person on the floor has a short, wide box."""
    frames = _fall_sequence()
    tall = max(f.height for f in frames)
    short = frames[-1].height
    assert short < tall / 2
    evidence = _classify(frames)["evidence"]
    assert evidence["drop_ratio"] < 1.0        # would exceed 1.0 if divided by `short`


def test_a_raised_fall_threshold_suppresses_the_same_fall():
    """The fall thresholds are instance config, not constants: the same fall
    measures differently depending on where the camera is."""
    frames = _fall_sequence()
    assert _classify(frames)["action"] == "fall"
    strict = _classify(frames, thresholds={"fall_drop_ratio": 0.95})
    assert strict["action"] == "lying"
    assert strict["evidence"]["is_fall"] is False


# ── classifier plumbing ─────────────────────────────────────────────────────

def test_unknown_threshold_keys_are_ignored():
    classifier = PoseActionClassifier(thresholds={"nonsense": 1, "upright_deg": 10})
    assert "nonsense" not in classifier.thresholds
    assert classifier.thresholds["upright_deg"] == 10


def test_none_valued_thresholds_fall_back_to_the_default():
    """An unset field on a canvas card arrives as None, not as absent."""
    classifier = PoseActionClassifier(thresholds={"upright_deg": None})
    assert classifier.thresholds["upright_deg"] == DEFAULT_THRESHOLDS["upright_deg"]


def test_history_s_covers_the_whole_fall_window():
    """Ask for too little history and fall can never fire, with nothing in any
    log to say why."""
    classifier = PoseActionClassifier(action_window_s=0.5)
    th = classifier.thresholds
    assert classifier.history_s >= th["fall_drop_window_s"] + th["fall_settle_s"]


def test_the_action_window_bounds_what_the_arm_rules_see():
    """A hand raised two seconds ago is not a hand raised now."""
    frames = _steady(_raised_arm(_body()), _standing_box(), duration=1.0)
    frames += [_frame(_body(), _standing_box(), 1.0 + i / 10.0)
               for i in range(1, 12)]
    assert "raising_hand" not in _classify(frames, action_window_s=0.5)["actions"]


def test_every_priority_entry_has_a_chinese_label():
    assert set(ACTION_LABELS_ZH) == set(ACTION_PRIORITY)
    assert ACTION_PRIORITY[-1] == "unknown"
    assert ACTION_PRIORITY[0] == "fall"


def test_the_catalogue_marks_fall_as_an_event_not_a_pose():
    catalogue = {entry["action"]: entry for entry in action_catalogue()}
    assert catalogue["fall"]["kind"] == "event"
    assert catalogue["standing"]["kind"] == "pose"
    assert set(EVENT_ACTIONS) == {"fall"}
    assert all(entry["label_zh"] for entry in catalogue.values())


# ── tracking ────────────────────────────────────────────────────────────────

def test_an_overlapping_box_keeps_its_track_id():
    tracker = PoseTracker()
    keypoints = [_kp(**_body())]
    first = tracker.update([_standing_box()], keypoints, 0.0)[0]
    moved = (CX - 0.15 * H + 10, TOP + 5, CX + 0.15 * H + 10, TOP + H + 5)
    second = tracker.update([moved], keypoints, 0.1)[0]
    assert first.id == second.id
    assert len(second.history) == 2


def test_a_disjoint_box_starts_a_new_track():
    tracker = PoseTracker()
    keypoints = [_kp(**_body())]
    first = tracker.update([(0, 0, 50, 100)], keypoints, 0.0)[0]
    second = tracker.update([(500, 400, 560, 500)], keypoints, 0.1)[0]
    assert first.id != second.id


def test_two_people_keep_separate_timelines():
    tracker = PoseTracker()
    left, right = (0, 0, 100, 300), (400, 0, 500, 300)
    keypoints = [_kp(**_body()), _kp(**_body(cx=450))]
    a1, b1 = tracker.update([left, right], keypoints, 0.0)
    # Same two people, reported in the opposite order this frame.
    b2, a2 = tracker.update([right, left], list(reversed(keypoints)), 0.1)
    assert a1.id == a2.id and b1.id == b2.id
    assert a1.id != b1.id


def test_a_track_that_vanishes_for_too_long_is_not_reused():
    tracker = PoseTracker(timeout_s=0.5)
    keypoints = [_kp(**_body())]
    first = tracker.update([_standing_box()], keypoints, 0.0)[0]
    later = tracker.update([_standing_box()], keypoints, 2.0)[0]
    assert first.id != later.id
    assert len(tracker.tracks) == 1          # the stale one is gone


def test_history_is_trimmed_to_the_configured_window():
    tracker = PoseTracker(history_s=0.5)
    keypoints = [_kp(**_body())]
    for i in range(30):
        tracks = tracker.update([_standing_box()], keypoints, i / 10.0)
    history = tracks[0].history
    assert history[-1].t - history[0].t <= 0.5
    assert len(history) <= 7


def test_tracking_survives_a_frame_with_nobody_in_it():
    tracker = PoseTracker(timeout_s=1.0)
    keypoints = [_kp(**_body())]
    first = tracker.update([_standing_box()], keypoints, 0.0)[0]
    assert tracker.update([], [], 0.1) == []
    again = tracker.update([_standing_box()], keypoints, 0.2)[0]
    assert first.id == again.id


def test_still_is_not_reported_for_a_body_too_occluded_to_place():
    """A motionless person whose posture is unreadable must come back as
    `unknown`, not as `still` — otherwise "I can't see them" is reported as a
    positive observation about them."""
    upper = {**_head(),
             "left_shoulder": (CX - 0.10 * H, TOP + 0.18 * H),
             "right_shoulder": (CX + 0.10 * H, TOP + 0.18 * H)}
    result = _classify(_steady(upper, _standing_box()))
    assert "still" not in result["actions"]


def test_an_occluded_body_can_still_report_its_arms():
    """The arm rules only need shoulder/elbow/wrist, so a person visible from
    the waist up can still be seen calling the robot over."""
    upper = _raised_arm({**_head(),
                         "left_shoulder": (CX - 0.10 * H, TOP + 0.18 * H),
                         "right_shoulder": (CX + 0.10 * H, TOP + 0.18 * H)})
    result = _classify(_steady(upper, _standing_box(), duration=1.0))
    assert result["action"] == "raising_hand"

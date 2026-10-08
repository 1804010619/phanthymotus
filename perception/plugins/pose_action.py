#!/usr/bin/env python3
"""
plugins/pose_action.py — turn a COCO-17 keypoint sequence into an action label.

Pure numpy. No model, no weights, no GPU — which is the point: this layer is
fully testable on a laptop, and the only part of the pose plugin that needs
hardware is `VisionEngineSession.infer()`. Same split as
`actucore/tests/test_smolvla_provider.py` gets from `_build_policy`/`_predict`.

Three things shape everything here.

**Two output fields, not one.** `action` is the single primary label (what goes
into the LLM prompt) and `actions` is every label that holds. "Standing while
waving" is two things that are simultaneously true, and collapsing them into one
string loses whichever matters to the caller.

**Normalised by body height, never by pixels.** Every threshold below is a
fraction of the person's bounding-box height, so a person 3 m away and the same
person 1 m away produce the same numbers. A pixel threshold would be a distance
threshold wearing a disguise.

**Occlusion yields `unknown`, not a guess.** A person behind a desk has no
visible hips or knees, and there is no way to tell sitting from standing from
the torso alone. Inferring one anyway is worse than silence — "I don't know"
and "nothing is wrong" have to be two different answers, the same rule
`visual_depth`'s lens-barrel mask exists for. Arm labels still work in that
case, because they only need shoulder/elbow/wrist.

`fall` is the one label that is not a posture: see `_fall_evidence`.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Optional

import numpy as np

from plugins.vision_runtime import COCO_INDEX, N_KEYPOINTS

log = logging.getLogger(__name__)

# ── label set ────────────────────────────────────────────────────────────────

# Priority for picking the single primary label out of everything that holds.
#
# It is NOT "posture before motion". A walking person is also standing, and
# `action: "standing"` for someone crossing the room in front of the robot is
# the less useful of two true answers — so `walking` sits above `standing`
# while staying below the postures that contradict it. `still` is last for the
# same reason in reverse: it says less than `standing` does.
ACTION_PRIORITY = (
    "fall",
    "lying",
    "waving",
    "raising_hand",
    "pointing",
    "arms_crossed",
    "crouching",
    "sitting",
    "bending",
    "walking",
    "turning",
    "standing",
    "still",
    "unknown",
)

ACTIONS = tuple(a for a in ACTION_PRIORITY if a != "unknown")

# Labels whose meaning is "something happened", not "this is the current pose".
EVENT_ACTIONS = ("fall",)

# Chinese names, for the `list_actions` reply and the card. Kept beside the
# labels so the two cannot drift.
ACTION_LABELS_ZH = {
    "standing": "站立",
    "sitting": "坐",
    "crouching": "蹲",
    "bending": "弯腰",
    "lying": "躺",
    "raising_hand": "举手",
    "waving": "挥手/招手",
    "pointing": "指向",
    "arms_crossed": "抱臂",
    "walking": "走动",
    "turning": "转身",
    "still": "静止",
    "fall": "跌倒",
    "unknown": "不确定",
}

# Every threshold the rules use, in one place, so the plugin can expose them
# per instance. The fall ones in particular are NOT constants: `drop_ratio`
# measured for the same fall differs with camera height, pitch and focal
# length, so a default that works on one rig is wrong on the next.
DEFAULT_THRESHOLDS = {
    # visibility
    "kpt_confidence": 0.3,
    # posture
    "upright_deg": 25.0,          # torso within this of vertical = upright
    "bend_deg": 35.0,             # torso past this = leaning
    "lying_deg": 60.0,            # torso past this = horizontal
    "lying_aspect": 1.2,          # bbox w/h past this = lying down
    "leg_straight_deg": 150.0,    # knee angle past this = straight leg
    "knee_bent_max_deg": 120.0,   # sitting: knee between bent_min and this
    "knee_bent_min_deg": 70.0,    # crouching: knee below this
    "sit_hip_knee_dy": 0.15,      # |y_hip - y_knee| / h below this = sitting
    "crouch_hip_ankle_dy": 0.25,  # (y_ankle - y_hip) / h below this = crouching
    "stand_hip_knee_dy": 0.15,    # knees must be this far below the hips
    # arms
    "raise_wrist_above_shoulder": 0.10,   # (y_shoulder - y_wrist) / h
    "elbow_open_deg": 90.0,
    "arm_straight_deg": 150.0,
    "point_horizontal_deg": 30.0,
    "raise_hold_s": 0.3,
    "wave_reversals": 2,
    "wave_amplitude": 0.08,       # wrist x travel / h
    "wave_freq_min_hz": 0.5,
    "wave_freq_max_hz": 4.0,
    # motion
    "still_speed": 0.06,          # body-heights per second
    "walk_amplitude": 0.05,       # ankle separation travel / h
    "walk_cadence_min_hz": 0.7,
    "walk_cadence_max_hz": 3.0,
    "turn_width_change": 0.30,    # shoulder-width change across the window
    # fall — tune these on the rig, with the camera where it will actually be
    "fall_drop_ratio": 0.35,      # hip drop / standing height
    "fall_drop_window_s": 0.6,
    "fall_settle_s": 1.0,
    # output
    "min_confidence": 0.45,       # below this the primary label is `unknown`
}

_K = COCO_INDEX


# ── small geometry helpers ───────────────────────────────────────────────────

def _visible(keypoints: np.ndarray, index: int, min_conf: float) -> bool:
    return bool(keypoints[index, 2] >= min_conf)


def _point(keypoints: np.ndarray, name: str, min_conf: float) -> Optional[np.ndarray]:
    index = _K[name]
    if not _visible(keypoints, index, min_conf):
        return None
    return keypoints[index, :2].astype(np.float32)


def _midpoint(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Midpoint of two joints, or the one that is visible, or None.

    Falling back to a single side is deliberate: a person seen from 45 degrees
    routinely has one hip occluded, and refusing to produce a torso axis for
    that is refusing to classify most real frames.
    """
    if a is not None and b is not None:
        return (a + b) / 2.0
    return a if a is not None else b


def _mid_of(points: list) -> Optional[np.ndarray]:
    """Mean of however many of a joint pair came back visible."""
    if not points:
        return None
    return np.mean(np.stack(points), axis=0).astype(np.float32)


def _angle_deg(a: Optional[np.ndarray], b: Optional[np.ndarray],
               c: Optional[np.ndarray]) -> Optional[float]:
    """Interior angle at `b`, in degrees. 180 = straight."""
    if a is None or b is None or c is None:
        return None
    v1, v2 = a - b, c - b
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos = float(np.dot(v1, v2)) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _tilt_from_vertical_deg(vector: Optional[np.ndarray]) -> Optional[float]:
    """Angle between a vector and the image's vertical axis, 0-90.

    `abs` on the dot product so an inverted torso (someone upside down) reads
    as 0 rather than 180: what the postures care about is whether the body is
    aligned with gravity, not which end is up.
    """
    if vector is None:
        return None
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return None
    return math.degrees(math.acos(min(1.0, abs(float(vector[1])) / norm)))


def body_height(box) -> float:
    """The scale every threshold is a fraction of. Floored, never zero."""
    return max(float(box[3]) - float(box[1]), 1.0)


# ── per-frame features ──────────────────────────────────────────────────────

class PoseFrame:
    """Everything a single frame can say about one person.

    Separated from the rules so the rules read as thresholds on named
    quantities, and so a feature can be asserted on directly in a test.
    """

    __slots__ = ("t", "box", "keypoints", "height", "min_conf",
                 "shoulder", "hip", "torso_deg", "knee_deg",
                 "hip_knee_dy", "hip_ankle_dy", "aspect",
                 "has_torso", "has_legs")

    def __init__(self, t: float, box, keypoints: np.ndarray, min_conf: float):
        self.t = float(t)
        self.box = [float(v) for v in box]
        self.keypoints = keypoints
        self.min_conf = float(min_conf)
        self.height = body_height(box)
        width = max(self.box[2] - self.box[0], 1.0)
        self.aspect = width / self.height

        ls = _point(keypoints, "left_shoulder", min_conf)
        rs = _point(keypoints, "right_shoulder", min_conf)
        lh = _point(keypoints, "left_hip", min_conf)
        rh = _point(keypoints, "right_hip", min_conf)
        self.shoulder = _midpoint(ls, rs)
        self.hip = _midpoint(lh, rh)
        self.has_torso = self.shoulder is not None and self.hip is not None
        self.torso_deg = _tilt_from_vertical_deg(
            None if not self.has_torso else self.hip - self.shoulder)

        # Knee angle: the mean over whichever legs are fully visible. A single
        # visible leg is enough — see _midpoint.
        knees, knee_pts, ankle_pts = [], [], []
        for side in ("left", "right"):
            hip_p = _point(keypoints, f"{side}_hip", min_conf)
            knee_p = _point(keypoints, f"{side}_knee", min_conf)
            ankle_p = _point(keypoints, f"{side}_ankle", min_conf)
            angle = _angle_deg(hip_p, knee_p, ankle_p)
            if angle is not None:
                knees.append(angle)
            if knee_p is not None:
                knee_pts.append(knee_p)
            if ankle_p is not None:
                ankle_pts.append(ankle_p)
        self.knee_deg = float(np.mean(knees)) if knees else None
        self.has_legs = self.knee_deg is not None and self.hip is not None

        knee_mid = _mid_of(knee_pts)
        ankle_mid = _mid_of(ankle_pts)
        self.hip_knee_dy = (
            None if (self.hip is None or knee_mid is None)
            else float(knee_mid[1] - self.hip[1]) / self.height)
        self.hip_ankle_dy = (
            None if (self.hip is None or ankle_mid is None)
            else float(ankle_mid[1] - self.hip[1]) / self.height)

    def joint(self, name: str) -> Optional[np.ndarray]:
        return _point(self.keypoints, name, self.min_conf)

    def visible_count(self) -> int:
        return int(np.count_nonzero(self.keypoints[:, 2] >= self.min_conf))


# ── window features ─────────────────────────────────────────────────────────

def _confidence(value: float, threshold: float, *, below: bool = True) -> float:
    """Heuristic 0.5-1.0 score for "how far past its threshold is this".

    These are NOT probabilities and must not be read as any. They exist so two
    labels that both hold can be ordered, and so `min_confidence` has something
    to compare against — a rule that only just scrapes past its threshold gets
    ~0.5, one that clears it by a wide margin approaches 1.0.
    """
    if threshold <= 0:
        return 0.5
    margin = (threshold - value) / threshold if below else (value - threshold) / threshold
    return float(0.5 + 0.5 * max(0.0, min(1.0, margin)))


def _joint_speed(frames: list, name: str) -> Optional[float]:
    """Mean speed of one joint across the window, in body-heights per second."""
    samples = [(f.t, f.joint(name), f.height) for f in frames]
    samples = [(t, p, h) for t, p, h in samples if p is not None]
    if len(samples) < 2:
        return None
    total, seconds = 0.0, 0.0
    for (t0, p0, h0), (t1, p1, _h1) in zip(samples, samples[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        total += float(np.linalg.norm(p1 - p0)) / max(h0, 1.0)
        seconds += dt
    if seconds <= 0:
        return None
    return total / seconds


def _reversals(signal: list) -> int:
    """Direction changes in a 1-D signal.

    This is what tells waving from reaching and walking from standing still:
    both pairs differ by whether the motion comes back, not by how fast it is.
    """
    directions = [b - a for a, b in zip(signal, signal[1:])]
    directions = [d for d in directions if abs(d) > 1e-9]
    return sum(1 for a, b in zip(directions, directions[1:]) if a * b < 0)


def _body_speed(frames: list) -> Optional[float]:
    """Largest per-joint speed over the window, body-heights per second."""
    speeds = [s for s in (_joint_speed(frames, name) for name in COCO_INDEX)
              if s is not None]
    return max(speeds) if speeds else None


# ── posture rules (single frame) ─────────────────────────────────────────────

def _posture_labels(frame: PoseFrame, th: dict) -> list:
    """Postures that hold for this frame, as (label, confidence) pairs.

    Returns empty when the body is too occluded to say — which is the whole
    point of the has_torso / has_legs gates. A person behind a desk gives a
    perfectly good torso axis and no legs at all, and sitting and standing
    have the *same* torso axis.
    """
    out = []
    if frame.torso_deg is None:
        return out

    # lying needs no legs: the torso axis and the bbox shape settle it, and a
    # person on the floor usually has their legs in frame anyway.
    if frame.torso_deg >= th["lying_deg"] and frame.aspect >= th["lying_aspect"]:
        out.append(("lying", _confidence(frame.torso_deg, th["lying_deg"], below=False)))

    if not frame.has_legs:
        return out

    knee = frame.knee_deg
    upright = frame.torso_deg <= th["upright_deg"]

    if (upright and th["knee_bent_min_deg"] <= knee <= th["knee_bent_max_deg"]
            and frame.hip_knee_dy is not None
            and abs(frame.hip_knee_dy) <= th["sit_hip_knee_dy"]):
        out.append(("sitting", _confidence(abs(frame.hip_knee_dy),
                                           th["sit_hip_knee_dy"])))

    if (knee < th["knee_bent_min_deg"] and frame.hip_ankle_dy is not None
            and frame.hip_ankle_dy <= th["crouch_hip_ankle_dy"]):
        out.append(("crouching", _confidence(knee, th["knee_bent_min_deg"])))

    # Straight legs are what separates bending from crouching: both put the
    # torso over the floor, but only one folds the knees.
    if (th["bend_deg"] <= frame.torso_deg < th["lying_deg"]
            and knee >= th["leg_straight_deg"]):
        out.append(("bending", _confidence(frame.torso_deg, th["bend_deg"],
                                           below=False)))

    if (upright and knee >= th["leg_straight_deg"]
            and frame.hip_knee_dy is not None
            and frame.hip_knee_dy >= th["stand_hip_knee_dy"]):
        out.append(("standing", _confidence(frame.torso_deg, th["upright_deg"])))

    return out


# ── arm rules (frame + short window) ────────────────────────────────────────

def _raised_sides(frame: PoseFrame, th: dict) -> list:
    """Sides whose wrist is above the shoulder with the elbow not folded shut."""
    sides = []
    for side in ("left", "right"):
        shoulder = frame.joint(f"{side}_shoulder")
        elbow = frame.joint(f"{side}_elbow")
        wrist = frame.joint(f"{side}_wrist")
        if shoulder is None or wrist is None:
            continue
        rise = (float(shoulder[1]) - float(wrist[1])) / frame.height
        if rise < th["raise_wrist_above_shoulder"]:
            continue
        elbow_deg = _angle_deg(shoulder, elbow, wrist)
        # No visible elbow: the wrist being above the shoulder is still the
        # thing being claimed, so this is accepted rather than dropped.
        if elbow_deg is not None and elbow_deg < th["elbow_open_deg"]:
            continue
        sides.append((side, rise))
    return sides


def _arm_labels(frames: list, th: dict) -> list:
    """Arm/interaction labels, as (label, confidence, extra) triples."""
    current = frames[-1]
    out = []
    raised_now = _raised_sides(current, th)

    for side, rise in raised_now:
        # Held, not glimpsed: an arm swinging through shoulder height on one
        # frame is not someone raising their hand.
        held_from = None
        for frame in reversed(frames):
            if side not in [s for s, _ in _raised_sides(frame, th)]:
                break
            held_from = frame.t
        held_s = 0.0 if held_from is None else current.t - held_from
        if held_s < th["raise_hold_s"]:
            continue
        out.append(("raising_hand",
                    _confidence(rise, th["raise_wrist_above_shoulder"], below=False),
                    {"side": side, "held_s": round(held_s, 2)}))

        # Waving is raising_hand plus "it comes back". Gating it on the raise
        # is what keeps an arm swinging while walking from reading as a wave.
        xs, ts = [], []
        for frame in frames:
            wrist = frame.joint(f"{side}_wrist")
            shoulder = frame.joint(f"{side}_shoulder")
            if wrist is None or shoulder is None:
                continue
            xs.append(float(wrist[0] - shoulder[0]) / frame.height)
            ts.append(frame.t)
        if len(xs) < 3:
            continue
        reversals = _reversals(xs)
        amplitude = max(xs) - min(xs)
        duration = ts[-1] - ts[0]
        cycles = reversals / 2.0
        frequency = cycles / duration if duration > 0 else 0.0
        if (reversals >= th["wave_reversals"]
                and amplitude >= th["wave_amplitude"]
                and th["wave_freq_min_hz"] <= frequency <= th["wave_freq_max_hz"]):
            out.append(("waving",
                        _confidence(amplitude, th["wave_amplitude"], below=False),
                        {"side": side, "reversals": reversals,
                         "amplitude": round(amplitude, 3),
                         "frequency_hz": round(frequency, 2)}))

    for side in ("left", "right"):
        shoulder = current.joint(f"{side}_shoulder")
        elbow = current.joint(f"{side}_elbow")
        wrist = current.joint(f"{side}_wrist")
        if shoulder is None or wrist is None or elbow is None:
            continue
        if (_angle_deg(shoulder, elbow, wrist) or 0.0) < th["arm_straight_deg"]:
            continue
        reach = wrist - shoulder
        norm = float(np.linalg.norm(reach))
        if norm < 1e-6:
            continue
        from_horizontal = math.degrees(math.asin(min(1.0, abs(float(reach[1])) / norm)))
        if from_horizontal > th["point_horizontal_deg"]:
            continue
        other = "right" if side == "left" else "left"
        other_speed = _joint_speed(frames, f"{other}_wrist")
        if other_speed is not None and other_speed > th["still_speed"]:
            continue
        out.append(("pointing",
                    _confidence(from_horizontal, th["point_horizontal_deg"]),
                    # The direction is worth more than the label: "someone is
                    # pointing" is far less actionable than where.
                    {"side": side,
                     "point_direction": [round(float(reach[0] / norm), 3),
                                         round(float(reach[1] / norm), 3)]}))

    crossed = 0
    if current.shoulder is not None and current.hip is not None:
        midline = (float(current.shoulder[0]) + float(current.hip[0])) / 2.0
        chest_top, chest_bottom = float(current.shoulder[1]), float(current.hip[1])
        for side in ("left", "right"):
            wrist = current.joint(f"{side}_wrist")
            shoulder = current.joint(f"{side}_shoulder")
            if wrist is None or shoulder is None:
                continue
            own_side = float(shoulder[0]) - midline
            wrist_side = float(wrist[0]) - midline
            if own_side * wrist_side < 0 and chest_top <= float(wrist[1]) <= chest_bottom:
                crossed += 1
    if crossed == 2:
        out.append(("arms_crossed", 0.7, {}))

    return out


# ── motion rules (window only) ──────────────────────────────────────────────

def _motion_labels(frames: list, th: dict, postures: list) -> list:
    out = []
    duration = frames[-1].t - frames[0].t
    if duration <= 0 or len(frames) < 3:
        return out

    # `still` needs the same structural basis a posture does. "They are not
    # moving" derived from a nose and one shoulder is a confident claim built
    # on nothing, and it lands on exactly the frames where the honest answer is
    # `unknown` — a person behind a desk would come back as motionless rather
    # than as unreadable, which is the failure this plugin is supposed to avoid.
    # walking and turning carry their own structural requirements (two ankles,
    # two shoulders), so only this one needs the gate.
    speed = _body_speed(frames)
    if (frames[-1].has_torso and speed is not None
            and speed <= th["still_speed"]):
        out.append(("still", _confidence(speed, th["still_speed"]), {}))

    separations, widths = [], []
    for frame in frames:
        la, ra = frame.joint("left_ankle"), frame.joint("right_ankle")
        if la is not None and ra is not None:
            separations.append(float(la[0] - ra[0]) / frame.height)
        ls, rs = frame.joint("left_shoulder"), frame.joint("right_shoulder")
        if ls is not None and rs is not None:
            widths.append(abs(float(ls[0] - rs[0])) / frame.height)

    # Walking is gated on an upright torso, so a seated person fidgeting their
    # feet is not reported as walking.
    upright = any(label in ("standing", "bending") for label, _ in postures)
    if len(separations) >= 3 and upright:
        reversals = _reversals(separations)
        amplitude = max(separations) - min(separations)
        cadence = (reversals / 2.0) / duration
        if (amplitude >= th["walk_amplitude"]
                and th["walk_cadence_min_hz"] <= cadence <= th["walk_cadence_max_hz"]):
            out.append(("walking",
                        _confidence(amplitude, th["walk_amplitude"], below=False),
                        {"cadence_hz": round(cadence, 2),
                         "amplitude": round(amplitude, 3)}))

    if len(widths) >= 3:
        widest = max(widths)
        change = abs(widths[-1] - widths[0]) / widest if widest > 1e-6 else 0.0
        if change >= th["turn_width_change"]:
            out.append(("turning",
                        _confidence(change, th["turn_width_change"], below=False),
                        {"shoulder_width_change": round(change, 3)}))

    return out


# ── fall: the one label that judges a transition, not a state ───────────────

def _fall_evidence(frames: list, postures: list, th: dict) -> Optional[dict]:
    """Decide whether the person now on the floor *fell* onto it.

    Lying on the floor and lying on a sofa are the same terminal state, so a
    posture test cannot tell them apart and neither can a single frame. What
    distinguishes a fall is the transition into it:

      1. the hips drop more than `fall_drop_ratio` of standing height
      2. in under `fall_drop_window_s`
      3. and the body stays horizontal for `fall_settle_s` afterwards
      4. with no `sitting` phase on the way down — sitting down is slow and
         has an unambiguous knee angle

    Returns the evidence whenever the person is settled and horizontal, with
    `is_fall` saying whether the drop was there too. The rejected case is
    returned rather than swallowed on purpose: "they are lying down and here is
    why we did not call it a fall" is what someone reads when asking why the
    robot said nothing.

    Height comes from the tallest box in the window, not the current one: a
    person on the floor has a short wide box, so normalising by the current
    height would divide the drop by the post-fall height and understate it.
    """
    if not frames or "lying" not in [label for label, _ in postures[-1]]:
        return None

    settle_start = len(frames) - 1
    while settle_start > 0 and "lying" in [label for label, _ in postures[settle_start - 1]]:
        settle_start -= 1
    settle_s = frames[-1].t - frames[settle_start].t
    if settle_s < th["fall_settle_s"]:
        # Horizontal, but not for long enough to tell a fall from bending to
        # pick something up. Deliberately not a fall *yet* rather than a
        # negative — the next frames decide.
        return None

    reference_height = max(frame.height for frame in frames)
    landed = frames[settle_start]
    if landed.hip is None:
        return {"is_fall": False, "reason": "hips not visible at landing",
                "settle_ms": int(settle_s * 1000)}

    best = None
    for index in range(settle_start, -1, -1):
        gap = landed.t - frames[index].t
        if gap > th["fall_drop_window_s"]:
            break
        start = frames[index]
        if start.hip is None:
            continue
        drop = float(landed.hip[1] - start.hip[1]) / reference_height
        if best is None or drop > best[0]:
            best = (drop, gap, index)

    if best is None:
        return {"is_fall": False, "reason": "no hip reading before landing",
                "settle_ms": int(settle_s * 1000)}

    drop_ratio, drop_s, from_index = best
    had_sitting = any(
        "sitting" in [label for label, _ in postures[i]]
        for i in range(from_index, settle_start + 1)
    )
    evidence = {
        "drop_ratio": round(drop_ratio, 3),
        "drop_ms": int(drop_s * 1000),
        "settle_ms": int(settle_s * 1000),
        "had_sitting_phase": had_sitting,
    }
    if drop_ratio < th["fall_drop_ratio"]:
        evidence["is_fall"] = False
        evidence["reason"] = "no fast drop into the horizontal pose"
        return evidence
    if had_sitting:
        evidence["is_fall"] = False
        evidence["reason"] = "passed through sitting on the way down"
        return evidence
    evidence["is_fall"] = True
    evidence["confidence"] = _confidence(drop_ratio, th["fall_drop_ratio"],
                                         below=False)
    return evidence


# ── classifier ──────────────────────────────────────────────────────────────

class PoseActionClassifier:
    """Rules backend: keypoint sequence in, action labels out.

    `action_backend` on the plugin selects this; a learned skeleton-action model
    would replace this class and nothing else, which is why `classify` takes
    plain PoseFrames and returns a plain dict.
    """

    def __init__(self, thresholds: Optional[dict] = None,
                 action_window_s: float = 1.5):
        self.thresholds = dict(DEFAULT_THRESHOLDS)
        self.thresholds.update({k: v for k, v in (thresholds or {}).items()
                                if k in DEFAULT_THRESHOLDS and v is not None})
        self.action_window_s = float(action_window_s)

    @property
    def history_s(self) -> float:
        """How much history the tracker has to keep for these rules to work.

        Fall needs the drop window *plus* the settle window, which is longer
        than the action window — ask for too little and fall can never fire,
        with nothing in any log to say why.
        """
        th = self.thresholds
        return max(self.action_window_s,
                   th["fall_drop_window_s"] + th["fall_settle_s"] + 0.5)

    def classify(self, frames: list) -> dict:
        th = self.thresholds
        if not frames:
            return {"action": "unknown", "actions": [], "action_confidence": 0.0,
                    "evidence": {"reason": "no frames"}}

        current = frames[-1]
        window = [f for f in frames
                  if current.t - f.t <= self.action_window_s] or [current]
        postures = [_posture_labels(f, th) for f in frames]

        scored: dict = {}
        extras: dict = {}

        def _offer(label, confidence, extra=None):
            if label not in scored or confidence > scored[label]:
                scored[label] = float(confidence)
                extras[label] = dict(extra or {})

        for label, confidence in postures[-1]:
            _offer(label, confidence)
        for label, confidence, extra in _arm_labels(window, th):
            _offer(label, confidence, extra)
        for label, confidence, extra in _motion_labels(window, th, postures[-1]):
            _offer(label, confidence, extra)

        fall = _fall_evidence(frames, postures, th)
        if fall is not None:
            if fall.get("is_fall"):
                _offer("fall", fall.get("confidence", 0.5), fall)
            else:
                # Attach the rejected evidence to `lying`, so the reason the
                # robot did not raise an alarm is readable.
                extras.setdefault("lying", {}).update(fall)

        # Events bypass min_confidence — they have their own, stricter gates.
        held = {label: conf for label, conf in scored.items()
                if conf >= th["min_confidence"] or label in EVENT_ACTIONS}

        actions = [label for label in ACTION_PRIORITY if label in held]
        if not actions:
            reason = ("occluded" if not current.has_torso
                      else "no rule matched with enough confidence")
            return {
                "action": "unknown",
                "actions": [],
                "action_confidence": 0.0,
                "evidence": {"reason": reason,
                             "visible_keypoints": current.visible_count(),
                             "has_torso": current.has_torso,
                             "has_legs": current.has_legs},
            }

        primary = actions[0]
        result = {
            "action": primary,
            "actions": actions,
            "action_confidence": round(held[primary], 2),
            "evidence": extras.get(primary, {}),
        }
        # Promoted out of `evidence` because it is the actionable number:
        # "someone is pointing" is far less useful than where they point.
        direction = extras.get("pointing", {}).get("point_direction")
        if direction is not None:
            result["point_direction"] = direction
        return result


# ── tracking ────────────────────────────────────────────────────────────────

def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class PoseTrack:
    """One person's timeline. `history` is what the classifier reads."""

    __slots__ = ("id", "history", "last_seen")

    def __init__(self, track_id: int):
        self.id = track_id
        self.history: deque = deque()
        self.last_seen = 0.0

    @property
    def current(self) -> Optional[PoseFrame]:
        return self.history[-1] if self.history else None


class PoseTracker:
    """Greedy IoU association, so actions have a timeline to be computed over.

    This is NOT re-identification and does not pretend to be: a person who
    leaves the frame and comes back gets a new id. Recognising *who* someone is
    is the `face_recognition` card's job, and conflating the two would promise
    an identity this cannot keep.
    """

    def __init__(self, history_s: float = 3.0, iou_min: float = 0.2,
                 timeout_s: float = 1.0, min_conf: float = 0.3):
        self.history_s = float(history_s)
        self.iou_min = float(iou_min)
        self.timeout_s = float(timeout_s)
        self.min_conf = float(min_conf)
        self._tracks: list = []
        self._next_id = 1

    @property
    def tracks(self) -> list:
        return list(self._tracks)

    def reset(self) -> None:
        self._tracks = []

    def update(self, boxes, keypoints, now: float) -> list:
        """Associate this frame's detections, returning one track per detection
        in the order the detections came in (so the caller can zip them)."""
        self._expire(now)

        boxes = [list(map(float, box)) for box in boxes]
        pairs = []
        for d_index, box in enumerate(boxes):
            for t_index, track in enumerate(self._tracks):
                current = track.current
                if current is None:
                    continue
                score = _iou(box, current.box)
                if score >= self.iou_min:
                    pairs.append((score, d_index, t_index))
        pairs.sort(reverse=True)

        taken_d: set = set()
        taken_t: set = set()
        assigned: dict = {}
        for _score, d_index, t_index in pairs:
            if d_index in taken_d or t_index in taken_t:
                continue
            taken_d.add(d_index)
            taken_t.add(t_index)
            assigned[d_index] = self._tracks[t_index]

        out = []
        for d_index, box in enumerate(boxes):
            track = assigned.get(d_index)
            if track is None:
                track = PoseTrack(self._next_id)
                self._next_id += 1
                self._tracks.append(track)
            frame = PoseFrame(now, box, np.asarray(keypoints[d_index],
                                                   dtype=np.float32),
                              self.min_conf)
            track.history.append(frame)
            track.last_seen = now
            self._trim(track, now)
            out.append(track)
        return out

    def _trim(self, track, now: float) -> None:
        while track.history and now - track.history[0].t > self.history_s:
            track.history.popleft()

    def _expire(self, now: float) -> None:
        self._tracks = [t for t in self._tracks
                        if now - t.last_seen <= self.timeout_s]


def action_catalogue() -> list:
    """What `list_actions` answers with."""
    return [
        {
            "action": name,
            "label_zh": ACTION_LABELS_ZH[name],
            "kind": "event" if name in EVENT_ACTIONS else "pose",
        }
        for name in ACTION_PRIORITY
    ]

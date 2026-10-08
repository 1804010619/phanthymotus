#!/usr/bin/env python3
"""
plugins/pose_stgcn.py — ST-GCN++ skeleton-action backend for the pose card.

The learned alternative to `plugins/pose_action.py`'s geometry. Takes the same
per-person `PoseFrame` history and returns the same dict, so the plugin does not
know which one it is talking to — that interface is why `classify()` has always
taken plain frames and returned a plain dict.

**Model: ST-GCN++, joint stream, NTU60-XSub, 2D 17-keypoint input.** 1.39 M
parameters and 1.95 GFLOPs at 100 frames (MMAction2's figures for the
NTU60-XSub-2D setting), top-1 89.3%. On the same benchmark ST-GCN is
3.1 M/3.8 G, AGCN 3.5 M/4.4 G and CTR-GCN 1.43 M/2.82 G, so this is both the
smallest and the most accurate of the options with published 2D-COCO17
checkpoints. Single joint stream rather than the four-stream ensemble: 4x the
compute for +3.9 points is not the small choice, and the card already spends
24.1 GFLOPs on `yolo26s-pose` ahead of it.

**TensorRT, not ONNX Runtime.** Not a performance preference — a second ONNX
Runtime in this process shares one provider bridge with the first, which throws
on jp6.1 and SIGSEGVs the whole of perception on jp5.11. See
`plugins/kokoro_worker.py`.

**What this backend cannot do, and why the default is `hybrid`.** NTU-60's label
space is built from *transitions and events*, not postures: it has `sit down`
(A08), `stand up` (A09), `falling down` (A43), `hand waving` (A23), `pointing to
something with finger` (A31). There is no `standing` class and no `sitting`
class, because a motionless person is not an action. So a straight swap would
lose the posture labels entirely. `hybrid` runs this backend for what it is good
at — the dynamic and event labels — and keeps the (now angle-based) geometry for
the postures.

**The preprocessing is the part that fails silently.** PYSKL's pipeline is
`PreNormalize2D` → `GenSkeFeat('j')` → `UniformSample(T)` → `FormatGCNInput`,
and a model fed a differently-normalised skeleton returns confident nonsense
rather than an error. It is reimplemented here rather than imported, because
pulling in mmaction2/pyskl would drag torch and a dependency tree into an image
that deliberately carries neither. The version-sensitive surface is kept to two
calls — `_build_input` and `_run` — which is the shape
`actucore/tests/test_smolvla_provider.py` established for a provider whose model
cannot be loaded on a laptop.

**Unverified against real weights.** There is no checkpoint published for this
yet, so everything below is covered against a fake engine only: the tensor
shapes, the normalisation arithmetic, the sampling, the label mapping and the
failure paths. Whether ST-GCN++ agrees with our labels on a real robot is not
something these tests can establish. Treat "merged" as "the plumbing is right".
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import numpy as np

from plugins.pose_action import (
    ACTION_PRIORITY,
    DEFAULT_THRESHOLDS,
    EVENT_ACTIONS,
    PoseActionClassifier,
)
from plugins.vision_runtime import COCO_INDEX, N_KEYPOINTS

log = logging.getLogger(__name__)

# Frames the network is fed. **100**, because that is what the published
# checkpoint was trained with (`UniformSample(clip_len=100)` in
# configs/stgcn++/stgcn++_ntu60_xsub_hrnet/j.py) — not a number we get to pick.
# `_build_input` still reads the engine's own declaration and prefers it; this
# is the fallback when the engine does not state one.
DEFAULT_WINDOW_FRAMES = 100

# Person slots the network expects. **2**, from `FormatGCNInput(num_person=2)`
# in the same config: the model was trained on a tensor with room for two
# skeletons, and a single person is zero-padded into the second slot. Feeding
# M=1 is not "the same thing with less data" — it is a different tensor shape,
# and the batch-norm the backbone applies over N*M would see a different
# population.
NUM_PERSON_SLOTS = 2

# Channels per joint: x, y **and the detector's confidence**. Confirmed against
# the checkpoint rather than assumed — `backbone.data_bn.weight` has 51 entries,
# which is 3 x 17, and the first conv takes 3 input channels. PreNormalize2D
# concatenates `keypoint_score` as that third channel.
NUM_CHANNELS = 3

# PreNormalize2D zeroes the x and y of any joint at or below this score, keeping
# the score itself. Not a visibility threshold of ours — it is part of the
# transform the weights were trained under.
PRENORM_SCORE_THRESHOLD = 0.01

# How much video the window should span. ST-GCN needs an action to be *in* the
# clip: NTU samples are 1-5 s. At the card's default 5 fps a 2 s window is 10
# real frames, which is thin — hence the fps coupling the plugin warns about.
DEFAULT_WINDOW_S = 2.5

# NTU-60 class index → our label. Only the classes this project has a label for
# and that a single-person skeleton can support; everything else maps to None and
# is treated as "this model has nothing to say", not as `unknown`.
#
# Indices are 0-based, i.e. NTU's A01 is 0. Mutual (two-person) classes A50-A60
# are deliberately absent: they are defined on a *pair* of skeletons and this
# backend is fed one person at a time, so a confident prediction there would be
# meaningless. That includes A59/A60 walking-towards/apart, which is why
# `walking` is not available from this backend and stays with the geometry.
NTU60_TO_ACTION = {
    7:  "sitting",        # A08 sit down      (transition → resulting state)
    8:  "standing",       # A09 stand up
    22: "waving",         # A23 hand waving
    30: "pointing",       # A31 pointing to something with finger
    42: "fall",           # A43 falling down
}

# Classes that are *transitions* in NTU but states for us. Reporting `sitting`
# because the model saw "sit down" is an inference beyond what it was asked, so
# in `hybrid` these defer to the geometry, which observes the state directly.
NTU60_TRANSITIONS = {7, 8}

PRENORM_MODE = "auto"

DEFAULT_MIN_SCORE = 0.40

# `fall` is held to a higher bar than the rest, and measured evidence says it
# has to be. Feeding the built engine **pure Gaussian noise** returns
# A43 "falling down" at **0.62** — above the general threshold, from a skeleton
# that is not a body at all. The class is evidently where this network puts
# input it cannot parse, which is the worst possible default for the one label a
# robot acts on: a false fall makes it drop what it is doing to ask whether
# somebody is hurt.
#
# Two guards, because neither alone is enough. This threshold, and in `hybrid`
# the requirement that the geometry agree the person is not upright — a
# corroboration the noise case cannot produce.
FALL_MIN_SCORE = 0.75


class ActionBackendError(RuntimeError):
    """Raised when the engine is unusable. Never swallowed into a label."""


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable, because an engine may emit raw logits or probabilities
    and we have to be able to threshold either."""
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    total = exp.sum()
    return exp / total if total > 0 else np.full_like(exp, 1.0 / exp.size)


def looks_like_probabilities(values: np.ndarray) -> bool:
    """Has this engine already applied a softmax?

    Decided by content, the same way `decode_poses` picks its tensor: an export
    with the softmax folded in and one without differ in nothing but the numbers,
    and running softmax twice flattens the distribution towards uniform — which
    shows up as every score sitting below the threshold and the backend
    reporting nothing, with no error anywhere.
    """
    if values.size == 0:
        return False
    return bool(np.all(values >= -1e-4) and np.all(values <= 1.0 + 1e-4)
                and abs(float(values.sum()) - 1.0) < 1e-2)


def uniform_sample_indices(available: int, wanted: int) -> np.ndarray:
    """PYSKL's UniformSample, deterministic (centre of each bin).

    Bins the clip into `wanted` equal spans and takes the middle of each.
    Random offsets are for training; a robot wants the same answer twice for
    the same input.

    Used for **downsampling only** — see `resample_clip`.
    """
    if available <= 0:
        raise ActionBackendError("cannot sample an empty keypoint sequence")
    edges = np.linspace(0, available, wanted + 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    return np.clip(centres.astype(np.int64), 0, available - 1)


def resample_clip(clip: np.ndarray, wanted: int) -> np.ndarray:
    """Fit a (T0, V, C) clip to exactly `wanted` frames.

    **Downsampling repeats upstream's index selection; upsampling interpolates,
    and that difference is measured, not stylistic.**

    NTU clips run 2-5 s at 30 fps, so upstream is almost always *reducing* a
    clip to 100 frames and index selection is what the weights were fitted on.
    Our situation is the reverse: a 2.5 s window at the card's 12 fps holds 30
    real frames and the network wants 100. Repeating each frame 3.3 times
    produces a staircase — plateaus of zero velocity separated by jumps — and an
    ST-GCN's temporal convolutions see velocity, so that is a motion signature
    the model was never trained on.

    Measured on a real falling skeleton, A43 "falling down":

        100 real frames, smooth                 0.948
        30 real frames repeated out to 100      0.572   <- below the fall bar
        30 real frames interpolated to 100      see tests

    0.572 sits under FALL_MIN_SCORE, so the repeat-upsampled fall was being
    withheld entirely — the model had seen the event and said so, and the
    resampler had taken most of its confidence away first.

    Interpolation is linear in time, between the two nearest real frames. The
    score channel rides along: a joint that was uncertain in both neighbours
    stays uncertain, which is what the downstream masking expects.
    """
    available = len(clip)
    if available <= 0:
        raise ActionBackendError("cannot resample an empty keypoint sequence")
    if available == wanted:
        return clip.astype(np.float32, copy=True)
    if available > wanted:
        return clip[uniform_sample_indices(available, wanted)].astype(np.float32)

    positions = np.linspace(0.0, available - 1.0, wanted)
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, available - 1)
    weight = (positions - lower).astype(np.float32)[:, None, None]
    return (clip[lower] * (1.0 - weight) + clip[upper] * weight).astype(np.float32)


def pre_normalize_2d(keypoints: np.ndarray, image_size, mode: str = "auto") -> np.ndarray:
    """PYSKL `PreNormalize2D`, both modes. (T, V, 3) in — x, y, score — out.

    This is the step whose details decide whether the network sees what it was
    trained on, and getting it wrong raises nothing, so the arithmetic is
    transcribed rather than paraphrased.

    `fix` is what the checkpoint was trained with: divide by the frame.

        keypoint[..., 0] = (x - w / 2) / (w / 2)
        keypoint[..., 1] = (y - h / 2) / (h / 2)

    `auto` centres and scales by the **clip's own extent** instead.

    **We default to `auto`, and the measurement is why.** The same real fall, a
    real skeleton moving, at four distances, scored on A43 "falling down":

        person fills ~42% of frame height    fix 0.950    auto 0.948
        half that                            fix 0.600    auto 0.948
        a quarter                            fix 0.074    auto 0.948
        an eighth                            fix 0.003    auto 0.948

    `fix` keeps the person's size and position in frame, which I argued was
    information the model trained with. It is worth ~nothing at the training
    scale (0.950 against 0.948) and is actively destructive away from it,
    because NTU's subjects all fill a similar fraction of frame and a skeleton
    from further away lands outside the distribution entirely. A robot sees
    people across a room; `auto` puts every one of them into the scale the
    weights were fitted on.

    `fix` remains selectable, because it is what the training pipeline used and
    a future checkpoint may be less tolerant.

    One detail both modes share: a joint scoring at or below 0.01 has its x and
    y **zeroed while its score is kept**. That is part of the transform the
    weights were trained under, not our visibility gate.
    """
    out = keypoints.astype(np.float32, copy=True)
    absent = (out[..., 2] <= PRENORM_SCORE_THRESHOLD
              if out.shape[-1] >= 3 else np.zeros(out.shape[:-1], dtype=bool))

    if mode == "auto":
        present = ~absent
        if present.any():
            xs, ys = out[..., 0][present], out[..., 1][present]
            x_max, x_min = float(xs.max()), float(xs.min())
            y_max, y_min = float(ys.max()), float(ys.min())
            # Upstream's guard: a body spanning under 10 px is not a body, and
            # dividing by it would amplify noise into coordinates.
            if (x_max - x_min) > 10 and (y_max - y_min) > 10:
                out[..., 0] = (out[..., 0] - (x_max + x_min) / 2) / (x_max - x_min) * 2
                out[..., 1] = (out[..., 1] - (y_max + y_min) / 2) / (y_max - y_min) * 2
    elif mode == "fix":
        width, height = image_size
        if not (width > 0 and height > 0):
            raise ActionBackendError(
                f"frame size {image_size!r} is unusable; keypoints are in frame "
                "pixels, so there is nothing to normalise against"
            )
        out[..., 0] = (out[..., 0] - width / 2.0) / (width / 2.0)
        out[..., 1] = (out[..., 1] - height / 2.0) / (height / 2.0)
    else:
        raise ActionBackendError(
            f"unknown normalisation mode {mode!r}; expected 'auto' or 'fix'")

    if out.shape[-1] >= 3:
        out[..., 0][absent] = 0.0
        out[..., 1][absent] = 0.0
    return out


class SkeletonActionBackend:
    """ST-GCN++ over a person's keypoint history.

    Interchangeable with `PoseActionClassifier`: same `classify(frames)` /
    `classify_frame(frame)` / `history_s`, same returned dict.
    """

    def __init__(self, *, window_s: float = DEFAULT_WINDOW_S,
                 min_score: float = DEFAULT_MIN_SCORE,
                 fall_min_score: float = FALL_MIN_SCORE,
                 prenorm_mode: str = PRENORM_MODE,
                 model_dir: Optional[str] = None,
                 engine=None, thresholds: Optional[dict] = None,
                 action_window_s: Optional[float] = None):
        self.window_s = float(action_window_s or window_s)
        self.min_score = float(min_score)
        self.fall_min_score = float(fall_min_score)
        self.prenorm_mode = str(prenorm_mode)
        self.thresholds = dict(DEFAULT_THRESHOLDS)
        self.thresholds.update({k: v for k, v in (thresholds or {}).items()
                                if k in DEFAULT_THRESHOLDS and v is not None})
        self._model_dir = model_dir or os.environ.get("ACTION_MODEL_DIR",
                                                      "/models/action")
        self._engine = engine          # injectable, which is what makes this testable
        self._lock = threading.Lock()
        self._load_error: Optional[str] = None

    # ── engine ───────────────────────────────────────────────────────────

    @property
    def last_error(self) -> Optional[str]:
        """The most recent engine failure, or None.

        Sticky on purpose, and surfaced by the card's `info`: there is no
        published action engine yet, so the common case is a backend that
        constructs fine and then cannot run. Without this the card reports
        `action_backend: hybrid` while silently answering from geometry.
        """
        return self._load_error

    @property
    def history_s(self) -> float:
        """Enough history to fill the window, and no attempt to also cover the
        rules' fall timings — this backend judges the fall itself."""
        return max(self.window_s, 1.0)

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is not None:
                return self._engine
            from utils.model_downloader import ensure_action_model
            from utils.model_progress import fetch_status

            progress_cb, _ = fetch_status(lambda text: None, "stgcn++")
            paths = ensure_action_model(self._model_dir, progress_cb=progress_cb)
            engine = next(p for name, p in paths.items() if name.endswith(".engine"))
            log.info("[pose/stgcn] loading action engine: %s", engine)
            # TensorRTEngine directly, not VisionEngineSession: that class
            # letterboxes an image, and this engine takes a skeleton tensor.
            from utils.tensorrt_runtime import TensorRTEngine
            self._engine = TensorRTEngine(engine)
            return self._engine

    def _window_frames(self) -> int:
        """Frames the engine wants, from the engine itself where possible.

        An ST-GCN export has a fixed temporal dimension, and feeding it a
        different one is a shape error at best and a silent reinterpretation at
        worst — so the engine's own declaration wins over our constant.
        """
        engine = self._engine
        shape = getattr(engine, "input_shape", None) or getattr(
            engine, "optimization_shape", None)
        # (N, M, T, V, C) is PYSKL's FormatGCNInput order, so T is index 2.
        if shape is not None and len(shape) == 5 and int(shape[2]) > 0:
            return int(shape[2])
        return DEFAULT_WINDOW_FRAMES

    # ── input construction ───────────────────────────────────────────────

    def _build_input(self, frames: list, window_frames: int) -> np.ndarray:
        """PoseFrames → (1, 2, T, 17, 3) float32, PYSKL's FormatGCNInput order.

        One of the two version-sensitive calls in this class. Steps, in PYSKL's
        order: joint stream with the score channel, normalise by the frame,
        uniform-sample to T, then pad to two person slots.

        The shape is taken from the published config, not chosen:
        `FormatGCNInput(num_person=2)` means the network was trained on a tensor
        with room for two skeletons and a single person zero-padded into the
        second slot. An earlier version of this method built (1, 1, T, 17, 2) —
        wrong in two dimensions, which is precisely the kind of mistake that
        loads without complaint and returns confident nonsense.
        """
        if not frames:
            raise ActionBackendError("no frames to classify")
        # `auto` derives its scale from the skeleton, so the frame is only
        # required by `fix`. Asking for it unconditionally would refuse frames
        # that the default mode has no use for.
        image_size = _frame_size(frames) if self.prenorm_mode == "fix" else (0, 0)
        arrays = [np.asarray(f.keypoints, dtype=np.float32) for f in frames]
        # Checked before the stack, not after: np.stack on a ragged list fails
        # with "all input arrays must have the same shape", which says nothing
        # about joints and sends the reader to numpy rather than to the caller
        # who supplied the wrong skeleton.
        bad = {a.shape for a in arrays
               if a.ndim != 2 or a.shape[0] != N_KEYPOINTS or a.shape[1] < 3}
        if bad:
            raise ActionBackendError(
                f"every frame needs {N_KEYPOINTS} keypoints with x, y and a "
                f"score; got {sorted(bad)}")
        raw = np.stack([a[:, :NUM_CHANNELS] for a in arrays])   # (T0, V, 3)
        normalised = pre_normalize_2d(raw, image_size, self.prenorm_mode)
        sampled = resample_clip(normalised, window_frames)      # (T, V, 3)

        # (M, T, V, C) with the unused person slot zeroed, then the clip axis.
        people = np.zeros((NUM_PERSON_SLOTS,) + sampled.shape, dtype=np.float32)
        people[0] = sampled
        return people[None]                                     # (1, M, T, V, C)

    def _run(self, blob: np.ndarray) -> np.ndarray:
        """Engine call → per-class scores. The other version-sensitive call."""
        engine = self._ensure_engine()
        outputs = engine.infer(blob)
        arrays = [np.asarray(o, dtype=np.float32)
                  for o in (outputs if isinstance(outputs, (list, tuple))
                            else [outputs])]
        for array in arrays:
            flat = array.reshape(-1)
            if flat.size >= max(NTU60_TO_ACTION) + 1:
                return flat
        raise ActionBackendError(
            f"no engine output has at least {max(NTU60_TO_ACTION) + 1} classes; "
            f"got shapes {[a.shape for a in arrays]} — is this an NTU-60 head?"
        )

    # ── classification ───────────────────────────────────────────────────

    def predict(self, frames: list) -> dict:
        """Raw model view: every mappable label with its score, best first.

        Separate from `classify` so the plugin's `info` can show what the model
        actually said, which is the only way to tell "the model is wrong" from
        "the threshold is wrong" on a robot.
        """
        current = frames[-1]
        window = [f for f in frames if current.t - f.t <= self.window_s] or [current]
        self._ensure_engine()
        blob = self._build_input(window, self._window_frames())
        scores = self._run(blob)
        if not looks_like_probabilities(scores):
            scores = softmax(scores)
        ranked = sorted(
            ((label, float(scores[index]), index)
             for index, label in NTU60_TO_ACTION.items() if index < scores.size),
            key=lambda item: item[1], reverse=True,
        )
        return {
            "frames_used": len(window),
            "window_frames": self._window_frames(),
            "scores": [{"action": label, "score": round(score, 3),
                        "ntu_class": index + 1,
                        "transition": index in NTU60_TRANSITIONS}
                       for label, score, index in ranked],
        }

    def _threshold_for(self, action: str) -> float:
        """Per-class score bar. `fall` is held higher — see FALL_MIN_SCORE."""
        if action == "fall":
            return max(self.min_score, self.fall_min_score)
        return self.min_score

    def classify(self, frames: list) -> dict:
        if not frames:
            return _nothing("no frames")
        try:
            prediction = self.predict(frames)
        except ActionBackendError as error:
            # Not turned into a label: a backend that cannot run must say so,
            # because `unknown` would be indistinguishable from "nobody is
            # doing anything" and would hide a broken engine for weeks.
            self._load_error = str(error)
            log.warning("[pose/stgcn] %s", error)
            return _nothing(str(error), backend_error=True)
        except Exception as error:                      # noqa: BLE001
            self._load_error = str(error)
            log.error("[pose/stgcn] engine failure: %s", error, exc_info=True)
            return _nothing(f"engine failure: {error}", backend_error=True)

        held = [entry for entry in prediction["scores"]
                if entry["score"] >= self._threshold_for(entry["action"])]
        if not held:
            best = prediction["scores"][0] if prediction["scores"] else None
            return {
                **_nothing("no class above min_score"),
                "evidence": {
                    "reason": "no class above min_score",
                    "min_score": self.min_score,
                    "best": best,
                    "frames_used": prediction["frames_used"],
                },
            }

        actions = [entry["action"] for entry in
                   sorted(held, key=lambda e: ACTION_PRIORITY.index(e["action"]))]
        primary = actions[0]
        best = next(e for e in held if e["action"] == primary)
        return {
            "action": primary,
            "actions": actions,
            "action_confidence": round(best["score"], 2),
            "evidence": {
                "backend": "stgcn",
                "ntu_class": best["ntu_class"],
                "transition": best["transition"],
                "frames_used": prediction["frames_used"],
                "scores": prediction["scores"][:3],
            },
        }

    def classify_frame(self, frame) -> dict:
        """A single image cannot drive a temporal model at all.

        Padding one frame to T and calling it an action would produce a
        confident answer from a clip in which nothing moves. The rules backend
        answers single images; this one declines, and says which labels it
        would have needed video for.
        """
        return {
            **_nothing("a skeleton-action model needs a sequence, not one frame"),
            "temporal": False,
            "unavailable_actions": sorted(set(NTU60_TO_ACTION.values())),
        }


class HybridActionBackend:
    """Geometry for the postures, ST-GCN++ for the events. The default.

    Not a hedge. NTU-60 has no `standing` and no `sitting` class — a motionless
    person is not an action — so the learned model cannot supply the postures at
    all, while the geometry cannot see a fall as anything but its terminal pose
    without the hand-built transition rules this replaces.

    Where both have an opinion, the learned one wins on its own classes
    (`waving`, `pointing`, `fall`) and defers on NTU's *transitions*
    (`sit down`, `stand up`): inferring a state from a transition is beyond what
    the model was asked, and the geometry observes the state directly.
    """

    #: Labels the learned backend owns outright.
    LEARNED = ("fall", "waving", "pointing")

    def __init__(self, *, rules: Optional[PoseActionClassifier] = None,
                 learned: Optional[SkeletonActionBackend] = None, **kwargs):
        self.rules = rules or PoseActionClassifier(
            thresholds=kwargs.get("thresholds"),
            action_window_s=kwargs.get("action_window_s", 1.5))
        self.learned = learned or SkeletonActionBackend(**kwargs)

    @property
    def thresholds(self) -> dict:
        """The geometry's thresholds. The learned half has no thresholds to
        tune — it has `min_score`, which is a different kind of knob."""
        return self.rules.thresholds

    @property
    def last_error(self) -> Optional[str]:
        return self.learned.last_error

    @property
    def history_s(self) -> float:
        return max(self.rules.history_s, self.learned.history_s)

    def classify(self, frames: list) -> dict:
        geometry = self.rules.classify(frames)
        model = self.learned.classify(frames)

        learned_labels = [a for a in model.get("actions", []) if a in self.LEARNED]
        # `fall` additionally needs the geometry to agree the body is not
        # upright. The model returns A43 at 0.62 on pure noise, so a score bar
        # alone is one guard against the single label the robot acts on; this is
        # the second, and it is one the noise case cannot satisfy. A real fall
        # ends with the person on the ground, which the geometry can see — and
        # where it cannot (camera looking along the body) the drop detector in
        # the rules is the third path to the same label.
        if "fall" in learned_labels:
            geometry_agrees = any(a in ("lying", "fall", "crouching", "bending")
                                  for a in geometry.get("actions", []))
            withheld = None
            if not geometry_agrees:
                learned_labels = [a for a in learned_labels if a != "fall"]
                withheld = ("the model called it a fall; the geometry still "
                            "reads the body as upright")
        else:
            withheld = None
        geometry_labels = [a for a in geometry.get("actions", [])
                           if a not in self.LEARNED]
        merged = [a for a in ACTION_PRIORITY
                  if a in set(learned_labels) | set(geometry_labels)]
        if not merged:
            # Neither had anything. Keep the geometry's reason — it is the one
            # that explains occlusion, which is the common case.
            return geometry if not model.get("backend_error") else model

        primary = merged[0]
        source = "stgcn" if primary in learned_labels else "rules"
        donor = model if source == "stgcn" else geometry
        result = {
            "action": primary,
            "actions": merged,
            "action_confidence": donor.get("action_confidence", 0.0),
            "evidence": {**(donor.get("evidence") or {}), "source": source},
        }
        if geometry.get("point_direction"):
            # The geometry measures the direction; the model only names the act.
            result["point_direction"] = geometry["point_direction"]
        if model.get("backend_error"):
            result["evidence"]["stgcn_error"] = model["evidence"].get("reason")
        if withheld:
            # Recorded whichever backend won the primary label: "the model
            # thought this was a fall and was overruled" is the thing somebody
            # needs when asking why the robot stayed quiet.
            result["evidence"]["fall_withheld"] = withheld
        return result

    def classify_frame(self, frame) -> dict:
        """One image: only the geometry can answer, and it says what it cannot."""
        return self.rules.classify_frame(frame)


def _nothing(reason: str, *, backend_error: bool = False) -> dict:
    result = {
        "action": "unknown",
        "actions": [],
        "action_confidence": 0.0,
        "evidence": {"reason": reason},
    }
    if backend_error:
        # Distinct from "nobody is doing anything": a broken engine must be
        # visible as broken, not as a quiet absence of actions.
        result["backend_error"] = True
    return result


def _frame_size(frames: list):
    """The frame the keypoints are in, for PreNormalize2D.

    Refuses rather than guessing. The bounding box is in frame pixels, so its
    extent looks like a usable substitute — and it is not: a person filling the
    left half of the frame would be normalised as though the frame ended at
    their shoulder, which is a *silent* misnormalisation, and this model answers
    a misnormalised skeleton with confident nonsense rather than an error. The
    plugin knows the real size and passes it; a caller that does not has to say
    so.
    """
    for frame in frames:
        declared = getattr(frame, "image_size", None)
        if declared and declared[0] > 0 and declared[1] > 0:
            return declared
    raise ActionBackendError(
        "no frame carries an image_size, and the bounding box is not a "
        "substitute for it — PYSKL normalises a skeleton by the frame, so "
        "guessing the frame would silently misnormalise every input. Pass "
        "image_size when building PoseFrames."
    )


BACKENDS = {
    "rules": PoseActionClassifier,
    "stgcn": SkeletonActionBackend,
    "hybrid": HybridActionBackend,
}


def build_backend(name: str, **kwargs):
    """Construct the selected backend, falling back loudly rather than quietly.

    An unknown name is a config error and raising is right — a card silently
    running geometry while its config says `stgcn` is the failure mode this
    whole exercise is about.
    """
    try:
        factory = BACKENDS[name]
    except KeyError:
        raise ActionBackendError(
            f"unknown action_backend {name!r}; expected one of {sorted(BACKENDS)}"
        ) from None
    if factory is PoseActionClassifier:
        return factory(thresholds=kwargs.get("thresholds"),
                       action_window_s=kwargs.get("action_window_s", 1.5))
    return factory(**kwargs)

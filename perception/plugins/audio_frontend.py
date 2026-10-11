#!/usr/bin/env python3
"""
plugins/audio_frontend.py — keep the robot's own voice out of its own microphone.

The robot's TTS reaches its microphone acoustically, and nothing in perception
removed it until this card existed. The consequences were spread across three
plugins and none of them looked like an echo problem: ASR transcribes the robot's
own sentence and hands it to the LLM as if a person had said it; speaker
recognition enrols that voice as a person (`plugins/speaker.py` says so in a
comment); and nobody can interrupt the robot mid-sentence, because the ASR it
would have to reach is busy listening to the robot.

This is a card in the audio path, not a switch inside ASR, for one reason that
decides it: `soundevent` subscribes to the microphone topic too, and speaker
recognition runs off the segment ASR cut. Cleaning the signal inside ASR cleans
it for ASR alone. Wired as `mic -> audio_frontend -> {asr, soundevent}` it is
cleaned once, for everything downstream, and the cleaned audio is a topic a human
can listen to on the dashboard.

Two modes in this version:

  off                 pass the microphone through untouched (for A/B)
  mute_while_talking  publish silence while the robot's voice could be in the mic

`mute_while_talking` is what this card ships with, and the honest description of
it is half-duplex: it cannot be interrupted, because while the robot talks the
microphone genuinely carries nothing. It is also the only mode that works on a
robot whose speech never passes through Agent Core at all (vendor TTS over a ROS
service — Tianyi), and it removes 100% of the echo rather than 20 dB of it.
Cancellation, which keeps the mic open and allows barge-in, needs an adaptive
filter and is the next step; the delay measurement it will depend on is already
here (see _DelayTracker) precisely so that its one unanswerable parameter can be
read off a running robot rather than guessed.

### The reference signal, and why it is not a canvas connection

The card needs to know what the robot is saying. That is the TTS card's audio
output — the same PCM the speaker driver plays.

It is a config field and not a second in-port because the canvas cannot express
it: connections are matched by `format` alone (agent-core/web/js/canvas.js), a
card-level drop picks the first in-port of that format, and what reaches the
plugin is a role-less `input_topics` list ordered by click order
(web/js/topic-derive.js). Two `audio/pcm-16k` in-ports would be a coin flip.
More to the point, the reference is not input data; it is "what another card on
this machine is publishing right now", and there is almost always exactly one
right answer — so the default (`auto`) resolves it by asking the TTS plugin in
this same process, the way ASR already takes a handle to the speaker-recognition
plugin (main.py). Nobody has to fill anything in.

The cost of `auto` is that it can resolve to nothing — a robot with vendor TTS,
or a TTS card in another container — and silently degrade this card to a
passthrough. So `info()` always reports which topic it resolved to and whether
any reference audio has actually arrived, and the card's own description says so
when it has not.

### Why the mute has a tail, and why the tail is `auto` by default

Publishing the last frame of an utterance is not the end of the sound. The
speaker path buffers: the G1 driver prefills 9600 bytes (~300 ms), merges into
300 ms blocks and runs up to 240 ms ahead of the MCU's queue
(phanthymotus-driver unitree/g1/device.py), and the measured gap between "TTS
finished publishing" and "the robot stopped being audible" is 0.7-1.2 s on that
robot. A mute that ends when the reference stream ends therefore transcribes the
last few words of the robot's own sentence — which is the exact failure this card
exists to remove.

That number is a property of the robot and its speaker, nobody knows it in
advance, and it drifts with bus load. So the default is `auto`: the tail follows
the delay the card measures for itself, and the explicit values stay available
and authoritative for anyone who wants to pin it.
"""

from __future__ import annotations

import array
import copy
import logging
import math
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Mapping, Optional, Tuple

import numpy as np

from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from utils.ros_lifecycle import dispose_node

log = logging.getLogger(__name__)


# ── audio contract (perception/README.md § Audio Requirements) ───────────────

SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
AUDIO_FORMAT = "audio/pcm-16k"
# Same alias set the other audio plugins accept: some drivers still publish the
# old name and refusing them here would silence a robot that works today.
_AUDIO_FORMATS = frozenset((AUDIO_FORMAT, "pcm_16k_16bit_mono"))

# The 8-byte end-of-utterance marker plugins/tts.py publishes on its audio
# topic. It is 4 samples of square wave, not speech: counted as "the utterance
# ended", never as reference audio, and never fed to the delay tracker.
AUDIO_EOF_MAGIC = b'\x01\x00\xff\xff\x01\x00\xff\xff'

OUTPUT_SUFFIX = "/clean"

_AUDIO_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
    durability=DurabilityPolicy.VOLATILE,
)


# ── config vocabulary ────────────────────────────────────────────────────────

MODE_OFF = "off"
MODE_MUTE = "mute_while_talking"
MODES = (MODE_OFF, MODE_MUTE)
DEFAULT_MODE = MODE_MUTE

REF_AUTO = "auto"
DEFAULT_ROBOT_VOICE_TOPIC = REF_AUTO

MUTE_TAIL_AUTO = "auto"
# Explicit choices, in the units the card reports back in `info()` so the two
# can be compared by eye. Not a free-form number field: the answer is between
# 0.3 and 1.5 s on every robot measured, and a text box invites 5000.
MUTE_TAIL_CHOICES: Dict[str, Optional[float]] = {
    MUTE_TAIL_AUTO: None,
    "300ms": 0.3,
    "800ms": 0.8,
    "1500ms": 1.5,
}
DEFAULT_MUTE_TAIL = MUTE_TAIL_AUTO
# What `auto` uses before it has measured anything. 0.8 s is the middle of the
# 0.7-1.2 s measured on G1 — deliberately not the smallest of them, because
# erring short leaks the end of the robot's sentence into ASR while erring long
# only delays when the mic reopens.
AUTO_TAIL_FALLBACK_S = 0.8
# Added on top of the measured delay in `auto`, for the acoustic path and the
# jitter in the measurement itself.
AUTO_TAIL_MARGIN_S = 0.2


# ── delay measurement ────────────────────────────────────────────────────────

# 10 ms envelope hop: what is being measured is a 300-1200 ms bulk transport
# delay, and an envelope carries that as well as the waveform does at 1/160th of
# the arithmetic.
ENVELOPE_HOP_SAMPLES = 160
ENVELOPE_HOP_S = ENVELOPE_HOP_SAMPLES / SAMPLE_RATE
ENVELOPE_HISTORY_S = 6.0
# Search range. Covers G1's measured 0.7-1.2 s with room above it; a longer
# range costs nothing here (it is 120 lags of a 300-point dot product) but a
# shorter one would silently report the wrong peak on a slow speaker.
DELAY_SEARCH_MAX_S = 1.5
DELAY_WINDOW_S = 3.0
DELAY_ESTIMATE_INTERVAL_S = 1.0
# Below this much reference audio in the window there is nothing to correlate
# against and the peak would be noise wearing a confidence interval.
DELAY_MIN_REF_ACTIVE_S = 0.5
# An accepted peak has to be both strong in absolute terms and clearly above the
# rest of the lag profile. Without the second test a flat profile still has a
# maximum, and reporting its position is how a card ends up stating a delay of
# 0 ms with no echo in the room at all.
DELAY_MIN_PEAK = 0.30
DELAY_MIN_PROMINENCE = 2.0
DELAY_HISTORY = 20

# RMS below this (on int16 scale) is silence rather than speech, for deciding
# whether the robot is talking and whether a window is worth correlating.
SILENCE_RMS = 60.0


def _chunk_timestamp(message: Any) -> float:
    """Publisher's stamp, falling back to now.

    Same rule as plugins/asr.py and plugins/soundevent.py: a stamp below 1e9 is
    an unset header rather than 1970, and a producer that leaves it unset is
    common enough that refusing its audio would be a regression.
    """
    try:
        stamp = message.header.stamp
        timestamp = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except (AttributeError, TypeError, ValueError):
        return time.time()
    if not math.isfinite(timestamp) or timestamp < 1e9:
        return time.time()
    return timestamp


def _envelope(pcm: bytes) -> np.ndarray:
    """Per-hop RMS of a PCM chunk, int16 scale, dropping the partial tail hop."""
    samples = np.frombuffer(pcm, dtype='<i2')
    hops = samples.size // ENVELOPE_HOP_SAMPLES
    if hops == 0:
        return np.empty(0, dtype=np.float32)
    block = samples[:hops * ENVELOPE_HOP_SAMPLES].reshape(hops, ENVELOPE_HOP_SAMPLES)
    return np.sqrt(np.mean(np.square(block.astype(np.float32)), axis=1))


# A chunk boundary is not an envelope boundary: the documented minimum chunk is
# 512 samples (perception/README.md) and the hop is 160, so every chunk leaves a
# 32-sample remainder. Dropping it loses 6% of the signal at that size — and far
# more at the 170-sample chunks a 48 kHz USB mic produces before the driver
# buffers them. The remainder is carried into the next chunk instead, and only
# discarded when the stream is not continuous there.
MAX_ENVELOPE_JOIN_GAP_S = 0.05


class _EnvelopeStream:
    """One stream's 100 Hz RMS envelope, with the sub-hop remainder carried over.

    Separate from the tracker because the carry has a failure mode of its own:
    joining a chunk onto a remainder from before a gap would date the joined
    audio to the wrong time, which is indistinguishable downstream from a change
    in the echo delay — the one thing this whole file exists to measure.
    """

    def __init__(self) -> None:
        self.points: Deque[Tuple[float, float]] = deque()
        self._carry = b''
        self._carry_t = 0.0

    def add(self, pcm: bytes, start_t: float, now: float) -> None:
        if self._carry:
            expected = self._carry_t + (len(self._carry) // BYTES_PER_SAMPLE) / SAMPLE_RATE
            if abs(start_t - expected) <= MAX_ENVELOPE_JOIN_GAP_S:
                pcm = self._carry + pcm
                start_t = self._carry_t
            # else: a gap — the stub is dropped rather than mis-dated. At most
            # one hop of audio is lost, and the gap counter already records it.
            self._carry = b''

        hops = len(pcm) // BYTES_PER_SAMPLE // ENVELOPE_HOP_SAMPLES
        if hops:
            consumed = hops * ENVELOPE_HOP_SAMPLES * BYTES_PER_SAMPLE
            envelope = _envelope(pcm[:consumed])
            for index, value in enumerate(envelope):
                self.points.append((start_t + index * ENVELOPE_HOP_S, float(value)))
            self._carry = pcm[consumed:]
            self._carry_t = start_t + hops * ENVELOPE_HOP_S
        else:
            self._carry = pcm
            self._carry_t = start_t

        horizon = now - ENVELOPE_HISTORY_S
        while self.points and self.points[0][0] < horizon:
            self.points.popleft()

    def __len__(self) -> int:
        return len(self.points)

    def arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        count = len(self.points)
        times = np.fromiter((t for t, _ in self.points), dtype=np.float64, count=count)
        values = np.fromiter((v for _, v in self.points), dtype=np.float64, count=count)
        return times, values


class _DelayTracker:
    """How long after the robot sends audio its own voice reaches the mic.

    Normalised cross-correlation of the two envelopes over a lag range. Kept
    separate from the node so it can be tested against synthetic signals with no
    ROS, no threads and no audio hardware, which is the only way the numbers in
    its thresholds can be defended.

    Thread-safety is the caller's: the node feeds it from the two subscription
    callbacks, which `ThreadingHTTPServer` and the executor run concurrently.
    """

    def __init__(self) -> None:
        self._mic = _EnvelopeStream()
        self._ref = _EnvelopeStream()
        self._estimates: Deque[float] = deque(maxlen=DELAY_HISTORY)
        self._last_estimate_at = 0.0
        self._attempts = 0
        self._rejected = 0

    # ── feeding ──────────────────────────────────────────────────────────

    def add_mic(self, pcm: bytes, start_t: float, now: float) -> None:
        self._mic.add(pcm, start_t, now)

    def add_ref(self, pcm: bytes, start_t: float, now: float) -> None:
        self._ref.add(pcm, start_t, now)

    # ── estimating ───────────────────────────────────────────────────────

    def due(self, now: float) -> bool:
        return (now - self._last_estimate_at) >= DELAY_ESTIMATE_INTERVAL_S

    def estimate(self, now: float) -> Optional[float]:
        """Try to measure the delay, in seconds. None when the data cannot say.

        Returning None is the common case and not a failure: it means no
        reference audio was playing, or the correlation had no clear peak. A
        measurement is only offered when it is defensible.
        """
        self._last_estimate_at = now
        mic_t, mic_v = self._mic.arrays()
        ref_t, ref_v = self._ref.arrays()
        if mic_t.size < 32 or ref_t.size < 10:
            return None

        # The window ends where the microphone data ends, not at `now`. This
        # runs from the microphone callback, so the newest audio on hand is
        # always a chunk or two old — anchoring the window to the wall clock
        # made the coverage check fail every single time, which looks exactly
        # like "the correlation never finds anything".
        window_end = min(now, float(mic_t[-1]) + ENVELOPE_HOP_S)
        window_start = window_end - DELAY_WINDOW_S
        if float(mic_t[0]) > window_start:
            return None            # the mic does not cover the whole window yet

        self._attempts += 1
        points = int(round(DELAY_WINDOW_S / ENVELOPE_HOP_S))
        grid = window_start + np.arange(points) * ENVELOPE_HOP_S
        mic = np.interp(grid, mic_t, mic_v)

        # mic[t] can only carry the echo of ref[t - lag], so the reference grid
        # reaches DELAY_SEARCH_MAX_S further back. Outside the data it is zero,
        # not the nearest sample: np.interp clamps by default, which would
        # extend the last loud value of a finished utterance into a flat tail
        # and then count that tail as the robot still talking.
        lags = int(round(DELAY_SEARCH_MAX_S / ENVELOPE_HOP_S))
        ref_grid = (grid[0] - DELAY_SEARCH_MAX_S
                    + np.arange(points + lags) * ENVELOPE_HOP_S)
        ref_full = np.interp(ref_grid, ref_t, ref_v, left=0.0, right=0.0)

        active = float(np.count_nonzero(ref_full[lags:] > SILENCE_RMS)) * ENVELOPE_HOP_S
        if active < DELAY_MIN_REF_ACTIVE_S:
            return None

        mic_z = mic - mic.mean()
        mic_norm = float(np.linalg.norm(mic_z))
        if mic_norm <= 0.0:
            return None
        mic_z /= mic_norm

        profile = np.empty(lags, dtype=np.float64)
        for lag in range(lags):
            # ref_full is oldest-first and starts DELAY_SEARCH_MAX_S before the
            # mic window, so lag 0 sits at the far end of the search range.
            offset = lags - lag
            window = ref_full[offset:offset + points]
            window = window - window.mean()
            norm = float(np.linalg.norm(window))
            profile[lag] = 0.0 if norm <= 0.0 else float(np.dot(mic_z, window / norm))

        peak_index = int(np.argmax(profile))
        peak = float(profile[peak_index])
        typical = float(np.median(np.abs(profile)))
        if peak < DELAY_MIN_PEAK or peak < DELAY_MIN_PROMINENCE * max(typical, 1e-6):
            self._rejected += 1
            return None
        if peak_index >= lags - 1:
            # A peak pinned to the far end of the search range is the signature
            # of a delay *beyond* it, not of one exactly at it: measured with a
            # 2 s echo and a 1.5 s range, this bin is what comes back. Reporting
            # it would hand the gate a number that is wrong in the one direction
            # that matters, so it is refused and the range is the thing to widen.
            self._rejected += 1
            return None

        delay = peak_index * ENVELOPE_HOP_S
        self._estimates.append(delay)
        return delay

    # ── reporting ────────────────────────────────────────────────────────

    @property
    def measured(self) -> Optional[float]:
        """Median of the accepted estimates, in seconds. None before the first.

        Median rather than latest: a single estimate is one 3 s window, and the
        gate should not follow its noise.
        """
        if not self._estimates:
            return None
        return float(np.median(np.fromiter(self._estimates, dtype=np.float64)))

    @property
    def spread(self) -> Optional[float]:
        """p90-p10 of the accepted estimates, in seconds — the 'it moves' number."""
        if len(self._estimates) < 4:
            return None
        values = np.fromiter(self._estimates, dtype=np.float64)
        return float(np.percentile(values, 90) - np.percentile(values, 10))

    def report(self) -> Dict[str, Any]:
        measured = self.measured
        spread = self.spread
        return {
            "measured_echo_delay_ms": None if measured is None else round(measured * 1000),
            "echo_delay_spread_ms": None if spread is None else round(spread * 1000),
            "echo_delay_samples": len(self._estimates),
            "echo_delay_attempts": self._attempts,
            "echo_delay_rejected": self._rejected,
        }


# ── MCP tool declaration ─────────────────────────────────────────────────────

# Field names and titles are English and the choices are enumerated, because the
# people who fill this in are operators wiring a canvas, not DSP engineers.
# Every field has to be answerable from something they can hear; anything that
# can only be answered with a measurement is reported by `info()` instead, and
# the one field that genuinely needs a number (`mute_tail`) defaults to `auto`
# so that it follows that measurement until somebody overrides it.
#
# Deliberately *not* offered: frame size, filter length, adaptation step size,
# double-talk thresholds. Those either derive from the fields below or stay
# internal constants. A card with sixteen knobs is a card nobody dares touch.
TOOLS = [
    {
        "name": "audio_frontend",
        "type": "processor",
        "multiInstance": True,
        "description": (
            "语音前端处理 — 把机器人自己说话的声音从麦克风里拿掉，否则 ASR 会把"
            "机器人自己说的话当成别人说的。接在麦克风和 ASR 之间"
            "（Audio frontend: keeps the robot's own voice out of its microphone）"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "info", "config"],
                    "description": "Action to perform",
                },
                "input_topic": {
                    "type": "string",
                    "description": "Microphone AudioChunk topic (required for start)",
                },
                "instance_id": {
                    "type": "string",
                    "description": "Canvas card id, for multi-instance use",
                },
            },
            "required": ["action"],
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "title": "How to handle the robot's own voice",
                    "default": DEFAULT_MODE,
                    "scope": "instance",
                    "oneOf": [
                        {"const": MODE_MUTE,
                         "title": "Mute the mic while the robot is talking"},
                        {"const": MODE_OFF,
                         "title": "Off — pass the microphone through unchanged"},
                    ],
                    "description": (
                        "Muting is complete and foolproof, but nobody can interrupt "
                        "the robot mid-sentence: while it talks, the microphone "
                        "carries nothing. 'Off' is for comparing against no "
                        "processing at all."
                    ),
                },
                "robot_voice_topic": {
                    "type": "string",
                    "title": "Where the robot's own voice comes from",
                    "default": DEFAULT_ROBOT_VOICE_TOPIC,
                    "scope": "instance",
                    # `oneOf` is filled in per request by get_tools() with the
                    # topics the TTS card is actually publishing right now, so
                    # this is a list of real topics rather than a string to
                    # type. A topic set in config.yaml is still honoured even if
                    # it is not in the list — that is the escape hatch for a TTS
                    # running in another container.
                    "oneOf": [
                        {"const": REF_AUTO,
                         "title": "Automatic — follow the TTS card on this robot"},
                    ],
                    "description": (
                        "The audio the TTS card publishes — the same sound the "
                        "speaker plays. 'Automatic' is right unless this robot's "
                        "speech never passes through Agent Core (vendor TTS), in "
                        "which case this card cannot tell when it is talking."
                    ),
                },
                "mute_tail": {
                    "type": "string",
                    "title": "Keep the mic muted after the robot stops",
                    "default": DEFAULT_MUTE_TAIL,
                    "scope": "instance",
                    "x-show-when": {"mode": [MODE_MUTE]},
                    "oneOf": [
                        {"const": MUTE_TAIL_AUTO,
                         "title": "Automatic — follow the delay this card measures"},
                        {"const": "300ms",
                         "title": "300 ms — speaker plays instantly (USB / analog)"},
                        {"const": "800ms",
                         "title": "800 ms — speaker buffers (most robots)"},
                        {"const": "1500ms",
                         "title": "1.5 s — heavy buffering or Bluetooth"},
                    ],
                    "description": (
                        "The robot is still audible for a while after its last "
                        "audio is sent, because the speaker buffers — 0.7-1.2 s on "
                        "a G1. Too short and the end of the robot's own sentence "
                        "gets transcribed; too long and the first word of the "
                        "reply is lost. The card reports the delay it measured in "
                        "`info()`, which is what 'Automatic' follows."
                    ),
                },
            },
        },
        "topic_in": [{"format": AUDIO_FORMAT, "desc": "microphone audio"}],
        "topic_out": [{"format": AUDIO_FORMAT,
                       "desc": "microphone audio, robot's own voice removed"}],
    }
]


# ── ROS2 node ────────────────────────────────────────────────────────────────

class _AudioFrontendNode(Node):
    """One microphone in, the same microphone cleaned out.

    Processing happens in the microphone callback rather than on a worker
    thread, and that is a deliberate choice for this version: the work is a
    comparison and a memset, the delay estimate is a 300x150 dot product once a
    second, and a queue plus a thread would add latency to the audio path and a
    second lifecycle to get wrong. It stops being the right choice as soon as an
    adaptive filter lands here — revisit it then, with the CPU figure this card
    reports.

    Lifecycle follows the five rules in perception/README.md § Plugin
    Concurrency: the plugin registers this node before starting it, `stop`
    signals cancellation before taking the lock, and the subscriptions outlive
    `stop()` so the executor's wait set is never torn out from under it.
    """

    def __init__(self, input_topic: str, ref_topic: str = '', *,
                 node_suffix: str = '', mode: str = DEFAULT_MODE,
                 mute_tail: str = DEFAULT_MUTE_TAIL) -> None:
        super().__init__(f"audio_frontend_{node_suffix}" if node_suffix
                         else "audio_frontend")
        self._input_topic = input_topic or ''
        self._output_topic = f"{self._input_topic}{OUTPUT_SUFFIX}" if self._input_topic else ''
        # A reference topic equal to the microphone topic would make the card
        # correlate the mic against itself and mute permanently at delay 0.
        self._ref_topic = ref_topic if ref_topic and ref_topic != self._input_topic else ''
        if ref_topic and not self._ref_topic:
            log.warning("[audio_frontend] ignoring robot_voice_topic %r: it is the "
                        "microphone topic", ref_topic)

        self._mode = mode if mode in MODES else DEFAULT_MODE
        self._mute_tail = mute_tail if mute_tail in MUTE_TAIL_CHOICES else DEFAULT_MUTE_TAIL

        self._node_lock = threading.RLock()
        self._retired = False
        self.state = "idle"

        self._gate_lock = threading.Lock()
        self._mute_until = 0.0          # mic-clock time the gate may reopen
        self._last_ref_audio_t = 0.0    # publisher stamp of the last ref chunk
        self._ref_seen = False
        self._last_eof_t = 0.0

        self._tracker = _DelayTracker()
        self._tracker_lock = threading.Lock()

        self._stats_lock = threading.Lock()
        self._stats: Dict[str, int] = {
            "mic_chunks": 0, "mic_bytes": 0, "published": 0, "muted_chunks": 0,
            "muted_bytes": 0, "ref_chunks": 0, "ref_eof": 0, "mic_gaps": 0,
            "format_violations": 0, "misaligned_chunks": 0, "publish_errors": 0,
        }
        self._last_mic_t = 0.0
        self._process_s = 0.0   # processing time, for the CPU figure
        self._audio_s = 0.0     # audio time it was spent on

        self._pub = None
        self._sub = None
        self._ref_sub = None

        log.info("[audio_frontend] node created: mic=%s -> %s, robot_voice=%s, "
                 "mode=%s, mute_tail=%s",
                 self._input_topic or "(none)", self._output_topic or "(none)",
                 self._ref_topic or "(none)", self._mode, self._mute_tail)

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> Dict[str, Any]:
        with self._node_lock:
            if self._retired:
                # A concurrent stop destroyed this node between the plugin
                # registering it and this call.
                return self.status()
            if self.state == "running":
                return self.status()
            if not self._input_topic:
                return {"state": "error",
                        "message": "audio_frontend needs a microphone topic "
                                   "(connect one on the canvas)"}
            from audio_msgs.msg import AudioChunk
            if self._pub is None:
                self._pub = self.create_publisher(AudioChunk, self._output_topic, _AUDIO_QOS)
            if self._sub is None:
                self._sub = self.create_subscription(
                    AudioChunk, self._input_topic, self._mic_cb, _AUDIO_QOS)
            if self._ref_topic and self._ref_sub is None:
                self._ref_sub = self.create_subscription(
                    AudioChunk, self._ref_topic, self._ref_cb, _AUDIO_QOS)
            self.state = "running"
            log.info("[audio_frontend] started: %s -> %s (mode=%s)",
                     self._input_topic, self._output_topic, self._mode)
            return self.status()

    def request_retire(self) -> None:
        """Non-blocking cancellation, callable before any lock is taken."""
        self._retired = True

    def retire(self) -> Dict[str, Any]:
        self.request_retire()
        return self.stop()

    def stop(self) -> Dict[str, Any]:
        # The subscriptions are left in place for dispose_node() to release with
        # the node: destroying them here can race the executor's wait set and
        # raise InvalidHandle, which has taken the whole ROS layer down before
        # (perception/README.md).
        with self._node_lock:
            self.state = "idle"
        with self._gate_lock:
            self._mute_until = 0.0
        log.info("[audio_frontend] stopped: %s", self._input_topic)
        return {"state": "idle"}

    def apply_config(self, mode: Optional[str] = None,
                     mute_tail: Optional[str] = None) -> None:
        """Change the live knobs. Neither needs the node rebuilt.

        `robot_voice_topic` is not here on purpose — a subscription cannot be
        recreated on a node the executor is spinning (it wedges: the callback
        never fires again while the graph and the executor both look healthy),
        so the plugin disposes the node and builds a new one instead.
        """
        with self._node_lock:
            if mode in MODES:
                self._mode = mode
            if mute_tail in MUTE_TAIL_CHOICES:
                self._mute_tail = mute_tail

    # ── the gate ─────────────────────────────────────────────────────────

    def _tail_s(self) -> float:
        explicit = MUTE_TAIL_CHOICES.get(self._mute_tail)
        if explicit is not None:
            return explicit
        with self._tracker_lock:
            measured = self._tracker.measured
        if measured is None:
            return AUTO_TAIL_FALLBACK_S
        return measured + AUTO_TAIL_MARGIN_S

    def _ref_cb(self, message: Any) -> None:
        if self.state != "running" or self._retired:
            return
        try:
            pcm = bytes(message.data)
        except (TypeError, ValueError):
            return
        now = time.time()
        timestamp = _chunk_timestamp(message)

        if pcm == AUDIO_EOF_MAGIC:
            # End of the utterance's *publishing*, not of the sound: the speaker
            # path still holds several hundred ms. Recorded for info(), and
            # deliberately not used to open the gate.
            with self._gate_lock:
                self._last_eof_t = timestamp
            with self._stats_lock:
                self._stats["ref_eof"] += 1
            return

        if not pcm or len(pcm) % BYTES_PER_SAMPLE:
            return

        with self._stats_lock:
            self._stats["ref_chunks"] += 1

        duration = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        envelope = _envelope(pcm)
        talking = bool(envelope.size) and float(envelope.max()) > SILENCE_RMS

        with self._tracker_lock:
            self._tracker.add_ref(pcm, timestamp, now)

        with self._gate_lock:
            self._ref_seen = True
            if talking:
                self._last_ref_audio_t = timestamp
                # Extend rather than overwrite: reference chunks arrive in a
                # burst of five at the start of an utterance (TTS pre-buffer),
                # so an overwrite would shorten the window the burst just set.
                self._mute_until = max(self._mute_until,
                                       timestamp + duration + self._tail_s())

    # ── the audio path ───────────────────────────────────────────────────

    def _mic_cb(self, message: Any) -> None:
        if self.state != "running" or self._retired:
            return
        started = time.perf_counter()

        fmt = getattr(message, 'format', '') or ''
        if fmt not in _AUDIO_FORMATS:
            with self._stats_lock:
                self._stats["format_violations"] += 1
        try:
            pcm = bytes(message.data)
        except (TypeError, ValueError):
            return
        if not pcm:
            return
        if len(pcm) % BYTES_PER_SAMPLE:
            # Truncate rather than drop, as ASR does: an odd byte count breaks
            # int16 framing, and dropping would change the behaviour of drivers
            # that work today.
            pcm = pcm[:len(pcm) // BYTES_PER_SAMPLE * BYTES_PER_SAMPLE]
            with self._stats_lock:
                self._stats["misaligned_chunks"] += 1
            if not pcm:
                return

        now = time.time()
        timestamp = _chunk_timestamp(message)
        duration = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)

        # A gap in the microphone stream is not the same thing as a change in
        # delay, and conflating them is how a correlator ends up chasing a
        # phantom: count it so a busy bus is visible in info().
        with self._stats_lock:
            self._stats["mic_chunks"] += 1
            self._stats["mic_bytes"] += len(pcm)
            if self._last_mic_t and timestamp - self._last_mic_t > 1.5 * duration:
                self._stats["mic_gaps"] += 1
            self._last_mic_t = timestamp

        # Fed from the raw microphone, before any muting — measuring the delay
        # is exactly what this card can do for an operator *while* running in
        # the mode that does not need it yet.
        with self._tracker_lock:
            self._tracker.add_mic(pcm, timestamp, now)
            if self._tracker.due(now):
                self._tracker.estimate(now)

        muted = False
        if self._mode == MODE_MUTE:
            with self._gate_lock:
                muted = timestamp < self._mute_until
            if muted:
                # Silence, not a dropped chunk. Downstream VAD treats a gap and
                # silence the same way, but a continuous stream keeps ASR's
                # "have I ever received audio" check and its pre-roll sane, and
                # keeps this card's own gap counter meaningful.
                pcm = b'\x00' * len(pcm)

        self._publish(pcm, message, muted)

        with self._stats_lock:
            self._process_s += time.perf_counter() - started
            self._audio_s += duration

    def _publish(self, pcm: bytes, source: Any, muted: bool) -> None:
        publisher = self._pub
        if publisher is None:
            return
        from audio_msgs.msg import AudioChunk
        out = AudioChunk()
        # The *input's* stamp, not now(): downstream reads it as the capture
        # time (ASR's audio_start_ts comes from it), so re-stamping would charge
        # this card's own latency to the speaker's timeline.
        out.header.stamp = source.header.stamp
        out.format = AUDIO_FORMAT
        # `array.array('B')` and not `list(pcm)`, which is what the rest of the
        # project writes. The generated `data` setter validates a list
        # element-by-element — `all(isinstance(v, int) and 0 <= v < 256 ...)`,
        # 1024 iterations per chunk — and takes a buffer fast path for an array
        # of matching typecode. Measured on Orin 5: 0.226 ms vs 0.021 ms per
        # chunk, which was the single largest cost in this callback. The same
        # saving is available to plugins/tts.py and the mic drivers.
        out.data = array.array('B', pcm)
        try:
            publisher.publish(out)
        except Exception as error:  # noqa: BLE001 - a dead publisher must not kill the callback
            with self._stats_lock:
                self._stats["publish_errors"] += 1
            log.warning("[audio_frontend] publish failed on %s: %s",
                        self._output_topic, error)
            return
        with self._stats_lock:
            self._stats["published"] += 1
            if muted:
                self._stats["muted_chunks"] += 1
                self._stats["muted_bytes"] += len(pcm)

    # ── reporting ────────────────────────────────────────────────────────

    def topics(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        if not self._input_topic:
            return [], []
        return ([{"topic": self._input_topic, "format": AUDIO_FORMAT,
                  "desc": "microphone"}],
                [{"topic": self._output_topic, "format": AUDIO_FORMAT,
                  "desc": "microphone, robot's own voice removed"}])

    def diagnostics(self) -> Dict[str, Any]:
        """The numbers an operator needs to tell "working" from "wired wrong"."""
        now = time.time()
        with self._stats_lock:
            stats = dict(self._stats)
            process_s, audio_s = self._process_s, self._audio_s
        with self._gate_lock:
            ref_seen = self._ref_seen
            last_ref = self._last_ref_audio_t
            muting = now < self._mute_until
        with self._tracker_lock:
            report = self._tracker.report()
            measured = self._tracker.measured

        published = stats["published"] or 1
        tail = self._tail_s()
        result: Dict[str, Any] = {
            "mode": self._mode,
            "robot_voice_topic": self._ref_topic or None,
            "robot_voice_seen": ref_seen,
            "last_robot_voice_s_ago": (None if not last_ref
                                       else round(now - last_ref, 2)),
            "muting_now": muting,
            "mute_tail": self._mute_tail,
            "mute_tail_effective_ms": round(tail * 1000),
            "muted_fraction": round(stats["muted_chunks"] / published, 4),
            "mic_chunks": stats["mic_chunks"],
            "published_chunks": stats["published"],
            "mic_gaps": stats["mic_gaps"],
            "format_violations": stats["format_violations"],
            "misaligned_chunks": stats["misaligned_chunks"],
            "publish_errors": stats["publish_errors"],
            "ref_chunks": stats["ref_chunks"],
            "ref_utterances_ended": stats["ref_eof"],
            # Wall time spent inside the microphone callback per second of
            # audio, in ms. **Not** CPU time, and the difference is not small:
            # nine other plugins' threads contend for the GIL in this process,
            # so this inflates when the box is busy. Measured on Orin 5 with a
            # reference playing: 29.7 ms/s live against 3.6 ms/s for the same
            # callback profiled in isolation. Read it as a headroom indicator
            # and a regression tripwire, not as a CPU share — and profile in
            # isolation before concluding anything about cost.
            "callback_wall_ms_per_audio_s": (None if audio_s <= 0
                                             else round(process_s / audio_s * 1000, 2)),
            **report,
        }
        # The one comparison nobody can do in their head, spelled out: an
        # explicit tail shorter than the delay the card measured means the end
        # of the robot's own sentence reaches ASR.
        if measured is not None and MUTE_TAIL_CHOICES.get(self._mute_tail) is not None:
            result["mute_tail_shorter_than_measured_delay"] = tail < measured
        return result

    def status(self) -> Dict[str, Any]:
        topic_in, topic_out = self.topics()
        return {
            "state": self.state,
            "topic_in": topic_in,
            "topic_out": topic_out,
            "diagnostics": self.diagnostics(),
        }


# ── Plugin ───────────────────────────────────────────────────────────────────

_DEFAULT_INSTANCE = "_default"


class AudioFrontendPlugin:
    """Perception facade for the audio frontend card.

    Holds no model and loads nothing, so unlike the vision cards it has no
    `loading` state and the constructor cannot fail on a missing file. What it
    does hold is the resolution of the reference topic, which is the only part
    of this card an operator can get wrong.

    `tts` is the TTSPlugin living in this same process, handed in by main.py the
    way ASR is handed the speaker-recognition plugin. It is optional: without it
    `auto` resolves to nothing and the card says so.
    """

    PREFIX = "audio_frontend"
    _DESC = "Keeps the robot's own voice out of its microphone"

    def __init__(self, plugin_cfg: Mapping[str, Any], executor: Any,
                 tts: Any = None) -> None:
        self._plugin_cfg: Dict[str, Any] = dict(plugin_cfg or {})
        self._executor = executor
        self._tts = tts
        self._nodes: Dict[str, _AudioFrontendNode] = {}
        self._instance_configs: Dict[str, Dict[str, Any]] = {}
        self._state_lock = threading.RLock()
        log.info("[audio_frontend] plugin init: mode=%s, robot_voice_topic=%s, "
                 "mute_tail=%s, tts_handle=%s",
                 self._plugin_cfg.get("mode", DEFAULT_MODE),
                 self._plugin_cfg.get("robot_voice_topic", DEFAULT_ROBOT_VOICE_TOPIC),
                 self._plugin_cfg.get("mute_tail", DEFAULT_MUTE_TAIL),
                 "yes" if tts is not None else "no")

    # ── the reference topic ──────────────────────────────────────────────

    def _tts_audio_topics(self) -> List[str]:
        """Audio topics the TTS card is publishing, asked of the plugin itself.

        Goes through the same `info` action the dashboard uses rather than
        reaching into TTSPlugin's internals: that plugin is a facade over a
        swappable engine, and its node dict lives inside whichever engine is
        currently loaded. Never raises — a TTS in a bad state must not stop this
        card from answering `tools/list`.
        """
        if self._tts is None:
            return []
        try:
            info = self._tts.dispatch("tts", {"action": "info"}) or {}
            topics = [str(entry.get("topic") or "")
                      for entry in (info.get("topic_out") or [])
                      if (entry.get("format") or "") in _AUDIO_FORMATS]
        except Exception as error:  # noqa: BLE001
            log.debug("[audio_frontend] could not ask TTS for its topics: %s", error)
            return []
        seen: List[str] = []
        for topic in topics:
            if topic and topic not in seen:
                seen.append(topic)
        return seen

    def _resolve_ref(self, cfg: Mapping[str, Any]) -> Tuple[str, str]:
        """(topic, how) for the configured reference setting.

        `how` is reported verbatim in `info()`, because "automatic found the TTS
        card" and "automatic found nothing" are the same empty string downstream
        and only the second one is a misconfiguration.
        """
        configured = str(cfg.get("robot_voice_topic") or DEFAULT_ROBOT_VOICE_TOPIC).strip()
        if configured and configured != REF_AUTO:
            return configured, "configured"
        topics = self._tts_audio_topics()
        if not topics:
            return "", "auto:none"
        if len(topics) > 1:
            # Several TTS cards: picking one silently would be a coin flip in a
            # place nobody would look. Take the first and say which.
            log.info("[audio_frontend] auto found %d TTS audio topics, using %s",
                     len(topics), topics[0])
        return topics[0], "auto"

    def get_tools(self) -> List[Dict[str, Any]]:
        """TOOLS with the reference-topic dropdown filled from live TTS topics.

        Rebuilt per call because `main.py get_all_tools` does not cache, so the
        dashboard sees the topics that exist at the moment it asks. A deep copy:
        mutating the module-level TOOLS would accumulate stale topics for the
        life of the process.
        """
        tools = copy.deepcopy(TOOLS)
        options = tools[0]["configSchema"]["properties"]["robot_voice_topic"]["oneOf"]
        configured = str(self._plugin_cfg.get("robot_voice_topic") or "").strip()
        known = {REF_AUTO}
        for topic in self._tts_audio_topics():
            if topic not in known:
                options.append({"const": topic, "title": topic})
                known.add(topic)
        if configured and configured not in known:
            # A topic set in config.yaml or by an earlier `config` call stays
            # selectable even when its publisher is currently down; otherwise
            # the form would silently reset it to `auto`.
            options.append({"const": configured, "title": f"{configured} (configured)"})
        return tools

    # ── MCP dispatch ─────────────────────────────────────────────────────

    def dispatch(self, name: str, args: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        action = str(args.get("action", name))
        instance_id = str(args.get("instance_id", "") or "")
        input_topic = str(args.get("input_topic", "") or "")

        if action == "info":
            return self._do_info(instance_id, input_topic)
        if action == "start":
            return self._do_start(instance_id, input_topic)
        if action == "stop":
            return self._do_stop(instance_id)
        if action == "config":
            return self._do_config(instance_id, args)
        return None

    # ── config ───────────────────────────────────────────────────────────

    def _merged_cfg(self, node_key: str = "") -> Dict[str, Any]:
        with self._state_lock:
            return {**self._plugin_cfg, **self._instance_configs.get(node_key, {})}

    def _do_config(self, instance_id: str, args: Mapping[str, Any]) -> Dict[str, Any]:
        cfg = {k: v for k, v in args.items()
               if k not in ("action", "instance_id", "input_topic")
               and v is not None and v != ""}

        with self._state_lock:
            if instance_id:
                previous = self._instance_configs.get(instance_id, {})
                self._instance_configs[instance_id] = {**previous, **cfg}
                keys = [instance_id]
            else:
                self._plugin_cfg = {**self._plugin_cfg, **cfg}
                keys = list(self._nodes)
            targets = [(key, self._nodes.get(key)) for key in keys]

        rebuild: List[Tuple[str, str]] = []
        for key, node in targets:
            if node is None:
                continue
            merged = self._merged_cfg(key)
            node.apply_config(mode=merged.get("mode"),
                              mute_tail=merged.get("mute_tail"))
            wanted, _ = self._resolve_ref(merged)
            if wanted != node._ref_topic:
                # A subscription cannot be recreated on a spinning node — it
                # wedges silently (perception/README.md § Plugin Concurrency),
                # so the node is replaced rather than re-subscribed.
                rebuild.append((key, node._input_topic))

        for key, input_topic in rebuild:
            log.info("[audio_frontend] robot_voice_topic changed, rebuilding %s", key)
            self._do_stop(key)
            self._do_start(key, input_topic)

        result = {"status": "configured"}
        if instance_id:
            result["instance_id"] = instance_id
        return result

    # ── start / stop ─────────────────────────────────────────────────────

    def _create_node(self, node_key: str, input_topic: str) -> _AudioFrontendNode:
        cfg = self._merged_cfg(node_key)
        ref_topic, _ = self._resolve_ref(cfg)
        return _AudioFrontendNode(
            input_topic,
            ref_topic,
            node_suffix=node_key.replace('/', '_').replace('-', '_').strip('_'),
            mode=str(cfg.get("mode") or DEFAULT_MODE),
            mute_tail=str(cfg.get("mute_tail") or DEFAULT_MUTE_TAIL),
        )

    def _dispose(self, node_key: str, node: _AudioFrontendNode) -> None:
        try:
            node.retire()
        finally:
            dispose_node(self._executor, node, label=f"audio_frontend/{node_key}")
        log.info("[audio_frontend] node disposed: %s", node_key)

    def _do_start(self, instance_id: str, input_topic: str) -> Dict[str, Any]:
        node_key = instance_id or input_topic or _DEFAULT_INSTANCE

        retired = None
        with self._state_lock:
            existing = self._nodes.get(node_key)
            if existing is not None and existing._input_topic != input_topic:
                # Topics are fixed at construction; rebind the key.
                retired = self._nodes.pop(node_key)
        if retired is not None:
            self._dispose(node_key, retired)

        with self._state_lock:
            existing = self._nodes.get(node_key)
        if existing is not None:
            return existing.start()        # idempotent re-start

        node = self._create_node(node_key, input_topic)
        # Registered under the lock *before* being started, so a concurrent stop
        # can always find it. A started-but-unregistered node keeps a live
        # subscription and a publisher nothing can reach for the life of the
        # process.
        with self._state_lock:
            current = self._nodes.get(node_key)
            registered = False
            if current is None:
                try:
                    self._executor.add_node(node)
                except Exception as error:  # noqa: BLE001
                    log.error("[audio_frontend] failed to register instance %r: %s",
                              node_key, error)
                else:
                    self._nodes[node_key] = node
                    registered = True
        if not registered:
            try:
                node.destroy_node()   # never started, never registered
            except Exception:  # noqa: BLE001
                pass
            if current is not None:
                return current.start()
            return {"state": "idle", "input": input_topic}
        try:
            result = node.start()
        except Exception:
            with self._state_lock:
                if self._nodes.get(node_key) is node:
                    del self._nodes[node_key]
            self._dispose(node_key, node)
            raise
        with self._state_lock:
            still_ours = self._nodes.get(node_key) is node
        if not still_ours:
            # A concurrent stop retired it while start() ran; its dispose did
            # the teardown — make sure this start left nothing running.
            node.stop()
            return {"state": "idle", "input": input_topic}
        return result

    def _do_stop(self, instance_id: str) -> Dict[str, Any]:
        to_dispose: List[Tuple[str, _AudioFrontendNode]] = []
        with self._state_lock:
            if instance_id:
                node = self._nodes.pop(instance_id, None)
                if node is not None:
                    to_dispose.append((instance_id, node))
            else:
                to_dispose.extend(self._nodes.items())
                self._nodes = {}
        for node_key, node in to_dispose:
            self._dispose(node_key, node)
        return {"state": "idle"}

    # ── info ─────────────────────────────────────────────────────────────

    def _desc(self, node: Optional[_AudioFrontendNode], how: str) -> str:
        """The card's one-line face. Carries the misconfiguration, when there is one.

        `auto` resolving to nothing is the failure this card can suffer silently:
        it keeps publishing, the audio is simply untouched. Saying so here puts
        it on the card in the dashboard rather than only in a diagnostics field
        nobody opens.
        """
        if node is None:
            return self._DESC
        if node._mode == MODE_OFF:
            return "Passing the microphone through unchanged (mode: off)"
        if not node._ref_topic:
            if how == "auto:none":
                return ("No TTS audio to listen for — this robot's speech does not "
                        "pass through Agent Core, so the mic is passed through "
                        "unchanged")
            return "No robot voice topic set — microphone passed through unchanged"
        diagnostics = node.diagnostics()
        if not diagnostics.get("robot_voice_seen"):
            return (f"Waiting for audio on {node._ref_topic} — nothing heard yet, "
                    f"so nothing is being muted")
        if diagnostics.get("mute_tail_shorter_than_measured_delay"):
            return (f"Mute tail {diagnostics['mute_tail_effective_ms']} ms is shorter "
                    f"than the {diagnostics['measured_echo_delay_ms']} ms delay "
                    f"measured — the end of the robot's own speech will get through")
        return self._DESC

    def _do_info(self, instance_id: str, input_topic: str) -> Dict[str, Any]:
        base = {"name": "AudioFrontend", "manufacture": "Embodied",
                "model": "echo-gate"}
        with self._state_lock:
            node = self._nodes.get(instance_id) if instance_id else None
            keys = list(self._nodes)
            nodes = [self._nodes[key] for key in keys]

        if instance_id:
            if node is not None:
                _, how = self._resolve_ref(self._merged_cfg(instance_id))
                return {**base, **node.status(), "desc": self._desc(node, how)}
            topic_in = ([{"topic": input_topic, "format": AUDIO_FORMAT, "desc": ""}]
                        if input_topic else [])
            topic_out = ([{"topic": f"{input_topic}{OUTPUT_SUFFIX}",
                           "format": AUDIO_FORMAT, "desc": ""}]
                         if input_topic else [])
            return {**base, "state": "idle", "desc": self._DESC,
                    "topic_in": topic_in, "topic_out": topic_out}

        topics_in: List[Dict[str, str]] = []
        topics_out: List[Dict[str, str]] = []
        for item in nodes:
            item_in, item_out = item.topics()
            topics_in.extend(item_in)
            topics_out.extend(item_out)
        if not nodes and input_topic:
            topics_in = [{"topic": input_topic, "format": AUDIO_FORMAT, "desc": ""}]
            topics_out = [{"topic": f"{input_topic}{OUTPUT_SUFFIX}",
                           "format": AUDIO_FORMAT, "desc": ""}]

        state = "running" if any(item.state == "running" for item in nodes) else "idle"
        if any(item.state == "error" for item in nodes):
            state = "error"
        result: Dict[str, Any] = {
            **base,
            "state": state,
            "desc": self._DESC,
            "topic_in": topics_in,
            "topic_out": topics_out,
        }
        if nodes:
            result["instances"] = {key: {"state": self._nodes[key].state}
                                   for key in keys if key in self._nodes}
            first = nodes[0]
            _, how = self._resolve_ref(self._merged_cfg(keys[0]))
            result["desc"] = self._desc(first, how)
            result["diagnostics"] = first.diagnostics()
        return result


__all__ = ["AudioFrontendPlugin", "TOOLS", "AUDIO_FORMAT", "OUTPUT_SUFFIX"]

"""Host-side tests for the audio frontend card — no ROS, no audio hardware.

Two halves. The first drives `_DelayTracker` with synthetic signals, which is
the only way the thresholds in it can be defended: every one of them is a
number chosen to separate "there is an echo and it arrives 420 ms late" from "a
flat correlation profile still has a maximum". The second drives the node and
the plugin facade through the lifecycle the canvas actually produces
(config→start→stop→config→start within seconds, from several threads).

What these tests deliberately cannot cover is the acoustics: whether a given
robot's speaker is linear enough for cancellation, and what its real delay is.
That needs a machine with a speaker and a microphone in the same room.
"""

from __future__ import annotations

import threading
import types

import numpy as np
import pytest

from vision_stubs import _FakeAudioChunk, _FakeExecutor, _FakeNode

import plugins.audio_frontend as af


BASE_T = 1_700_000_000.0
MIC_CHUNK_SAMPLES = 512        # 32 ms, the documented minimum
REF_CHUNK_SAMPLES = 1600       # 100 ms, what plugins/tts.py publishes


# ── helpers ──────────────────────────────────────────────────────────────────

def _chunk(pcm: bytes, timestamp: float, fmt: str = af.AUDIO_FORMAT) -> _FakeAudioChunk:
    message = _FakeAudioChunk()
    seconds = int(timestamp)
    message.header.stamp = types.SimpleNamespace(
        sec=seconds, nanosec=round((timestamp - seconds) * 1_000_000_000))
    message.format = fmt
    message.data = pcm
    return message


def _bursts(seconds: float, seed: int = 0, amplitude: int = 6000) -> np.ndarray:
    """Speech-like audio: loud stretches separated by pauses.

    The envelope has to actually vary — correlating two constant-level noises
    measures nothing, and a test built on them would pass with the lag search
    broken.
    """
    rng = np.random.default_rng(seed)
    total = int(seconds * af.SAMPLE_RATE)
    wave = rng.normal(0, amplitude, total)
    gate = np.zeros(total)
    position = 0
    while position < total:
        on = int(rng.uniform(0.15, 0.35) * af.SAMPLE_RATE)
        off = int(rng.uniform(0.10, 0.25) * af.SAMPLE_RATE)
        gate[position:position + on] = rng.uniform(0.4, 1.0)
        position += on + off
    return (wave * gate).astype(np.int16)


def _delayed(wave: np.ndarray, delay_s: float, gain: float = 0.5,
             noise: int = 40, seed: int = 7) -> np.ndarray:
    """`wave` as it would arrive at a microphone `delay_s` later, attenuated."""
    rng = np.random.default_rng(seed)
    shift = int(round(delay_s * af.SAMPLE_RATE))
    echo = np.zeros_like(wave, dtype=np.float64)
    if shift < wave.size:
        echo[shift:] = wave[:wave.size - shift] * gain
    return (echo + rng.normal(0, noise, wave.size)).astype(np.int16)


def _feed(tracker: af._DelayTracker, mic: np.ndarray, ref: np.ndarray,
          now: float) -> None:
    for start in range(0, ref.size - REF_CHUNK_SAMPLES + 1, REF_CHUNK_SAMPLES):
        timestamp = BASE_T + start / af.SAMPLE_RATE
        tracker.add_ref(ref[start:start + REF_CHUNK_SAMPLES].tobytes(), timestamp, now)
    for start in range(0, mic.size - MIC_CHUNK_SAMPLES + 1, MIC_CHUNK_SAMPLES):
        timestamp = BASE_T + start / af.SAMPLE_RATE
        tracker.add_mic(mic[start:start + MIC_CHUNK_SAMPLES].tobytes(), timestamp, now)


def _published(node) -> list[bytes]:
    return [bytes(bytearray(message.data)) for message in node._pub.raw_messages]


def _silent(pcm: bytes) -> bool:
    return not any(pcm)


# ── _DelayTracker ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("delay_s", [0.12, 0.42, 0.95])
def test_tracker_measures_a_known_delay(delay_s):
    ref = _bursts(6.0, seed=1)
    mic = _delayed(ref, delay_s)
    tracker = af._DelayTracker()
    now = BASE_T + 6.0
    _feed(tracker, mic, ref, now)

    measured = tracker.estimate(now)
    assert measured is not None, "a 6 s window with a clean echo must produce an estimate"
    # One envelope hop is 10 ms, so two hops of error is the resolution floor.
    assert abs(measured - delay_s) <= 0.03


def test_tracker_says_nothing_when_the_robot_is_silent():
    ref = np.zeros(int(6.0 * af.SAMPLE_RATE), dtype=np.int16)
    mic = _bursts(6.0, seed=2)        # a person talking, robot quiet
    tracker = af._DelayTracker()
    now = BASE_T + 6.0
    _feed(tracker, mic, ref, now)

    assert tracker.estimate(now) is None
    assert tracker.measured is None


def test_tracker_refuses_an_uncorrelated_microphone():
    """No acoustic path at all must report nothing — not a confident 0 ms.

    This is the failure that matters: a flat correlation profile still has an
    argmax, and reporting its position is how a card ends up stating a delay
    nobody can reproduce.
    """
    ref = _bursts(6.0, seed=3)
    mic = _bursts(6.0, seed=99)       # independent audio, same statistics
    tracker = af._DelayTracker()
    now = BASE_T + 6.0
    _feed(tracker, mic, ref, now)

    assert tracker.estimate(now) is None
    assert tracker.report()["echo_delay_rejected"] >= 1


def test_tracker_reports_a_median_and_a_spread():
    tracker = af._DelayTracker()
    for value in (0.40, 0.42, 0.44, 0.60):
        tracker._estimates.append(value)
    assert tracker.measured == pytest.approx(0.43)
    assert tracker.report()["measured_echo_delay_ms"] == 430
    assert tracker.report()["echo_delay_spread_ms"] is not None


def test_a_quiet_echo_is_still_measurable():
    """A speaker at low volume still has to be found: 20x attenuation, measured."""
    ref = _bursts(6.0, seed=5)
    mic = _delayed(ref, 0.42, gain=0.05)
    tracker = af._DelayTracker()
    now = BASE_T + 6.0
    _feed(tracker, mic, ref, now)
    assert tracker.estimate(now) == pytest.approx(0.42, abs=0.03)


def test_a_delay_beyond_the_search_range_is_mostly_refused():
    """An echo later than the search range must not be reported as a short one.

    Measured with a 2 s echo against the 1.5 s range: most windows return
    nothing, and the ones that do not used to land on the outermost lag — which
    is now refused outright, because a peak pinned to the end of the range is
    the signature of a delay beyond it rather than one exactly at it. A few
    windows still find a spurious correlation between two burst patterns; that
    is what the median over many windows is for, and it is the reason this
    asserts a rate rather than a flat "never".
    """
    accepted = []
    for seed in range(10):
        ref = _bursts(6.0, seed=seed)
        mic = _delayed(ref, 2.0, seed=seed + 7)
        tracker = af._DelayTracker()
        _feed(tracker, mic, ref, BASE_T + 6.0)
        estimate = tracker.estimate(BASE_T + 6.0)
        if estimate is not None:
            accepted.append(estimate)

    assert len(accepted) <= 3, f"too many spurious measurements: {accepted}"
    limit = af.DELAY_SEARCH_MAX_S - af.ENVELOPE_HOP_S
    assert all(value < limit for value in accepted), \
        "a peak at the end of the search range must be refused"


def test_double_talk_fools_a_single_window_but_not_the_median():
    """The measured weakness of this estimator, and the defence that covers it.

    A person talking over the robot at the robot's own level can push a single
    window's peak anywhere — 1.43 s and 0 ms both turned up in a sweep. That is
    why `measured` is a median over many windows rather than the latest value,
    and why `echo_delay_spread_ms` is reported: the median held at 420 ms with
    80% of windows contaminated, while the spread grew to ~1 s and says so.
    """
    truth = 0.42
    tracker = af._DelayTracker()
    rng = np.random.default_rng(3)
    for window in range(20):
        ref = _bursts(6.0, seed=window)
        mic = _delayed(ref, truth, seed=window + 7)
        if rng.random() < 0.5:
            talker = _bursts(6.0, seed=window + 500, amplitude=8000)
            mic = np.clip(mic.astype(np.int32) + talker.astype(np.int32),
                          -32768, 32767).astype(np.int16)
        window_tracker = af._DelayTracker()
        _feed(window_tracker, mic, ref, BASE_T + 6.0)
        estimate = window_tracker.estimate(BASE_T + 6.0)
        if estimate is not None:
            tracker._estimates.append(estimate)

    assert len(tracker._estimates) >= 10
    assert tracker.measured == pytest.approx(truth, abs=0.08)
    assert tracker.spread is not None


def test_tracker_needs_both_streams_covering_the_window():
    """A reference that only started a moment ago cannot be correlated yet."""
    ref = _bursts(1.0, seed=4)
    mic = _delayed(_bursts(6.0, seed=4), 0.4)
    tracker = af._DelayTracker()
    now = BASE_T + 6.0
    _feed(tracker, mic, ref, now)
    assert tracker.estimate(now) is None


# ── the gate ─────────────────────────────────────────────────────────────────

@pytest.fixture
def node():
    instance = af._AudioFrontendNode('/mic/audio', '/tts/audio',
                                     node_suffix='t', mute_tail='300ms')
    instance.start()
    yield instance
    instance.retire()


def _say(node, seconds: float, start_t: float) -> None:
    """Publish `seconds` of robot speech on the reference topic."""
    wave = _bursts(seconds, seed=11)
    for start in range(0, wave.size - REF_CHUNK_SAMPLES + 1, REF_CHUNK_SAMPLES):
        node._ref_cb(_chunk(wave[start:start + REF_CHUNK_SAMPLES].tobytes(),
                            start_t + start / af.SAMPLE_RATE))


def _hear(node, pcm: bytes, timestamp: float) -> None:
    node._mic_cb(_chunk(pcm, timestamp))


def test_the_microphone_is_silenced_while_the_robot_talks(node):
    speech = _bursts(0.032, seed=12).tobytes()
    _hear(node, speech, BASE_T)                       # before: untouched
    _say(node, 1.0, BASE_T + 0.1)
    _hear(node, speech, BASE_T + 0.5)                 # during: silenced

    published = _published(node)
    assert len(published) == 2
    assert not _silent(published[0])
    assert _silent(published[1])
    assert len(published[1]) == len(speech), "a muted chunk keeps its length"


def test_the_mute_outlasts_the_robots_last_frame_by_the_tail(node):
    speech = _bursts(0.032, seed=13).tobytes()
    _say(node, 0.5, BASE_T)
    # The reference ends at BASE_T + 0.5; with a 300 ms tail the gate reopens at
    # 0.8 (plus the final chunk's own 100 ms).
    _hear(node, speech, BASE_T + 0.70)
    _hear(node, speech, BASE_T + 1.20)

    published = _published(node)
    assert _silent(published[0]), "still muted inside the tail"
    assert not _silent(published[1]), "mic reopens once the tail has passed"


def test_off_mode_passes_the_microphone_through_untouched():
    instance = af._AudioFrontendNode('/mic/audio', '/tts/audio',
                                     node_suffix='off', mode=af.MODE_OFF)
    instance.start()
    speech = _bursts(0.032, seed=14).tobytes()
    _say(instance, 1.0, BASE_T)
    _hear(instance, speech, BASE_T + 0.5)

    assert _published(instance) == [speech]
    assert instance.diagnostics()["muted_fraction"] == 0.0
    instance.retire()


def test_the_published_chunk_keeps_the_input_stamp(node):
    """Re-stamping would charge this card's latency to the speaker's timeline.

    ASR derives `audio_start_ts` from this header, so the stamp has to describe
    when the sound was captured, not when it was forwarded.
    """
    speech = _bursts(0.032, seed=15).tobytes()
    incoming = _chunk(speech, BASE_T + 3.25)
    node._mic_cb(incoming)
    published = node._pub.raw_messages[0]
    assert published.header.stamp.sec == incoming.header.stamp.sec
    assert published.header.stamp.nanosec == incoming.header.stamp.nanosec
    assert published.format == af.AUDIO_FORMAT


def test_the_end_of_utterance_marker_is_not_reference_audio(node):
    """TTS's 8-byte EOF magic means 'finished publishing', not 'finished playing'.

    Treating it as audio would extend the mute by its own duration and, worse,
    feed four samples of square wave to the delay tracker.
    """
    node._ref_cb(_chunk(af.AUDIO_EOF_MAGIC, BASE_T))
    diagnostics = node.diagnostics()
    assert diagnostics["ref_utterances_ended"] == 1
    assert diagnostics["ref_chunks"] == 0
    assert diagnostics["muting_now"] is False

    speech = _bursts(0.032, seed=16).tobytes()
    _hear(node, speech, BASE_T + 0.01)
    assert not _silent(_published(node)[0])


def test_a_silent_reference_chunk_does_not_arm_the_gate(node):
    """TTS pads and the speaker driver's own silence must not mute the mic."""
    node._ref_cb(_chunk(b'\x00\x00' * REF_CHUNK_SAMPLES, BASE_T))
    speech = _bursts(0.032, seed=17).tobytes()
    _hear(node, speech, BASE_T + 0.05)
    assert not _silent(_published(node)[0])
    assert node.diagnostics()["robot_voice_seen"] is True


def test_the_microphone_topic_is_refused_as_a_reference():
    """Correlating the mic against itself would mute permanently at delay 0."""
    instance = af._AudioFrontendNode('/mic/audio', '/mic/audio', node_suffix='self')
    assert instance._ref_topic == ''
    instance.start()
    assert [sub.topic for sub in instance.subscriptions] == ['/mic/audio']
    instance.retire()


def test_an_odd_length_chunk_is_truncated_not_dropped(node):
    _hear(node, b'\x01\x02\x03', BASE_T)
    published = _published(node)
    assert len(published) == 1 and len(published[0]) == 2
    assert node.diagnostics()["misaligned_chunks"] == 1


def test_a_gap_in_the_microphone_stream_is_counted(node):
    speech = _bursts(0.032, seed=18).tobytes()
    _hear(node, speech, BASE_T)
    _hear(node, speech, BASE_T + 0.032)      # on time
    _hear(node, speech, BASE_T + 0.500)      # late — bus was busy
    assert node.diagnostics()["mic_gaps"] == 1


def test_a_foreign_format_is_counted_but_still_processed(node):
    speech = _bursts(0.032, seed=19).tobytes()
    node._mic_cb(_chunk(speech, BASE_T, fmt="audio/pcm-48k"))
    assert node.diagnostics()["format_violations"] == 1
    assert _published(node) == [speech]


def test_auto_tail_follows_the_measured_delay(node):
    node.apply_config(mute_tail=af.MUTE_TAIL_AUTO)
    assert node._tail_s() == pytest.approx(af.AUTO_TAIL_FALLBACK_S)
    with node._tracker_lock:
        node._tracker._estimates.append(1.10)
    assert node._tail_s() == pytest.approx(1.10 + af.AUTO_TAIL_MARGIN_S)


def test_the_output_topic_is_the_input_plus_a_suffix(node):
    topic_in, topic_out = node.topics()
    assert topic_in[0]["topic"] == '/mic/audio'
    assert topic_out[0]["topic"] == '/mic/audio' + af.OUTPUT_SUFFIX
    assert topic_out[0]["format"] == af.AUDIO_FORMAT


def test_starting_without_a_microphone_is_an_error_not_a_silent_no_op():
    instance = af._AudioFrontendNode('', '/tts/audio', node_suffix='bare')
    result = instance.start()
    assert result["state"] == "error"
    assert "microphone" in result["message"]


# ── plugin facade ────────────────────────────────────────────────────────────

class _FakeTTS:
    """Enough of TTSPlugin for the reference lookup: the `info` action."""

    def __init__(self, topics=('/tts/audio',)):
        self.topics = list(topics)
        self.calls = 0

    def dispatch(self, name, args):
        self.calls += 1
        return {"topic_out": [{"topic": topic, "format": af.AUDIO_FORMAT,
                               "desc": ""} for topic in self.topics]}


class _BrokenTTS:
    def dispatch(self, name, args):
        raise RuntimeError("engine is still loading")


@pytest.fixture
def plugin():
    executor = _FakeExecutor()
    instance = af.AudioFrontendPlugin({}, executor, tts=_FakeTTS())
    yield instance
    instance.dispatch("audio_frontend", {"action": "stop"})


def test_auto_resolves_the_reference_from_the_tts_plugin(plugin):
    plugin.dispatch("audio_frontend", {"action": "start", "instance_id": "c1",
                                       "input_topic": "/mic/audio"})
    node = plugin._nodes["c1"]
    assert node._ref_topic == '/tts/audio'
    assert sorted(sub.topic for sub in node.subscriptions) == ['/mic/audio', '/tts/audio']


def test_auto_with_no_tts_in_this_process_says_so_on_the_card():
    plugin = af.AudioFrontendPlugin({}, _FakeExecutor(), tts=None)
    plugin.dispatch("audio_frontend", {"action": "start", "instance_id": "c1",
                                       "input_topic": "/mic/audio"})
    info = plugin.dispatch("audio_frontend", {"action": "info", "instance_id": "c1"})
    assert info["diagnostics"]["robot_voice_topic"] is None
    assert "does not pass through Agent Core" in info["desc"]
    # And it is a passthrough rather than a gate that mutes at random.
    node = plugin._nodes["c1"]
    _hear(node, _bursts(0.032, seed=20).tobytes(), BASE_T)
    assert not _silent(_published(node)[0])


def test_a_broken_tts_does_not_stop_this_card_answering(plugin):
    plugin._tts = _BrokenTTS()
    tools = plugin.get_tools()
    options = tools[0]["configSchema"]["properties"]["robot_voice_topic"]["oneOf"]
    assert [option["const"] for option in options] == [af.REF_AUTO]
    assert plugin.dispatch("audio_frontend", {"action": "info"})["state"] == "idle"


def test_the_dropdown_lists_the_topics_tts_is_publishing_now(plugin):
    plugin._tts = _FakeTTS(('/g1/tts/audio', '/ext/tts/audio'))
    options = plugin.get_tools()[0]["configSchema"]["properties"][
        "robot_voice_topic"]["oneOf"]
    assert [option["const"] for option in options] == [
        af.REF_AUTO, '/g1/tts/audio', '/ext/tts/audio']


def test_a_configured_topic_stays_selectable_when_its_publisher_is_down():
    plugin = af.AudioFrontendPlugin({"robot_voice_topic": "/other/box/tts/audio"},
                                    _FakeExecutor(), tts=_FakeTTS(()))
    options = plugin.get_tools()[0]["configSchema"]["properties"][
        "robot_voice_topic"]["oneOf"]
    assert [option["const"] for option in options] == [
        af.REF_AUTO, "/other/box/tts/audio"]


def test_the_module_level_tools_are_not_mutated_by_a_lookup(plugin):
    before = len(af.TOOLS[0]["configSchema"]["properties"]["robot_voice_topic"]["oneOf"])
    plugin.get_tools()
    plugin.get_tools()
    after = len(af.TOOLS[0]["configSchema"]["properties"]["robot_voice_topic"]["oneOf"])
    assert before == after == 1


def test_changing_the_reference_topic_rebuilds_the_node(plugin):
    plugin.dispatch("audio_frontend", {"action": "start", "instance_id": "c1",
                                       "input_topic": "/mic/audio"})
    first = plugin._nodes["c1"]
    plugin.dispatch("audio_frontend", {"action": "config", "instance_id": "c1",
                                       "robot_voice_topic": "/other/tts/audio"})
    second = plugin._nodes["c1"]
    # A new node, because a subscription cannot be recreated on a spinning one:
    # it wedges, with the graph and the executor both looking healthy.
    assert second is not first
    assert first.destroyed is True
    assert second._ref_topic == "/other/tts/audio"
    assert second.state == "running"


def test_changing_the_mode_does_not_rebuild_the_node(plugin):
    plugin.dispatch("audio_frontend", {"action": "start", "instance_id": "c1",
                                       "input_topic": "/mic/audio"})
    node = plugin._nodes["c1"]
    plugin.dispatch("audio_frontend", {"action": "config", "instance_id": "c1",
                                       "mode": af.MODE_OFF, "mute_tail": "1500ms"})
    assert plugin._nodes["c1"] is node
    assert node._mode == af.MODE_OFF
    assert node._tail_s() == pytest.approx(1.5)


def test_start_is_idempotent(plugin):
    args = {"action": "start", "instance_id": "c1", "input_topic": "/mic/audio"}
    plugin.dispatch("audio_frontend", args)
    node = plugin._nodes["c1"]
    plugin.dispatch("audio_frontend", args)
    assert plugin._nodes["c1"] is node
    assert [sub.topic for sub in node.subscriptions].count('/mic/audio') == 1


def test_rewiring_the_microphone_replaces_the_node(plugin):
    plugin.dispatch("audio_frontend", {"action": "start", "instance_id": "c1",
                                       "input_topic": "/mic/audio"})
    first = plugin._nodes["c1"]
    plugin.dispatch("audio_frontend", {"action": "start", "instance_id": "c1",
                                       "input_topic": "/ext_mic/1/audio"})
    second = plugin._nodes["c1"]
    assert second is not first and first.destroyed is True
    assert second._output_topic == '/ext_mic/1/audio' + af.OUTPUT_SUFFIX


def test_stop_disposes_the_node_and_leaves_the_executor_clean(plugin):
    plugin.dispatch("audio_frontend", {"action": "start", "instance_id": "c1",
                                       "input_topic": "/mic/audio"})
    node = plugin._nodes["c1"]
    plugin.dispatch("audio_frontend", {"action": "stop", "instance_id": "c1"})
    assert plugin._nodes == {}
    assert node.destroyed is True
    assert plugin._executor.nodes == []


def test_a_concurrent_stop_cannot_orphan_a_live_node(plugin):
    """The canvas really does this: config→start→stop→start within a second.

    An orphaned node is the expensive failure here — nothing can reach it to
    stop it, so its subscription and publisher outlive every card on the page
    (perception/README.md § Plugin Concurrency).
    """
    args = {"action": "start", "instance_id": "c1", "input_topic": "/mic/audio"}
    errors = []

    def hammer(action):
        for _ in range(40):
            try:
                plugin.dispatch("audio_frontend",
                                args if action == "start"
                                else {"action": "stop", "instance_id": "c1"})
            except Exception as error:  # noqa: BLE001
                errors.append(error)

    threads = [threading.Thread(target=hammer, args=(name,))
               for name in ("start", "stop", "start", "stop")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors

    plugin.dispatch("audio_frontend", {"action": "stop"})
    assert plugin._nodes == {}
    assert plugin._executor.nodes == []
    live = [instance for instance in _FakeNode.instances
            if isinstance(instance, af._AudioFrontendNode) and not instance.destroyed]
    assert live == [], "every node built must end up destroyed"


def test_info_says_nothing_has_been_heard_yet(plugin):
    plugin.dispatch("audio_frontend", {"action": "start", "instance_id": "c1",
                                       "input_topic": "/mic/audio"})
    info = plugin.dispatch("audio_frontend", {"action": "info", "instance_id": "c1"})
    assert info["diagnostics"]["robot_voice_seen"] is False
    assert "nothing heard yet" in info["desc"]


def test_info_flags_a_tail_shorter_than_the_measured_delay(plugin):
    plugin.dispatch("audio_frontend", {"action": "config", "instance_id": "c1",
                                       "mute_tail": "300ms"})
    plugin.dispatch("audio_frontend", {"action": "start", "instance_id": "c1",
                                       "input_topic": "/mic/audio"})
    node = plugin._nodes["c1"]
    _say(node, 0.3, BASE_T)
    with node._tracker_lock:
        node._tracker._estimates.append(1.05)

    info = plugin.dispatch("audio_frontend", {"action": "info", "instance_id": "c1"})
    assert info["diagnostics"]["mute_tail_shorter_than_measured_delay"] is True
    assert "shorter than" in info["desc"]


def test_info_without_an_instance_reports_the_card(plugin):
    plugin.dispatch("audio_frontend", {"action": "start", "instance_id": "c1",
                                       "input_topic": "/mic/audio"})
    info = plugin.dispatch("audio_frontend", {"action": "info"})
    assert info["state"] == "running"
    assert info["topic_out"][0]["topic"] == '/mic/audio' + af.OUTPUT_SUFFIX
    assert info["instances"] == {"c1": {"state": "running"}}


def test_an_unknown_action_is_declined_not_answered(plugin):
    assert plugin.dispatch("audio_frontend", {"action": "speak"}) is None


def test_the_cpu_cost_is_reported(plugin):
    plugin.dispatch("audio_frontend", {"action": "start", "instance_id": "c1",
                                       "input_topic": "/mic/audio"})
    node = plugin._nodes["c1"]
    speech = _bursts(0.032, seed=21).tobytes()
    for index in range(30):
        _hear(node, speech, BASE_T + index * 0.032)
    cost = node.diagnostics()["cpu_ms_per_audio_s"]
    assert cost is not None and cost >= 0.0

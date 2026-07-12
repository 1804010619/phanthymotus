import struct
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugins.asr_runtime import (
    VadSession,
    normalize_vad_backend,
    resolve_vad_settings,
)


class ASRRuntimeTest(unittest.TestCase):
    def test_vad_backend_alias_and_validation(self):
        self.assertEqual(normalize_vad_backend("silero"), "sherpa_onnx")
        self.assertEqual(normalize_vad_backend("webrtc"), "webrtc")
        with self.assertRaises(ValueError):
            normalize_vad_backend("ignored-backend")

    def test_vad_settings_support_nested_config_and_flat_overrides(self):
        settings = resolve_vad_settings(
            {
                "vad": {
                    "model": "silero",
                    "threshold": 0.6,
                    "silence_ms": 700,
                    "pre_roll_ms": 300,
                },
                "vad_threshold": 0.55,
            }
        )

        self.assertEqual(settings["backend"], "sherpa_onnx")
        self.assertEqual(settings["threshold"], 0.55)
        self.assertEqual(settings["silence_ms"], 700)
        self.assertEqual(settings["pre_roll_ms"], 300)

    def test_energy_vad_preserves_audio_before_speech(self):
        session = VadSession(
            backend="energy",
            threshold=0.5,
            silence_ms=60,
            pre_roll_ms=60,
        )
        silence = b"\x00\x00" * 480
        speech = struct.pack("<480h", *([6000] * 480))
        chunks = [silence, silence, speech, speech, silence, silence]

        result = None
        for index, chunk in enumerate(chunks):
            result = session.process_chunk(chunk, 10.0 + index * 0.03) or result

        self.assertIsNotNone(result)
        utterance, start_ts, end_ts = result
        self.assertTrue(utterance.startswith(silence + silence))
        self.assertIn(speech + speech, utterance)
        self.assertAlmostEqual(start_ts, 10.0, places=3)
        self.assertGreater(end_ts, start_ts)

    def test_sherpa_vad_preroll_uses_audio_before_segment_start(self):
        class FakeVad:
            def __init__(self, _config, buffer_size_in_seconds):
                self.buffer_size_in_seconds = buffer_size_in_seconds
                self.segments = []
                self.total_samples = 0

            def accept_waveform(self, samples):
                self.total_samples += len(samples)
                if self.total_samples >= 960 and not self.segments:
                    self.segments.append(
                        types.SimpleNamespace(start=480, samples=[0.2] * 480)
                    )

            def empty(self):
                return not self.segments

            @property
            def front(self):
                return self.segments[0]

            def pop(self):
                self.segments.pop(0)

            def reset(self):
                self.segments.clear()

        sherpa = types.ModuleType("sherpa_onnx")
        sherpa.VadModelConfig = lambda **kwargs: kwargs
        sherpa.SileroVadModelConfig = lambda **kwargs: kwargs
        sherpa.VoiceActivityDetector = FakeVad
        silence = b"\x00\x00" * 480
        speech = struct.pack("<480h", *([6000] * 480))

        with mock.patch.dict(sys.modules, {"sherpa_onnx": sherpa}):
            with mock.patch("utils.model_downloader.ensure_model"):
                session = VadSession(
                    backend="sherpa_onnx",
                    threshold=0.5,
                    silence_ms=60,
                    pre_roll_ms=30,
                )
            self.assertIsNone(session.process_chunk(silence, 10.0))
            result = session.process_chunk(speech, 20.0)

        utterance, start_ts, end_ts = result
        self.assertTrue(utterance.startswith(silence))
        self.assertEqual(len(utterance), len(silence) * 2)
        self.assertAlmostEqual(start_ts, 10.0, places=3)
        self.assertAlmostEqual(end_ts, 20.03, places=3)

    def test_flush_keeps_partial_frame_at_end_of_speech(self):
        session = VadSession(
            backend="energy",
            threshold=0.5,
            silence_ms=400,
            pre_roll_ms=0,
        )
        speech_frame = struct.pack("<480h", *([6000] * 480))
        partial_frame = struct.pack("<160h", *([6000] * 160))

        session.process_chunk(speech_frame, 10.0)
        session.process_chunk(partial_frame, 10.03)

        self.assertEqual(session.flush(), speech_frame + partial_frame)

    def test_frame_vad_reanchors_timestamps_after_input_gap(self):
        session = VadSession(
            backend="energy",
            threshold=0.5,
            silence_ms=30,
            pre_roll_ms=0,
        )
        speech_frame = struct.pack("<480h", *([6000] * 480))
        silence_frame = b"\x00\x00" * 480

        session.process_chunk(speech_frame, 10.0)
        session.process_chunk(speech_frame, 20.0)
        result = session.process_chunk(silence_frame, 20.03)

        self.assertAlmostEqual(result[1], 10.0, places=3)
        self.assertAlmostEqual(result[2], 20.06, places=3)


if __name__ == "__main__":
    unittest.main()

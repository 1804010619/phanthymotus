import importlib
import io
import sys
import tempfile
import threading
import time
import types
import unittest
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ASROfflineSherpaCompatTest(unittest.TestCase):
    def setUp(self):
        self._old_sherpa = sys.modules.get("sherpa_onnx")

    def tearDown(self):
        if self._old_sherpa is None:
            sys.modules.pop("sherpa_onnx", None)
        else:
            sys.modules["sherpa_onnx"] = self._old_sherpa

    def test_paraformer_uses_official_constructor_with_minimal_sherpa_api(self):
        calls = []

        class OfflineRecognizer:
            @classmethod
            def from_paraformer(cls, **kwargs):
                calls.append(kwargs)
                return types.SimpleNamespace(create_stream=lambda: None)

        sherpa = types.ModuleType("sherpa_onnx")
        sherpa.OfflineRecognizer = OfflineRecognizer
        sys.modules["sherpa_onnx"] = sherpa

        asr_offline = importlib.import_module("plugins.asr_offline")

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "model.int8.onnx").write_bytes(b"")
            (model_dir / "tokens.txt").write_text("", encoding="utf-8")

            recognizer = asr_offline._create_sherpa_recognizer(
                str(model_dir),
                {
                    "tokens": "tokens.txt",
                    "modelCategory": "paraformer",
                    "numThreads": 2,
                    "provider": "cpu",
                    "debug": False,
                    "featureConfig": {"featureDim": 80},
                    "recognizerConfig": {"decodingMethod": "greedy_search"},
                },
            )

        self.assertIsNotNone(recognizer)
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0]["paraformer"],
            str((model_dir / "model.int8.onnx").resolve()),
        )
        self.assertEqual(
            calls[0]["tokens"], str((model_dir / "tokens.txt").resolve())
        )
        self.assertEqual(calls[0]["num_threads"], 2)
        self.assertEqual(calls[0]["provider"], "cpu")

    def test_cached_recognizer_serializes_concurrent_decode_calls(self):
        class Stream:
            def __init__(self):
                self.result = types.SimpleNamespace(text="ok")

            def accept_waveform(self, _sample_rate, _samples):
                pass

        class Recognizer:
            def __init__(self):
                self.active = 0
                self.max_active = 0
                self.state_lock = threading.Lock()

            def create_stream(self):
                return Stream()

            def decode_stream(self, _stream):
                with self.state_lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.05)
                with self.state_lock:
                    self.active -= 1

        asr_offline = importlib.import_module("plugins.asr_offline")
        recognizer = Recognizer()
        adapter = object.__new__(asr_offline.OfflineASRAdapter)
        adapter._recognizer = recognizer
        adapter._decode_lock = threading.Lock()

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 160)
        wav_bytes = wav_buffer.getvalue()

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(lambda _: adapter.transcribe(wav_bytes), range(2))
            )

        self.assertEqual(results, ["ok", "ok"])
        self.assertEqual(recognizer.max_active, 1)


if __name__ == "__main__":
    unittest.main()

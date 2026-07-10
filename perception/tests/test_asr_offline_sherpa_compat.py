import importlib
import sys
import tempfile
import types
import unittest
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


if __name__ == "__main__":
    unittest.main()

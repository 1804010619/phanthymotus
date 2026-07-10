import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class ASRPackagingTest(unittest.TestCase):
    def test_jetson_image_downloads_offline_model_instead_of_copying_models(self):
        dockerfile = (REPO_ROOT / "perception" / "Dockerfile.jetson").read_text(
            encoding="utf-8"
        )

        self.assertIn("asr_model_downloader.py", dockerfile)
        self.assertIn(
            "http://172.28.4.81:34567/zengzhitao/embodied-ai/official_paraformer",
            dockerfile,
        )
        self.assertNotIn("COPY perception/models", dockerfile)

    def test_default_config_targets_asr_offline_benchmark(self):
        config = (REPO_ROOT / "perception" / "config.yaml").read_text(
            encoding="utf-8"
        )

        self.assertIn("  asr:\n    enabled: true\n    mode: offline", config)
        self.assertIn("model_path: /models/sherpa-onnx/asr-offline", config)
        self.assertIn("    kws:\n      enabled: false", config)
        for plugin in ("tts", "htmsg", "vop"):
            self.assertIn(f"  {plugin}:\n    enabled: false", config)

    def test_gitignore_blocks_model_artifacts(self):
        gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")

        self.assertIn("perception/models/", gitignore)
        self.assertIn("*.onnx", gitignore)


if __name__ == "__main__":
    unittest.main()

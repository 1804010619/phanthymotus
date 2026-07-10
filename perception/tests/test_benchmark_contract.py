import unittest
from pathlib import Path


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]


class BenchmarkContractTest(unittest.TestCase):
    def test_ports_use_environment_with_config_fallback(self):
        source = (PERCEPTION_ROOT / "main.py").read_text(encoding="utf-8")

        self.assertIn(
            'os.environ.get("MCP_PORT") or cfg.get("mcp_port", 15720)', source
        )
        self.assertIn(
            'os.environ.get("WS_PORT") or cfg.get("ws_port", 15721)', source
        )

    def test_five_audio_fixtures_are_tracked_under_one_megabyte(self):
        fixture_dir = PERCEPTION_ROOT / "tests" / "fixtures" / "audio"
        expected = {
            "demo_01_short.wav",
            "demo_02_short.wav",
            "demo_03_normal.wav",
            "demo_04_normal.wav",
            "demo_05_normal.wav",
        }
        fixtures = {path.name for path in fixture_dir.glob("*.wav")}

        self.assertEqual(fixtures, expected)
        for filename in expected:
            with self.subTest(filename=filename):
                self.assertLess((fixture_dir / filename).stat().st_size, 1024 * 1024)


if __name__ == "__main__":
    unittest.main()

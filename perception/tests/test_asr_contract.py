import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERCEPTION_ROOT))


def _install_ros_stubs():
    rclpy = types.ModuleType("rclpy")
    rclpy.node = types.ModuleType("rclpy.node")
    rclpy.node.Node = object
    rclpy.qos = types.ModuleType("rclpy.qos")
    rclpy.qos.QoSProfile = lambda **kwargs: kwargs
    rclpy.qos.ReliabilityPolicy = types.SimpleNamespace(BEST_EFFORT="BEST_EFFORT")
    rclpy.qos.HistoryPolicy = types.SimpleNamespace(KEEP_LAST="KEEP_LAST")
    rclpy.qos.DurabilityPolicy = types.SimpleNamespace(VOLATILE="VOLATILE")
    std_msgs = types.ModuleType("std_msgs")
    std_msgs.msg = types.ModuleType("std_msgs.msg")
    std_msgs.msg.String = type("String", (), {})

    sys.modules["rclpy"] = rclpy
    sys.modules["rclpy.node"] = rclpy.node
    sys.modules["rclpy.qos"] = rclpy.qos
    sys.modules["std_msgs"] = std_msgs
    sys.modules["std_msgs.msg"] = std_msgs.msg


class ASRContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_ros_stubs()
        cls.asr = importlib.import_module("plugins.asr")

    def test_tool_contract_stays_asr_with_original_io_formats(self):
        tool = self.asr.TOOLS[0]

        self.assertEqual(tool["name"], "asr")
        self.assertEqual(
            tool["topic_in"],
            [{"format": "audio/pcm-16k", "desc": "mic audio input"}],
        )
        self.assertEqual(
            tool["topic_out"],
            [{"format": "data/json", "desc": "ASR result event"}],
        )

    def test_asr_mode_defaults_to_offline(self):
        self.assertEqual(self.asr._resolve_asr_mode({}), "offline")

    def test_asr_mode_accepts_four_internal_modes(self):
        for mode in ("offline", "online", "streaming", "segmented"):
            with self.subTest(mode=mode):
                self.assertEqual(self.asr._resolve_asr_mode({"mode": mode}), mode)

    def test_asr_mode_rejects_unknown_value(self):
        with self.assertRaises(ValueError):
            self.asr._resolve_asr_mode({"mode": "cloud"})

    def test_output_topic_stays_compatible(self):
        self.assertEqual(
            self.asr._asr_output_topic("/robot/mic/audio"),
            "/robot/mic/audio/asr",
        )

    def test_vad_session_is_available_for_websocket_server(self):
        self.assertTrue(hasattr(self.asr, "VadSession"))

    def test_kws_is_disabled_unless_explicitly_requested(self):
        self.assertFalse(self.asr._is_kws_enabled(None))
        self.assertFalse(self.asr._is_kws_enabled({}))
        self.assertFalse(
            self.asr._is_kws_enabled({"enabled": False, "trigger_mode": "kws"})
        )
        self.assertFalse(self.asr._is_kws_enabled({"enabled": True}))
        self.assertTrue(
            self.asr._is_kws_enabled({"enabled": True, "trigger_mode": "kws"})
        )

    def test_offline_mode_uses_official_paraformer_adapter(self):
        expected = object()
        with mock.patch(
            "plugins.asr_offline.OfflineASRAdapter.get_instance",
            return_value=expected,
        ) as get_instance:
            adapter = self.asr._build_asr_adapter(
                {
                    "mode": "offline",
                    "model_path": "/models/sherpa-onnx/asr-offline",
                    "device": "cpu",
                    "num_threads": 2,
                }
            )

        self.assertIs(adapter, expected)
        get_instance.assert_called_once_with(
            model_path="/models/sherpa-onnx/asr-offline",
            config=None,
            num_threads=2,
            provider="cpu",
        )

    def test_latest_main_vad_and_loading_fixes_remain_present(self):
        source = (PERCEPTION_ROOT / "plugins" / "asr.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("PREBUF_FRAMES = 5", source)
        self.assertIn("if self._vad_proc and not self._vad_proc.is_alive()", source)
        self.assertIn("def _load_model_async", source)


if __name__ == "__main__":
    unittest.main()

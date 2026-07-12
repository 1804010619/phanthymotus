import importlib
import io
import queue
import sys
import threading
import time
import types
import unittest
import wave
from concurrent.futures import ThreadPoolExecutor
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

    def test_asr_mode_resolves_legacy_names_to_canonical_modes(self):
        expected = {
            "offline": "offline",
            "segmented": "offline",
            "streaming": "streaming",
            "online": "streaming",
        }
        for configured, canonical in expected.items():
            with self.subTest(mode=configured):
                self.assertEqual(
                    self.asr._resolve_asr_mode({"mode": configured}), canonical
                )

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

    def test_initial_model_load_is_async(self):
        load_started = threading.Event()
        release_load = threading.Event()

        def slow_build(_cfg):
            load_started.set()
            release_load.wait(timeout=0.25)
            return object()

        with mock.patch("plugins.asr._build_asr_adapter", side_effect=slow_build):
            started_at = time.monotonic()
            plugin = self.asr.ASRPlugin({"mode": "offline"}, executor=object())
            elapsed = time.monotonic() - started_at
            self.assertTrue(load_started.wait(timeout=0.1))
            self.assertLess(elapsed, 0.1)
            self.assertTrue(plugin._loading)
            release_load.set()

        deadline = time.monotonic() + 1
        while plugin._loading and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(plugin._loading)
        self.assertIsNotNone(plugin._adapter)

    def test_ros_node_receives_configured_vad_model_directory(self):
        class Executor:
            def add_node(self, _node):
                pass

        node = types.SimpleNamespace(start=lambda: {"state": "running"})
        with mock.patch.object(self.asr.ASRPlugin, "_load_model_async"):
            plugin = self.asr.ASRPlugin(
                {
                    "mode": "offline",
                    "vad": {"model_dir": "/custom/vad"},
                },
                executor=Executor(),
            )
        with plugin._state_lock:
            plugin._loading = False
            plugin._adapter = object()

        with mock.patch("plugins.asr._ASRNode", return_value=node) as node_class:
            plugin.dispatch(
                "asr",
                {
                    "action": "start",
                    "instance_id": "mic",
                    "input_topic": "/robot/mic/audio",
                },
            )

        self.assertIn("/custom/vad", node_class.call_args.args)

    def test_concurrent_starts_create_one_ros_node(self):
        class Executor:
            def add_node(self, _node):
                pass

        entered_constructor = threading.Event()
        release_constructor = threading.Event()
        constructor_calls = 0
        calls_lock = threading.Lock()

        def create_node(*_args, **_kwargs):
            nonlocal constructor_calls
            with calls_lock:
                constructor_calls += 1
            entered_constructor.set()
            release_constructor.wait(timeout=1)
            return types.SimpleNamespace(start=lambda: {"state": "running"})

        with mock.patch.object(self.asr.ASRPlugin, "_load_model_async"):
            plugin = self.asr.ASRPlugin({"mode": "offline"}, Executor())
        with plugin._state_lock:
            plugin._loading = False
            plugin._adapter = object()
        args = {
            "action": "start",
            "instance_id": "mic",
            "input_topic": "/robot/mic/audio",
        }

        with mock.patch("plugins.asr._ASRNode", side_effect=create_node):
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(plugin.dispatch, "asr", args)
                self.assertTrue(entered_constructor.wait(timeout=0.2))
                second = executor.submit(plugin.dispatch, "asr", args)
                time.sleep(0.05)
                release_constructor.set()
                first.result(timeout=1)
                second.result(timeout=1)

        self.assertEqual(constructor_calls, 1)

    def test_audio_queue_drop_is_reported_in_status(self):
        class FullQueue:
            def put_nowait(self, _item):
                raise queue.Full

            def qsize(self):
                return 1000

        node = object.__new__(self.asr._ASRNode)
        node._stop_event = threading.Event()
        node._vad_proc = None
        node._pcm_queue = FullQueue()
        node._utterance_queue = None
        node._input_topic = "/robot/mic/audio"
        node._output_topic = "/robot/mic/audio/asr"
        node.state = "running"
        node._received_chunks = 0
        node._dropped_chunks = 0
        node._completed_utterances = 0
        node._transcribe_errors = 0
        node._last_audio_ts = None
        node._last_result_ts = None
        node._last_error = None

        msg = types.SimpleNamespace(
            data=b"\x00\x00" * 160,
            header=types.SimpleNamespace(
                stamp=types.SimpleNamespace(sec=12, nanosec=500_000_000)
            ),
        )
        node._audio_cb(msg)
        status = node._status_dict()

        self.assertEqual(status["metrics"]["received_chunks"], 1)
        self.assertEqual(status["metrics"]["dropped_chunks"], 1)
        self.assertEqual(status["metrics"]["pcm_queue_depth"], 1000)

    def test_instance_info_includes_node_metrics(self):
        metrics = {"received_chunks": 12, "dropped_chunks": 3}
        node = types.SimpleNamespace(
            state="running",
            _input_topic="/robot/mic/audio",
            _output_topic="/robot/mic/audio/asr",
            _status_dict=lambda: {"state": "running", "metrics": metrics},
        )
        plugin = object.__new__(self.asr.ASRPlugin)
        plugin._state_lock = threading.Lock()
        plugin._lifecycle_lock = threading.RLock()
        plugin._loading = False
        plugin._load_error = None
        plugin._adapter = object()
        plugin._asr_model = "paraformer-zh-en"
        plugin._nodes = {"mic": node}

        result = plugin.dispatch("asr", {"action": "info", "instance_id": "mic"})

        self.assertEqual(result["metrics"], metrics)

    def test_streaming_adapter_serializes_shared_recognizer(self):
        class Stream:
            def __init__(self):
                self.ready = True

            def accept_waveform(self, _sample_rate, _samples):
                pass

            def input_finished(self):
                pass

        class Recognizer:
            def __init__(self):
                self.active = 0
                self.max_active = 0
                self.state_lock = threading.Lock()

            def create_stream(self):
                return Stream()

            def is_ready(self, stream):
                return stream.ready

            def decode_streams(self, streams):
                with self.state_lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.05)
                streams[0].ready = False
                with self.state_lock:
                    self.active -= 1

            def get_result(self, _stream):
                return types.SimpleNamespace(text="ok")

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 160)

        recognizer = Recognizer()
        adapter = object.__new__(self.asr.SherpaOnnxASRAdapter)
        adapter._recognizer = recognizer
        adapter._decode_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(
                    lambda _: adapter.transcribe(wav_buffer.getvalue(), "zh-CN"),
                    range(2),
                )
            )

        self.assertEqual(results, ["ok", "ok"])
        self.assertEqual(recognizer.max_active, 1)


if __name__ == "__main__":
    unittest.main()

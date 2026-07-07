#!/usr/bin/env python3
"""
plugins/kws.py — KWSPlugin: Keyword Spotting (wake word detection) using sherpa-onnx.

Subscribes to audio/pcm-16k topics, runs keyword detection via sherpa-onnx,
publishes detection events (data/json) to {topic}/kws.
"""

from __future__ import annotations

import json
import logging
import os
import struct
import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000

_LOW_LAT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
    durability=DurabilityPolicy.VOLATILE,
)

_PUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

TOOLS = [
    {
        "name": "kws",
        "type": "processor",
        "multiInstance": True,
        "description": "KWS — keyword spotting (wake word detection)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "info", "config"],
                    "description": "Action to perform"
                },
                "input_topic": {
                    "type": "string",
                    "description": "ROS2 audio topic (required for action=start)"
                },
            },
            "required": ["action"]
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "model_dir":      {"type": "string", "description": "sherpa-onnx KWS 模型目录路径", "scope": "shared"},
                "keywords":       {"type": "array", "items": {"type": "string"}, "description": "唤醒词列表 (如 ['你好小幻', '小幻小幻'])", "scope": "shared"},
                "hw_provider":    {"type": "string", "enum": ["cuda", "cpu"], "default": "cpu", "description": "推理后端", "scope": "shared"},
                "num_threads":    {"type": "integer", "description": "推理线程数", "default": 2, "scope": "shared"},
            },
            "required": []
        },
        "topic_in":  [{"format": "audio/pcm-16k", "desc": "mic audio input"}],
        "topic_out": [{"format": "data/json",     "desc": "keyword detection event"}],
    }
]


class _KWSNode(Node):
    def __init__(self, input_topic: str, spotter, node_suffix: str = ''):
        node_name = f"kws_{node_suffix}" if node_suffix else "kws"
        super().__init__(node_name)
        self._input_topic = input_topic
        self._output_topic = f"{input_topic}/kws"
        self._spotter = spotter
        self._stream = spotter.create_stream()
        self.state = "idle"
        self._sub = None
        self._pub = self.create_publisher(String, self._output_topic, _PUB_QOS)
        self._lock = threading.Lock()
        self._cooldown_until = 0.0  # prevent rapid re-triggers

    def start(self) -> dict:
        if self.state == "running":
            return self._status_dict()
        from audio_msgs.msg import AudioChunk
        self._sub = self.create_subscription(
            AudioChunk, self._input_topic, self._audio_cb, _LOW_LAT_QOS)
        self.state = "running"
        log.info(f"[kws] started on {self._input_topic}")
        return self._status_dict()

    def stop(self) -> dict:
        if self._sub:
            self.destroy_subscription(self._sub)
            self._sub = None
        self.state = "idle"
        return {"state": "idle"}

    def _audio_cb(self, msg):
        pcm = bytes(msg.data)
        n = len(pcm) // 2
        if n < 160:  # minimum 10ms
            return
        samples = struct.unpack(f'<{n}h', pcm)
        float_samples = [s / 32768.0 for s in samples]

        with self._lock:
            self._stream.accept_waveform(SAMPLE_RATE, float_samples)
            while self._spotter.is_ready(self._stream):
                self._spotter.decode_stream(self._stream)
            result = self._spotter.get_result(self._stream)
            if result.keyword:
                now = time.time()
                if now < self._cooldown_until:
                    return
                self._cooldown_until = now + 1.0  # 1s cooldown
                event = {
                    "keyword": result.keyword.strip(),
                    "timestamp": now,
                }
                ros_msg = String()
                ros_msg.data = json.dumps(event, ensure_ascii=False)
                self._pub.publish(ros_msg)
                log.info(f"[kws] detected: {result.keyword.strip()}")

    def _status_dict(self) -> dict:
        return {
            "state": self.state,
            "topic_in":  [{"topic": self._input_topic, "format": "audio/pcm-16k", "desc": ""}],
            "topic_out": [{"topic": self._output_topic, "format": "data/json", "desc": "keyword event"}],
        }


class KWSPlugin:
    PREFIX = "kws"

    def __init__(self, plugin_cfg: dict, executor):
        self._model_dir = plugin_cfg.get('model_dir', '/work/models/sherpa-onnx/kws')
        self._keywords = plugin_cfg.get('keywords', ['你好小幻'])
        self._hw_provider = plugin_cfg.get('hw_provider', 'cpu')
        self._num_threads = int(plugin_cfg.get('num_threads', 2))
        self._spotter = self._build_spotter()
        self._nodes: dict[str, _KWSNode] = {}
        self._executor = executor
        log.info(f"[kws] plugin init: model_dir={self._model_dir}, "
                 f"keywords={self._keywords}")

    def _build_spotter(self):
        from utils.model_downloader import ensure_model
        ensure_model("kws", self._model_dir)

        import sherpa_onnx

        # Find encoder/decoder/joiner files (prefer int8 + chunk-8 for low latency)
        encoder = self._find_model_file("encoder", prefer_int8=True, prefer_chunk=8)
        decoder = self._find_model_file("decoder", prefer_int8=False, prefer_chunk=8)
        joiner = self._find_model_file("joiner", prefer_int8=True, prefer_chunk=8)
        tokens = os.path.join(self._model_dir, "tokens.txt")

        log.info(f"[kws] loading models: encoder={os.path.basename(encoder)}, "
                 f"decoder={os.path.basename(decoder)}, joiner={os.path.basename(joiner)}")

        # Write keywords to a temp file (sherpa-onnx requires a file path)
        keywords_file = os.path.join(self._model_dir, "keywords.txt")
        with open(keywords_file, 'w', encoding='utf-8') as f:
            for kw in self._keywords:
                f.write(f"{kw} @{kw}\n")
        log.info(f"[kws] keywords written to {keywords_file}: {self._keywords}")

        return sherpa_onnx.KeywordSpotter(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            keywords_file=keywords_file,
            num_threads=self._num_threads,
            provider=self._hw_provider,
        )

    def _find_model_file(self, prefix: str, prefer_int8: bool = True, prefer_chunk: int = 8) -> str:
        """Find the best matching model file for a given prefix (encoder/decoder/joiner)."""
        import glob
        pattern = os.path.join(self._model_dir, f"{prefix}-*.onnx")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No {prefix} model found in {self._model_dir}")

        # Filter by chunk preference
        chunk_str = f"chunk-{prefer_chunk}"
        chunk_files = [f for f in files if chunk_str in f]
        candidates = chunk_files if chunk_files else files

        # Filter by int8 preference
        if prefer_int8:
            int8_files = [f for f in candidates if "int8" in f]
            if int8_files:
                return int8_files[0]
        else:
            # Prefer non-int8 (fp32) for decoder
            fp32_files = [f for f in candidates if "int8" not in f]
            if fp32_files:
                return fp32_files[0]

        return candidates[0]

    def get_tools(self) -> list:
        return TOOLS

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action") if name == "kws" else name
        instance_id = args.get("instance_id", "")

        if action == "info":
            input_topic = args.get("input_topic", "")
            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                return {
                    "name": "KWS", "manufacture": "Embodied", "model": "kws",
                    "state": node.state,
                    "topic_in":  [{"topic": node._input_topic, "format": "audio/pcm-16k", "desc": ""}],
                    "topic_out": [{"topic": node._output_topic, "format": "data/json", "desc": ""}],
                    "desc": "KWS — keyword spotting (wake word detection)",
                }
            if instance_id:
                inferred_out = f"{input_topic}/kws" if input_topic else ""
                return {
                    "name": "KWS", "manufacture": "Embodied", "model": "kws",
                    "state": "idle",
                    "topic_in":  [{"topic": input_topic, "format": "audio/pcm-16k", "desc": ""}] if input_topic else [],
                    "topic_out": [{"topic": inferred_out, "format": "data/json", "desc": ""}] if inferred_out else [],
                    "desc": "KWS — keyword spotting (wake word detection)",
                }
            # Aggregate
            if self._nodes:
                topics_in = [{"topic": n._input_topic, "format": "audio/pcm-16k", "desc": ""} for n in self._nodes.values()]
                topics_out = [{"topic": n._output_topic, "format": "data/json", "desc": ""} for n in self._nodes.values()]
                states = list(set(n.state for n in self._nodes.values()))
                state = "running" if "running" in states else states[0] if states else "idle"
            else:
                inferred_out = f"{input_topic}/kws" if input_topic else ""
                topics_in = [{"topic": input_topic, "format": "audio/pcm-16k", "desc": ""}]
                topics_out = [{"topic": inferred_out, "format": "data/json", "desc": ""}]
                state = "idle"
            return {
                "name": "KWS", "manufacture": "Embodied", "model": "kws",
                "state": state,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": "KWS — keyword spotting (wake word detection)",
            }

        elif action == "start":
            input_topic = args.get("input_topic")
            if not input_topic:
                topics_list = args.get("input_topics") or []
                if topics_list:
                    input_topic = topics_list[0]
            if not input_topic:
                raise ValueError("input_topic is required")
            node_key = instance_id or input_topic
            if node_key not in self._nodes:
                node = _KWSNode(input_topic, self._spotter,
                               node_suffix=node_key.replace('/', '_').replace('-', '_'))
                self._executor.add_node(node)
                self._nodes[node_key] = node
            return self._nodes[node_key].start()

        elif action == "stop":
            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                result = node.stop()
                self._executor.remove_node(node)
                del self._nodes[instance_id]
                return result
            elif not instance_id and self._nodes:
                results = []
                for key in list(self._nodes.keys()):
                    node = self._nodes[key]
                    node.stop()
                    self._executor.remove_node(node)
                    del self._nodes[key]
                    results.append(key)
                return {"state": "idle", "stopped_instances": results}
            return {"state": "idle"}

        elif action == "config":
            cfg = {k: v for k, v in args.items() if k not in ('action', 'instance_id') and v is not None and v != ''}
            if 'keywords' in cfg:
                self._keywords = cfg['keywords']
            # Rebuild spotter with new config and restart all nodes
            for key in list(self._nodes.keys()):
                self._nodes[key].stop()
                self._executor.remove_node(self._nodes[key])
                del self._nodes[key]
            self._spotter = self._build_spotter()
            return {"status": "configured"}

        return None

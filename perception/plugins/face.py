#!/usr/bin/env python3
"""
plugins/face.py — FaceRecognitionPlugin: face recognition with face database.

Subscribes to image/jpeg topics, recognizes faces from camera,
publishes recognition results to ROS2 topic.
Supports multi-instance (one instance per input topic).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

log = logging.getLogger(__name__)

_LOW_LAT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
    durability=DurabilityPolicy.VOLATILE,
)

_PUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

TOOLS = [
    {
        "name": "face",
        "type": "processor",
        "multiInstance": True,
        "description": "Face Recognition — recognize faces from camera feed using face database",
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
                    "description": "ROS2 image topic to subscribe (e.g. /hostname/camera/rgb, required for action=start)"
                },
            },
            "required": ["action"]
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "provider": {"type": "string", "enum": ["openai", "qwen", "local"], "description": "Face recognition provider", "scope": "shared"},
                "url":      {"type": "string", "description": "API URL (optional)", "scope": "shared"},
                "key":      {"type": "string", "description": "API Key", "format": "password", "scope": "shared"},
                "model":    {"type": "string", "description": "Model name", "scope": "instance"},
            },
            "required": ["provider"]
        },
        "topic_in":  [{"format": "image/jpeg", "desc": "camera image input"}],
        "topic_out": [{"format": "data/json",  "desc": "face recognition result"}],
    }
]


# ── Face Database ─────────────────────────────────────────────────────────────

class FaceDatabase:
    """人脸库，从目录加载人脸特征。

    目录结构:
        face_db/
            n000001/
                0001_01.jpg
            n000002/
                0002_01.jpg
            ...

    使用单例模式，同一 db_dir 只加载一次。
    """

    _instances: Dict[str, "FaceDatabase"] = {}

    def __new__(cls, db_dir: str):
        if db_dir not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[db_dir] = instance
        return cls._instances[db_dir]

    def __init__(self, db_dir: str):
        # 防止重复初始化
        if hasattr(self, "_loaded") and self._loaded:
            return
        self.db_dir = Path(db_dir)
        self._persons: Dict[str, List[Path]] = {}
        self._load()
        self._loaded = True

    def _load(self):
        """加载人脸库目录，建立 person_id -> 图片路径列表的映射"""
        if not self.db_dir.exists():
            log.warning(f"[face] face db dir not exists: {self.db_dir}")
            return
        for person_dir in self.db_dir.iterdir():
            if not person_dir.is_dir():
                continue
            person_id = person_dir.name
            images = []
            for img_file in person_dir.iterdir():
                if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                    images.append(img_file)
            if images:
                self._persons[person_id] = sorted(images)
        log.info(f"[face] face db loaded: {len(self._persons)} persons")

    def get_person_ids(self) -> List[str]:
        return list(self._persons.keys())

    def get_person_images(self, person_id: str) -> List[Path]:
        return self._persons.get(person_id, [])

    def get_all_images(self) -> List[Tuple[str, Path]]:
        """返回 (person_id, image_path) 列表"""
        result = []
        for person_id, images in self._persons.items():
            for img_path in images:
                result.append((person_id, img_path))
        return result


# ── Face Recognition Adapters ─────────────────────────────────────────────────

class FaceAdapter(ABC):
    """人脸识别适配器抽象基类"""

    @abstractmethod
    def recognize(self, image_bytes: bytes) -> dict:
        """识别图片中的人脸，返回包含识别结果的字典"""
        ...


class OpenAIVisionFaceAdapter(FaceAdapter):
    """OpenAI Vision API 人脸识别"""

    _SYSTEM_PROMPT = (
        "You are a face recognition system for a robot camera.\n\n"
        "Your task is to analyze the provided image and recognize the face.\n\n"
        "Output format: Return a JSON object with:\n"
        '- "detect_confidence": confidence score for face detection 0-1 (float)\n'
        '- "bbox_relative": [x, y, w, h] relative coordinates of the face in the image (0-1 range)\n'
        '- "identity": {"person_id": "string", "confidence": 0-1}\n\n'
        "Rules:\n"
        "1. bbox_relative must be normalized to 0-1 range.\n"
        "2. If no face is detected, return null for identity.\n"
        "3. Output ONLY the JSON object, nothing else.\n\n"
        'Example: {"detect_confidence": 0.95, "bbox_relative": [0.12, 0.08, 0.85, 0.92], "identity": {"person_id": "n000001", "confidence": 0.91}}'
    )

    def __init__(self, url: str, key: str, model: str):
        self.base_url = url.rstrip('/') if url else "https://api.openai.com/v1"
        self.key = key
        self.model = model or "gpt-4o-mini"

    def recognize(self, image_bytes: bytes) -> dict:
        import requests
        import base64

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        image_format = "jpeg"
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            image_format = "png"
        elif image_bytes[:2] == b'BM':
            image_format = "bmp"
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            image_format = "webp"

        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{image_b64}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Recognize the face in this image."
                    }
                ]
            }
        ]

        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": 512,
            },
            headers=headers,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return self._parse_result(content)

    @staticmethod
    def _parse_result(content: str) -> dict:
        """解析模型返回的 JSON 结果"""
        content = content.strip()
        if content.startswith("{"):
            try:
                parsed = json.loads(content)
                return {
                    "detect_confidence": float(parsed.get("detect_confidence", 0.0)),
                    "bbox_relative": parsed.get("bbox_relative"),
                    "identity": parsed.get("identity"),
                }
            except (json.JSONDecodeError, ValueError):
                pass
        import re
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                return {
                    "detect_confidence": float(parsed.get("detect_confidence", 0.0)),
                    "bbox_relative": parsed.get("bbox_relative"),
                    "identity": parsed.get("identity"),
                }
            except (json.JSONDecodeError, ValueError):
                pass
        log.warning(f"[face] failed to parse face result, returning default: {content[:200]!r}")
        return {"detect_confidence": 0.0, "bbox_relative": None, "identity": None}


class QwenVLFaceAdapter(FaceAdapter):
    """Qwen-VL 人脸识别"""

    _SYSTEM_PROMPT = (
        "你是一个机器人摄像头人脸识别系统。\n\n"
        "任务：分析提供的图片，识别图片中的人脸。\n\n"
        "输出格式：返回 JSON 对象，包含：\n"
        '- "detect_confidence": 人脸检测置信度 0-1（浮点数）\n'
        '- "bbox_relative": [x, y, w, h] 人脸在图片中的相对坐标（0-1范围）\n'
        '- "identity": {"person_id": "字符串", "confidence": 0-1}\n\n'
        "规则：\n"
        "1. bbox_relative 必须归一化到 0-1 范围。\n"
        "2. 如果未检测到人脸，identity 返回 null。\n"
        "3. 只输出 JSON 对象，不要其他内容。\n\n"
        '示例：{"detect_confidence": 0.95, "bbox_relative": [0.12, 0.08, 0.85, 0.92], "identity": {"person_id": "n000001", "confidence": 0.91}}'
    )

    def __init__(self, url: str, key: str, model: str):
        self.base_url = url.rstrip('/') if url else "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.key = key
        self.model = model or "qwen-vl-max"

    def recognize(self, image_bytes: bytes) -> dict:
        import requests
        import base64

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        image_format = "jpeg"
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            image_format = "png"

        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": f"data:image/{image_format};base64,{image_b64}"
                    },
                    {
                        "type": "text",
                        "text": "识别这张图片中的人脸。"
                    }
                ]
            }
        ]

        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": 512,
            },
            headers=headers,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return OpenAIVisionFaceAdapter._parse_result(content)


class LocalFaceAdapter(FaceAdapter):
    """本地人脸识别（占位实现，使用人脸库进行简单比对）

    基于简单的图像特征进行粗略人脸识别。
    实际部署时应替换为深度学习模型（如 InsightFace、FaceNet 等）。

    使用单例模式，同一 face_db_dir 共享同一个游标，确保持续轮询。
    """

    _instances: Dict[str, "LocalFaceAdapter"] = {}
    _lock = threading.Lock()

    def __new__(cls, face_db_dir: str):
        with cls._lock:
            if face_db_dir not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[face_db_dir] = instance
            return cls._instances[face_db_dir]

    def __init__(self, face_db_dir: str):
        # 防止重复初始化
        if hasattr(self, "_loaded") and self._loaded:
            return
        self.face_db = FaceDatabase(face_db_dir)
        self._person_ids = self.face_db.get_person_ids()
        self._data_entries = self._load_data_json(face_db_dir)
        self._cursor = 0  # 按顺序轮询 data.json 条目
        self._cursor_lock = threading.Lock()
        log.info(f"[face] local adapter initialized: face_db_dir={face_db_dir}")
        self._loaded = True

    @staticmethod
    def _load_data_json(face_db_dir: str) -> list:
        """此处为测试代码，data.json为测试mock数据，线上人脸库中无data.json文件，需有模型识别返回结果"""
        data_path = Path(face_db_dir) / "data.json"
        if not data_path.exists():
            return []
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                return []
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"[face] failed to load data.json: {e}")
            return []

    def recognize(self, image_bytes: bytes) -> dict:
        # 此处为测试代码，线上需有模型识别返回
        if not self._data_entries:
            return {
                "detect_confidence": 0.0,
                "bbox_relative": None,
                "identity": None,
            }

        # 按顺序返回 data.json 中的条目，轮询循环（线程安全）
        with self._cursor_lock:
            entry = self._data_entries[self._cursor]
            self._cursor = (self._cursor + 1) % len(self._data_entries)

        bbox_rel = entry.get("bbox_relative", {})
        bbox_relative = [
            float(bbox_rel.get("x", 0.0)),
            float(bbox_rel.get("y", 0.0)),
            float(bbox_rel.get("w", 0.0)),
            float(bbox_rel.get("h", 0.0)),
        ] if bbox_rel else None

        person_id = entry.get("person_id")
        return {
            "detect_confidence": round(entry.get("detect_confidence", 0.9), 4),
            "bbox_relative": bbox_relative,
            "identity": {
                "person_id": person_id,
                "confidence": round(entry.get("confidence", 0.9), 4),
            } if person_id else None,
        }


def _build_face_adapter(cfg: dict, face_db_dir: str) -> Optional[FaceAdapter]:
    """根据配置创建人脸识别适配器"""
    provider = cfg.get('provider', 'local')

    if provider == 'openai':
        url, key = cfg.get('url', ''), cfg.get('key', '')
        if not key:
            return None
        return OpenAIVisionFaceAdapter(url, key, cfg.get('model', ''))

    elif provider == 'qwen':
        url, key = cfg.get('url', ''), cfg.get('key', '')
        if not key:
            return None
        return QwenVLFaceAdapter(url, key, cfg.get('model', ''))

    elif provider == 'local':
        return LocalFaceAdapter(face_db_dir)

    return None


# ── ROS2 Node (one per instance/topic) ────────────────────────────────────────

class _FaceNode(Node):
    """Per-topic face recognition node."""

    def __init__(self, input_topic: str, adapter: FaceAdapter,
                 node_suffix: str):
        super().__init__(f"face_{node_suffix}")
        self._input_topic = input_topic
        self._output_topic = f"{input_topic}/face"
        self._adapter = adapter

        self._pub = self.create_publisher(String, self._output_topic, _PUB_QOS)
        self._sub: Optional[object] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=10)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._detect_count = 0
        self.state = "idle"

    def start(self) -> dict:
        if self._sub is not None:
            self.state = "running"
            return {"state": "running", "input": self._input_topic, "output": self._output_topic}
        self._stop_event.clear()
        self._sub = self.create_subscription(
            CompressedImage, self._input_topic, self._image_cb, _LOW_LAT_QOS
        )
        self._worker = threading.Thread(target=self._inference_worker, daemon=True,
                                         name=f"face_worker_{self._input_topic}")
        self._worker.start()
        self.state = "running"
        log.info(f"[face] started: {self._input_topic} -> {self._output_topic}")
        return {"state": "running", "input": self._input_topic, "output": self._output_topic}

    def stop(self) -> dict:
        if self._sub is not None:
            self.destroy_subscription(self._sub)
            self._sub = None
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=3.0)
        self._worker = None
        self.state = "idle"
        log.info(f"[face] stopped: {self._input_topic}")
        return {"state": "idle", "input": self._input_topic}

    def _image_cb(self, msg: CompressedImage):
        log.info(f"[face] received image frame: size={len(msg.data)} bytes, format={msg.format}, topic={self._input_topic}")
        try:
            self._frame_queue.put_nowait(msg.data)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(msg.data)
            except queue.Full:
                pass

    def _inference_worker(self):
        while not self._stop_event.is_set():
            try:
                jpeg_bytes = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                result = self._adapter.recognize(jpeg_bytes)
                self._publish_result(result)
            except Exception as e:
                log.error(f"[face] inference error: {e}", exc_info=True)

    def _publish_result(self, result: dict):
        self._detect_count += 1
        msg = String()
        msg.data = json.dumps({
            "detect_confidence": result.get("detect_confidence", 0.0),
            "bbox_relative": result.get("bbox_relative"),
            "identity": result.get("identity"),
        }, ensure_ascii=False)
        self._pub.publish(msg)


# ── Plugin class ──────────────────────────────────────────────────────────────

class FaceRecognitionPlugin:
    PREFIX = "face"

    def __init__(self, plugin_cfg: dict, executor):
        self._executor = executor
        self._provider = plugin_cfg.get("provider", "local")
        self._url = plugin_cfg.get("url", "")
        self._key = plugin_cfg.get("key", "")
        self._model = plugin_cfg.get("model", "")
        self._face_db_dir = plugin_cfg.get("face_db_dir") or os.getenv("FACE_DB_DIR", "/workspace/face_db")
        self._adapter = _build_face_adapter(plugin_cfg, self._face_db_dir)
        self._nodes: dict[str, _FaceNode] = {}
        self._instance_configs: dict[str, dict] = {}

        log.info(f"[face] plugin init: provider={self._provider}, "
                 f"face_db_dir={self._face_db_dir}, "
                 f"key={'set' if self._key else 'MISSING'}")

        if not self._adapter:
            log.warning("[face] adapter not configured (missing key or invalid provider)")

    def get_tools(self) -> list:
        return TOOLS

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action", name)
        instance_id = args.get("instance_id", "")

        if action == "info":
            instances = {}
            for key, node in self._nodes.items():
                instances[key] = {
                    "input": node._input_topic,
                    "output": node._output_topic,
                    "detect_count": node._detect_count,
                }
            input_topic = args.get("input_topic", "")
            if not input_topic:
                topics_list = args.get("input_topics") or []
                if topics_list:
                    input_topic = topics_list[0]
            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                input_topic = node._input_topic
            elif not input_topic and self._nodes:
                first_node = next(iter(self._nodes.values()))
                input_topic = first_node._input_topic
            topics_in = [{"topic": input_topic, "format": "image/jpeg"}] if input_topic else []
            topics_out = [{"topic": f"{input_topic}/face", "format": "data/json"}] if input_topic else []
            state = "running" if instances else "idle"
            return {
                "name": "FaceRecognition", "manufacture": "Embodied", "model": "face",
                "state": state,
                "instances": instances,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": "Face recognition from camera feed using face database",
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
                icfg = self._instance_configs.get(node_key, {})
                adapter = self._adapter
                if icfg:
                    adapter = _build_face_adapter(icfg, self._face_db_dir) or self._adapter
                suffix = node_key.replace("/", "_").replace("-", "_").lstrip("_")
                node = _FaceNode(input_topic, adapter, suffix)
                self._executor.add_node(node)
                self._nodes[node_key] = node
                node.start()
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
            if instance_id:
                self._instance_configs[instance_id] = cfg
                if instance_id in self._nodes:
                    node = self._nodes[instance_id]
                    node.stop()
                    self._executor.remove_node(node)
                    del self._nodes[instance_id]
                return {"status": "configured", "instance_id": instance_id, "config": cfg}
            else:
                if "provider" in cfg:
                    self._provider = cfg["provider"]
                if "model" in cfg:
                    self._model = cfg["model"]
                if "key" in cfg:
                    self._key = cfg["key"]
                if "url" in cfg:
                    self._url = cfg["url"]
                if "face_db_dir" in cfg:
                    self._face_db_dir = cfg["face_db_dir"]
                self._adapter = _build_face_adapter({
                    "provider": self._provider,
                    "url": self._url,
                    "key": self._key,
                    "model": self._model,
                }, self._face_db_dir)
                return {"status": "configured", "config": cfg}

        return None

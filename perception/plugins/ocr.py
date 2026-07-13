#!/usr/bin/env python3
"""
plugins/ocr.py — OCRPlugin: OCR 文字识别封装。

订阅 image/jpeg topic，持续进行 OCR 识别并发布结果到 ROS2 topic。
参考 asr.py 架构设计。
"""

from __future__ import annotations

import base64
import json
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

log = logging.getLogger(__name__)

_LOW_LAT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,  # 使用 RELIABLE 确保消息可靠送达
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

TOOLS = [
    {
        "name": "ocr",
        "type": "processor",
        "multiInstance": True,
        "description": "OCR — recognize text in camera feed via image topic subscription",
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
                "provider": {"type": "string", "enum": ["openai", "qwen", "google", "tesseract"], "description": "OCR 服务商", "scope": "shared"},
                "url":      {"type": "string", "description": "API URL (可选)", "scope": "shared"},
                "key":      {"type": "string", "description": "API Key", "format": "password", "scope": "shared"},
                "model":    {"type": "string", "description": "模型名称", "scope": "instance"},
                "language": {"type": "string", "description": "默认语言", "default": "zh", "scope": "instance"},
            },
            "required": ["provider"]
        },
        "topic_in":  [{"format": "image/jpeg", "desc": "camera image input"}],
        "topic_out": [{"format": "data/json",  "desc": "OCR result with text"}],
    }
]


# ── OCR Adapters ──────────────────────────────────────────────────────────────

class OCRAdapter(ABC):
    """OCR 适配器抽象基类"""

    @abstractmethod
    def recognize(self, image_bytes: bytes, language: str = "zh") -> str:
        """识别图片中的文字，返回文本"""
        ...


class OpenAIVisionAdapter(OCRAdapter):
    """OpenAI Vision API (GPT-4o / GPT-4o-mini) OCR"""

    _SYSTEM_PROMPT = (
        "You are an OCR (Optical Character Recognition) system. "
        "Your task is to extract ALL text from the provided image accurately.\n\n"
        "Rules:\n"
        "1. Extract text exactly as it appears in the image, preserving the original order and structure.\n"
        "2. Do NOT translate, summarize, or interpret the text.\n"
        "3. Do NOT add any explanations, prefixes, or comments.\n"
        "4. If there is no text in the image, return an empty string.\n"
        "5. For multi-language text, transcribe each language as-is.\n"
        "6. Preserve line breaks and spacing where appropriate.\n"
        "\n"
        "Output ONLY the extracted text, nothing else."
    )

    def __init__(self, url: str, key: str, model: str):
        self.base_url = url.rstrip('/') if url else "https://api.openai.com/v1"
        self.key = key
        self.model = model or "gpt-4o-mini"

    def recognize(self, image_bytes: bytes, language: str = "zh") -> str:
        import requests

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # 检测图片格式
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
                        "text": f"Please extract all text from this image. Language hint: {language}"
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
                "max_tokens": 4096,
            },
            headers=headers,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return text.strip()


class QwenVLAdapter(OCRAdapter):
    """Qwen-VL (通义千问视觉模型) OCR

    通过 OpenAI 兼容接口调用 Qwen-VL 进行 OCR。
    """

    _SYSTEM_PROMPT = (
        "你是一个 OCR 文字识别系统。\n\n"
        "任务：从图片中提取所有文字。\n\n"
        "规则：\n"
        "1. 准确提取图片中的所有文字，保持原有顺序和结构。\n"
        "2. 不要翻译、总结或解释文字内容。\n"
        "3. 不要添加任何前缀、解释或评论。\n"
        "4. 如果图片中没有文字，返回空字符串。\n"
        "5. 保持适当的换行和空格。\n"
        "\n"
        "只输出提取的文字，不要输出其他内容。"
    )

    def __init__(self, url: str, key: str, model: str):
        self.base_url = url.rstrip('/') if url else "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.key = key
        self.model = model or "qwen-vl-max"

    def recognize(self, image_bytes: bytes, language: str = "zh") -> str:
        import requests

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
                        "text": "请识别图片中的所有文字。"
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
                "max_tokens": 4096,
            },
            headers=headers,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return text.strip()


class TesseractAdapter(OCRAdapter):
    """Tesseract 本地 OCR 引擎

    离线 OCR，无需网络，但精度较低。
    """

    def __init__(self, language: str = "chi_sim+eng"):
        self._language = language

    def recognize(self, image_bytes: bytes, language: str = "zh") -> str:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise RuntimeError("pytesseract and PIL are required for Tesseract OCR")

        lang_map = {
            "zh": "chi_sim+eng",
            "ch": "chi_sim+eng",
            "zh-CN": "chi_sim+eng",
            "zh-TW": "chi_tra+eng",
            "en": "eng",
            "ja": "jpn+eng",
            "ko": "kor+eng",
        }
        tesseract_lang = lang_map.get(language, self._language)

        image = Image.open(BytesIO(image_bytes))
        text = pytesseract.image_to_string(image, lang=tesseract_lang)
        return text.strip()


def _build_ocr_adapter(cfg: dict) -> Optional[OCRAdapter]:
    """根据配置创建 OCR 适配器"""
    provider = cfg.get('provider', 'openai')

    if provider == 'openai':
        url, key = cfg.get('url', ''), cfg.get('key', '')
        if not key:
            return None
        return OpenAIVisionAdapter(url, key, cfg.get('model', ''))

    elif provider == 'qwen':
        url, key = cfg.get('url', ''), cfg.get('key', '')
        if not key:
            return None
        return QwenVLAdapter(url, key, cfg.get('model', ''))

    elif provider == 'tesseract':
        return TesseractAdapter(cfg.get('language', 'chi_sim+eng'))

    elif provider == 'google':
        log.warning("[ocr] Google Vision OCR not yet implemented")
        return None

    return None


# ── ROS2 Node (订阅模式) ───────────────────────────────────────────────────────

class _OCRNode(Node):
    """订阅 image/jpeg topic，持续进行 OCR 识别"""

    def __init__(self, input_topic: str, adapter: OCRAdapter, language: str = "zh",
                 node_suffix: str = ''):
        node_name = f"ocr_{node_suffix}" if node_suffix else "ocr"
        super().__init__(node_name)

        self._input_topic = input_topic
        self._output_topic = f"{input_topic}/ocr"
        self._adapter = adapter
        self._language = language
        self.state = "idle"

        self._sub = None
        self._pub = self.create_publisher(String, self._output_topic, _LOW_LAT_QOS)

        self._frame_queue: queue.Queue = queue.Queue(maxsize=5)
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_count = 0  # 收到的图片帧计数

        log.info(f"[ocr] node created: subscribing={self._input_topic}, publishing={self._output_topic}")

    def start(self) -> dict:
        if self.state == "running":
            return self._status_dict()

        if not self._adapter:
            raise RuntimeError("OCR adapter not configured")

        self._stop_event.clear()
        self._sub = self.create_subscription(
            CompressedImage, self._input_topic, self._image_cb, _LOW_LAT_QOS
        )
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        self.state = "running"

        log.info(f"[ocr] started: {self._input_topic} → {self._output_topic}")
        return self._status_dict()

    def stop(self) -> dict:
        if self._sub is not None:
            self.destroy_subscription(self._sub)
            self._sub = None

        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3)

        self.state = "idle"
        return {"state": "idle"}

    def _image_cb(self, msg: CompressedImage):
        """接收图片帧，放入队列"""
        self._frame_count += 1
        image_data = bytes(msg.data)
        log.info(f"[ocr] received image frame #{self._frame_count}: "
                 f"size={len(image_data)} bytes, format={msg.format}, "
                 f"topic={self._input_topic}")
        try:
            self._frame_queue.put_nowait((image_data, time.time()))
        except queue.Full:
            log.warning(f"[ocr] frame queue full, dropping old frame (queue_size=5)")
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait((image_data, time.time()))
            except queue.Full:
                pass

    def _worker(self):
        """后台工作线程：从队列取图片进行 OCR"""
        while not self._stop_event.is_set():
            try:
                image_bytes, ts = self._frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                log.info(f"[ocr] worker processing frame: size={len(image_bytes)} bytes, "
                         f"adapter={type(self._adapter).__name__}, language={self._language}")
                text = self._adapter.recognize(image_bytes, self._language)
                log.info(f"[ocr] adapter returned: text_len={len(text)}, text_preview={text[:200]!r}")

                if not text.strip():
                    log.warning(f"[ocr] recognition returned empty text, publishing empty result")
                    # 即使为空也发布，让调用方知道已处理
                    result = {
                        "text": "",
                        "timestamp": ts,
                        "language": self._language,
                    }
                    msg = String()
                    msg.data = json.dumps(result, ensure_ascii=False)
                    self._pub.publish(msg)
                    continue

                result = {
                    "text": text,
                    "timestamp": ts,
                    "language": self._language,
                }

                msg = String()
                msg.data = json.dumps(result, ensure_ascii=False)
                self._pub.publish(msg)

                log.info(f"[ocr] published result to {self._output_topic}: {text[:100]}...")
            except Exception as e:
                log.error(f"[ocr] recognition error: {e}", exc_info=True)
                # 发布错误结果，让调用方知道处理失败
                try:
                    error_result = {
                        "text": "",
                        "error": str(e),
                        "timestamp": ts,
                        "language": self._language,
                    }
                    msg = String()
                    msg.data = json.dumps(error_result, ensure_ascii=False)
                    self._pub.publish(msg)
                    log.info(f"[ocr] published error result to {self._output_topic}")
                except Exception:
                    pass

    def _status_dict(self) -> dict:
        return {
            "state": self.state,
            "topic_in": [{"topic": self._input_topic, "format": "image/jpeg", "desc": "image input"}],
            "topic_out": [{"topic": self._output_topic, "format": "data/json", "desc": "OCR result"}],
        }


# ── Plugin ────────────────────────────────────────────────────────────────────

class OCRPlugin:
    PREFIX = "ocr"

    def __init__(self, plugin_cfg: dict, executor):
        self._adapter = _build_ocr_adapter(plugin_cfg)
        self._language = plugin_cfg.get('language', 'zh')
        self._nodes: dict[str, _OCRNode] = {}
        self._instance_configs: dict[str, dict] = {}
        self._executor = executor

        log.info(f"[ocr] plugin init: provider={plugin_cfg.get('provider')}, "
                 f"language={self._language}, "
                 f"key={'set' if plugin_cfg.get('key') else 'MISSING'}")

        if not self._adapter:
            log.warning("[ocr] adapter not configured (missing key)")

    def get_tools(self) -> list:
        return TOOLS

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action") if name == "ocr" else name
        instance_id = args.get("instance_id", "")

        if action == "info":
            input_topic = args.get("input_topic", "")

            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                return {
                    "name": "OCR", "manufacture": "Embodied", "model": "ocr",
                    "state": node.state,
                    "topic_in": [{"topic": node._input_topic, "format": "image/jpeg", "desc": ""}],
                    "topic_out": [{"topic": node._output_topic, "format": "data/json", "desc": ""}],
                    "desc": "OCR service — extracts text from images",
                }

            if instance_id:
                inferred_out = f"{input_topic}/ocr" if input_topic else ""
                return {
                    "name": "OCR", "manufacture": "Embodied", "model": "ocr",
                    "state": "idle",
                    "topic_in": [{"topic": input_topic, "format": "image/jpeg", "desc": ""}] if input_topic else [],
                    "topic_out": [{"topic": inferred_out, "format": "data/json", "desc": ""}] if inferred_out else [],
                    "desc": "OCR service — extracts text from images",
                }

            # 聚合所有实例信息
            if self._nodes:
                topics_in = [{"topic": n._input_topic, "format": "image/jpeg", "desc": ""} for n in self._nodes.values()]
                topics_out = [{"topic": n._output_topic, "format": "data/json", "desc": ""} for n in self._nodes.values()]
                states = list(set(n.state for n in self._nodes.values()))
                state = "running" if "running" in states else states[0] if states else "idle"
            else:
                inferred_out = f"{input_topic}/ocr" if input_topic else ""
                topics_in = [{"topic": input_topic, "format": "image/jpeg", "desc": ""}]
                topics_out = [{"topic": inferred_out, "format": "data/json", "desc": ""}]
                state = "idle"

            return {
                "name": "OCR", "manufacture": "Embodied", "model": "ocr",
                "state": state,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": "OCR service — extracts text from images",
            }

        elif action == "start":
            input_topic = args.get("input_topic")
            if not input_topic:
                raise ValueError("input_topic is required for start action")

            node_key = instance_id or input_topic

            if node_key not in self._nodes:
                adapter = self._adapter
                language = self._language

                if instance_id and instance_id in self._instance_configs:
                    inst_adapter = _build_ocr_adapter(self._instance_configs[instance_id])
                    if inst_adapter:
                        adapter = inst_adapter
                    inst_lang = self._instance_configs[instance_id].get("language")
                    if inst_lang:
                        language = inst_lang

                node = _OCRNode(
                    input_topic, adapter, language,
                    node_suffix=node_key.replace('/', '_').replace('-', '_')
                )
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
                for key in list(self._nodes.keys()):
                    self._nodes[key].stop()
                    self._executor.remove_node(self._nodes[key])
                    del self._nodes[key]
                return {"state": "idle"}
            return {"state": "idle"}

        elif action == "config":
            cfg = {k: v for k, v in args.items() if k not in ('action', 'instance_id') and v is not None and v != ''}

            if instance_id:
                self._instance_configs[instance_id] = cfg
                if instance_id in self._nodes:
                    self._nodes[instance_id].stop()
                    self._executor.remove_node(self._nodes[instance_id])
                    del self._nodes[instance_id]
                return {"status": "configured", "instance_id": instance_id}
            else:
                self._adapter = _build_ocr_adapter(cfg)
                self._language = cfg.get('language', self._language)
                for key in list(self._nodes.keys()):
                    self._nodes[key].stop()
                    self._executor.remove_node(self._nodes[key])
                    del self._nodes[key]
                return {"status": "configured", "adapter_ok": self._adapter is not None}

        return None
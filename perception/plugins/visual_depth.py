#!/usr/bin/env python3
"""
plugins/visual_depth.py — VideoDepthPerceptionPlugin: monocular depth from a plain RGB camera.

Subscribes to image/jpeg topics, runs a prebuilt YOLO26-depth TensorRT engine,
and publishes two things per frame:

    {input}/depth          image/depth-zlib   for the dashboard's depth renderer
    {input}/depth_summary  data/json          for the agent

Most robots in this fleet carry only an RGB camera — the RealSense on the
Realman RM75 is the exception — so this gives the rest a depth map without new
hardware.

Two constraints worth knowing before changing anything here:

* **640x480 is not a suggestion.** agent-core's DepthZlibRenderer
  (web/js/renderers/camera.js) hardcodes a 640x480 canvas and returns early
  when the decompressed buffer is shorter than 640*480 samples. A depth map
  published at the model's native resolution renders as a blank panel with
  nothing logged anywhere. Everything is resampled to 640x480 before publishing.

* **The depth is relative until it is calibrated.** The released weights predict
  on an unbounded log scale; absolute metric accuracy needs `model.calibrate()`
  against a labelled split from the actual camera. Until that is done the
  numbers are self-consistent but not metres, so every payload carries an
  explicit `scale` field and the default is `"relative"`. An agent that reads a
  relative number as metres and plans around it is exactly the failure this
  guards against.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import zlib
from typing import Optional

import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from utils.ros_lifecycle import dispose_node

log = logging.getLogger(__name__)

# Fixed by the dashboard renderer — see the module docstring.
DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480

_LOW_LAT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=2,
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
        "name": "visual_depth",
        "type": "processor",
        "multiInstance": True,
        "description": "视觉深度 — 用一个普通 RGB 摄像头估计每个像素的距离，输出深度图与左/中/右三区的最近障碍摘要。未标定时是相对尺度，不是米",
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
                "fps":          {"type": "integer", "description": "Max inference frames per second", "default": 2, "scope": "instance"},
                "depth_scale":  {"type": "number",  "description": "Multiplier from model output to metres (1.0 = uncalibrated)", "default": 1.0, "scope": "instance"},
                "calibrated":   {"type": "boolean", "description": "Mark output as metric. Only set this after calibrating against this camera", "default": False, "scope": "instance"},
                "max_depth_m":  {"type": "number",  "description": "Values above this are published as invalid (0)", "default": 20.0, "scope": "instance"},
            },
        },
        "topic_in":  [{"format": "image/jpeg", "desc": "camera image input"}],
        "topic_out": [
            {"format": "image/depth-zlib", "desc": "per-pixel depth map (640x480 uint16)"},
            {"format": "data/json",        "desc": "nearest-obstacle summary by region"},
        ],
    }
]


# ── Depth encoding ───────────────────────────────────────────────────────────

def encode_depth(depth_m: np.ndarray, max_depth_m: float) -> bytes:
    """Renderer contract: zlib of 640x480 little-endian uint16 millimetres.

    Mirrors the driver-side encoder in
    phanthymotus-driver/realman/rm75_6f_v/realsense.py, including its rule that
    0 means "no reading": an out-of-range value must never wrap around into a
    plausible near-field distance, because the consumer cannot tell the
    difference and a wrapped 70 m reading looks like an obstacle at arm's length.
    """
    if depth_m.shape != (DEPTH_HEIGHT, DEPTH_WIDTH):
        raise ValueError(
            f"Expected a {DEPTH_WIDTH}x{DEPTH_HEIGHT} depth map, got "
            f"{depth_m.shape[1]}x{depth_m.shape[0]}"
        )
    mm = np.rint(np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0)
    ceiling = min(65535.0, max(1.0, max_depth_m) * 1000.0)
    mm[(mm < 1) | (mm > ceiling)] = 0
    return zlib.compress(mm.astype("<u2").tobytes(), 1)


def summarize_depth(depth_m: np.ndarray, scale: str, bands: int = 3) -> dict:
    """Nearest valid reading per vertical band, plus the overall range.

    Uses the 5th percentile rather than the raw minimum: a monocular depth map
    routinely has a handful of near-zero outlier pixels at object edges, and a
    summary driven by the single closest pixel reports an obstacle that is not
    there. The percentile is over valid pixels only.
    """
    valid = np.isfinite(depth_m) & (depth_m > 0)
    width = depth_m.shape[1]
    edges = [round(i * width / bands) for i in range(bands + 1)]
    names = ["left", "center", "right"] if bands == 3 else [f"band{i}" for i in range(bands)]

    regions = {}
    for i, name in enumerate(names):
        chunk = depth_m[:, edges[i]:edges[i + 1]]
        chunk_valid = valid[:, edges[i]:edges[i + 1]]
        if not chunk_valid.any():
            regions[name] = None
            continue
        regions[name] = round(float(np.percentile(chunk[chunk_valid], 5)), 3)

    overall = depth_m[valid]
    return {
        "scale": scale,
        "unit": "m" if scale == "metric" else "relative",
        "nearest_by_region": regions,
        "range": [round(float(overall.min()), 3), round(float(overall.max()), 3)] if overall.size else None,
        "valid_fraction": round(float(valid.mean()), 3),
    }


# ── ROS2 Node (one per instance/topic) ───────────────────────────────────────

class _DepthNode(Node):
    """Per-topic depth inference node."""

    def __init__(self, input_topic: str, model, fps: float, depth_scale: float,
                 calibrated: bool, max_depth_m: float, node_suffix: str):
        super().__init__(f"visual_depth_{node_suffix}")
        self._input_topic = input_topic
        self._depth_topic = f"{input_topic}/depth"
        self._summary_topic = f"{input_topic}/depth_summary"
        self._model = model
        self._fps = fps
        self._frame_interval = 1.0 / max(fps, 0.1)
        self._depth_scale = depth_scale
        self._scale_label = "metric" if calibrated else "relative"
        self._max_depth_m = max_depth_m

        self._depth_pub = self.create_publisher(CompressedImage, self._depth_topic, _PUB_QOS)
        self._summary_pub = self.create_publisher(String, self._summary_topic, _PUB_QOS)
        self._sub: Optional[object] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._last_inference_time = 0.0
        self._frame_count = 0
        # See perception/README.md § "Plugin Concurrency" — every dispatch runs
        # on its own ThreadingHTTPServer thread and the canvas issues
        # config→start→stop→start within seconds.
        self._lifecycle_lock = threading.RLock()

    def request_stop(self) -> None:
        """Signal cancellation without taking the lock, so stop can abort a start."""
        self._stop_event.set()

    def start(self) -> dict:
        with self._lifecycle_lock:
            if self._sub is not None:
                return self._state("running")
            self._stop_event.clear()
            self._sub = self.create_subscription(
                CompressedImage, self._input_topic, self._image_cb, _LOW_LAT_QOS
            )
            self._worker = threading.Thread(target=self._inference_worker, daemon=True,
                                            name=f"visual_depth_worker_{self._input_topic}")
            self._worker.start()
            log.info(f"[visual_depth] started: {self._input_topic} → {self._depth_topic}, {self._summary_topic}")
            return self._state("running")

    def stop(self) -> dict:
        # Worker only — the subscription is destroyed by destroy_node() after
        # the node leaves the executor. Destroying it here races the executor's
        # wait list and kills the spin thread with InvalidHandle, taking every
        # other subscription in the process down with it. See plugins/vop.py.
        self._stop_event.set()
        with self._lifecycle_lock:
            if self._worker and self._worker.is_alive():
                self._worker.join(timeout=3.0)
            self._worker = None
            log.info(f"[visual_depth] stopped: {self._input_topic}")
            return self._state("idle")

    def _state(self, state: str) -> dict:
        return {
            "state": state,
            "input": self._input_topic,
            "depth_topic": self._depth_topic,
            "summary_topic": self._summary_topic,
            "scale": self._scale_label,
        }

    def _image_cb(self, msg: CompressedImage):
        now = time.monotonic()
        if now - self._last_inference_time < self._frame_interval:
            return
        self._last_inference_time = now
        # Drop the stale frame rather than queue up: a depth map from two
        # seconds ago is worse than no depth map.
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
        import cv2
        from plugins.vision_runtime import decode_depth

        while not self._stop_event.is_set():
            try:
                jpeg_bytes = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                outputs, meta = self._model.infer(frame)
                depth_m = decode_depth(outputs, meta) * self._depth_scale
                # Resampled here, not by the model: the renderer's canvas is
                # fixed at 640x480 and a mismatch is dropped silently.
                if depth_m.shape != (DEPTH_HEIGHT, DEPTH_WIDTH):
                    depth_m = cv2.resize(depth_m, (DEPTH_WIDTH, DEPTH_HEIGHT),
                                         interpolation=cv2.INTER_NEAREST)
                self._publish(depth_m)
            except Exception as e:
                log.error(f"[visual_depth] inference error: {e}", exc_info=True)

    def _publish(self, depth_m: np.ndarray):
        self._frame_count += 1

        depth_msg = CompressedImage()
        depth_msg.format = "16UC1; compressedDepth zlib"
        depth_msg.data = encode_depth(depth_m, self._max_depth_m)
        self._depth_pub.publish(depth_msg)

        summary = summarize_depth(depth_m, self._scale_label)
        summary["timestamp"] = time.time()
        msg = String()
        msg.data = json.dumps(summary, ensure_ascii=False)
        self._summary_pub.publish(msg)


# ── Plugin class ─────────────────────────────────────────────────────────────

class VideoDepthPerceptionPlugin:
    PREFIX = "visual_depth"
    # `vdp` was the name this shipped under for one release. It said nothing to
    # anyone reading a card on the dashboard or the image's card list on
    # resource-center, so the tool is `visual_depth` now and the old spelling
    # stays as an alias — a card saved under `vdp` keeps dispatching instead of
    # coming back `state: error` after a restart. Same rule as the vop model
    # rename; see perception/README.md § "Vision".
    ALIASES = ("vdp",)

    def __init__(self, plugin_cfg: dict, namespace: str, executor):
        self._namespace = namespace
        self._executor = executor
        self._fps = int(plugin_cfg.get("fps", 2))
        self._depth_scale = float(plugin_cfg.get("depth_scale", 1.0))
        self._calibrated = bool(plugin_cfg.get("calibrated", False))
        self._max_depth_m = float(plugin_cfg.get("max_depth_m", 20.0))
        self._model = None  # lazy load
        self._model_loading = False
        self._model_load_error = None
        self._model_lock = threading.Lock()
        self._nodes: dict[str, _DepthNode] = {}
        self._instance_configs: dict[str, dict] = {}
        # Guards _nodes / _instance_configs; never held across a node start,
        # stop, or a model load.
        self._nodes_lock = threading.RLock()

    def _ensure_model(self):
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return
            from plugins.vision_runtime import VisionEngineSession
            from utils.model_downloader import ensure_depth_model

            model_dir = os.environ.get("DEPTH_MODEL_DIR", "/models/depth")
            paths = ensure_depth_model(model_dir)
            engine = next(p for name, p in paths.items() if name.endswith(".engine"))
            log.info(f"[visual_depth] loading engine: {engine}")
            self._model = VisionEngineSession(engine)
            log.info(f"[visual_depth] engine loaded, input={self._model.input_size}")

    def _start_node(self, node_key: str, input_topic: str):
        """Register before starting, so a concurrent stop can always cancel it."""
        with self._nodes_lock:
            if node_key in self._nodes:
                return
            icfg = self._instance_configs.get(node_key, {})
            node = _DepthNode(
                input_topic, self._model,
                fps=int(icfg.get("fps", self._fps)),
                depth_scale=float(icfg.get("depth_scale", self._depth_scale)),
                calibrated=bool(icfg.get("calibrated", self._calibrated)),
                max_depth_m=float(icfg.get("max_depth_m", self._max_depth_m)),
                node_suffix=node_key.replace("/", "_").replace("-", "_").lstrip("_"),
            )
            self._executor.add_node(node)
            self._nodes[node_key] = node
        node.start()
        log.info(f"[visual_depth] node started (background): {input_topic}")

    def _retire_node(self, node_key: str) -> Optional[dict]:
        with self._nodes_lock:
            node = self._nodes.pop(node_key, None)
        if node is None:
            return None
        node.request_stop()
        result = node.stop()
        # remove-then-destroy: the node must leave the executor before its
        # handles are destroyed, and it must be destroyed rather than merely
        # removed or the publishers and the ROS node name leak.
        dispose_node(self._executor, node, label=f"visual_depth/{node_key}")
        return result

    def get_tools(self) -> list:
        return TOOLS

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action", name)
        instance_id = args.get("instance_id", "")

        if action == "info":
            if self._model_loading:
                return {"name": "VideoDepthPerception", "manufacture": "Embodied",
                        "model": "yolo26n-depth", "state": "loading",
                        "desc": "Loading depth engine..."}
            if self._model_load_error:
                return {"name": "VideoDepthPerception", "manufacture": "Embodied",
                        "model": "yolo26n-depth", "state": "error",
                        "desc": f"Engine load failed: {self._model_load_error}"}

            with self._nodes_lock:
                nodes = dict(self._nodes)
            instances = {
                key: {
                    "input": node._input_topic,
                    "depth_topic": node._depth_topic,
                    "summary_topic": node._summary_topic,
                    "fps": node._fps,
                    "scale": node._scale_label,
                    "frame_count": node._frame_count,
                }
                for key, node in nodes.items()
            }

            input_topic = args.get("input_topic", "")
            if not input_topic:
                topics_list = args.get("input_topics") or []
                if topics_list:
                    input_topic = topics_list[0]
            if instance_id and instance_id in nodes:
                input_topic = nodes[instance_id]._input_topic
            elif not input_topic and nodes:
                input_topic = next(iter(nodes.values()))._input_topic

            topics_in = [{"topic": input_topic, "format": "image/jpeg"}] if input_topic else []
            topics_out = [
                {"topic": f"{input_topic}/depth", "format": "image/depth-zlib"},
                {"topic": f"{input_topic}/depth_summary", "format": "data/json"},
            ] if input_topic else []

            scale = "metric" if self._calibrated else "relative"
            info = {
                "name": "VideoDepthPerception", "manufacture": "Embodied",
                "model": "yolo26n-depth",
                "state": "running" if instances else "idle",
                "scale": scale,
                "instances": instances,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": "Monocular depth estimation (YOLO26-depth, TensorRT)",
            }
            if scale != "metric":
                info["warning"] = (
                    "Depth is uncalibrated and therefore RELATIVE, not metres. "
                    "Comparisons between regions are meaningful; absolute "
                    "distances are not. Calibrate against this camera and set "
                    "calibrated=true before using these values for navigation."
                )
            return info

        elif action == "start":
            input_topic = args.get("input_topic")
            if not input_topic:
                topics_list = args.get("input_topics") or []
                if topics_list:
                    input_topic = topics_list[0]
            if not input_topic:
                raise ValueError("input_topic is required")
            node_key = instance_id or input_topic

            with self._nodes_lock:
                running = self._nodes.get(node_key)
            if running is None:
                if self._model is None:
                    if self._model_loading:
                        return {"state": "loading", "message": "Engine is still loading, please wait..."}
                    if self._model_load_error:
                        return {"state": "error", "message": f"Engine failed to load: {self._model_load_error}"}

                    def _bg_start():
                        self._model_loading = True
                        self._model_load_error = None
                        try:
                            self._ensure_model()
                            self._model_loading = False
                            self._start_node(node_key, input_topic)
                        except Exception as e:
                            self._model_loading = False
                            self._model_load_error = str(e)
                            log.error(f"[visual_depth] engine load failed: {e}", exc_info=True)

                    threading.Thread(target=_bg_start, daemon=True, name="visual_depth_model_load").start()
                    return {"state": "loading", "input": input_topic,
                            "message": "Engine loading in background, will start automatically"}
                self._start_node(node_key, input_topic)
                with self._nodes_lock:
                    running = self._nodes.get(node_key)
                if running is None:
                    return {"state": "idle", "input": input_topic}
            return running.start()

        elif action == "stop":
            if instance_id:
                result = self._retire_node(instance_id)
                return result if result is not None else {"state": "idle"}
            with self._nodes_lock:
                keys = list(self._nodes.keys())
            results = [key for key in keys if self._retire_node(key) is not None]
            return {"state": "idle", "stopped_instances": results} if results else {"state": "idle"}

        elif action == "config":
            cfg = {k: v for k, v in args.items()
                   if k not in ("action", "instance_id") and v is not None and v != ""}
            if instance_id:
                with self._nodes_lock:
                    self._instance_configs[instance_id] = cfg
                    running = instance_id in self._nodes
                if running:
                    self._retire_node(instance_id)
                return {"status": "configured", "instance_id": instance_id, "config": cfg}
            if "fps" in cfg:
                self._fps = int(cfg["fps"])
            if "depth_scale" in cfg:
                self._depth_scale = float(cfg["depth_scale"])
            if "calibrated" in cfg:
                self._calibrated = bool(cfg["calibrated"])
            if "max_depth_m" in cfg:
                self._max_depth_m = float(cfg["max_depth_m"])
            return {"status": "configured", "config": cfg}

        return None

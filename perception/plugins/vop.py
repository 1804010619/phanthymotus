#!/usr/bin/env python3
"""
plugins/vop.py — VideoObjectPerceptionPlugin: YOLOE-26 open-vocabulary object detection.

Subscribes to image/jpeg topics, runs a prebuilt TensorRT engine, publishes
detected objects with center-relative normalized coordinates. Supports
multi-instance (one instance per input topic).

Two things changed together here, and the second is a consequence of the first:

* **TensorRT, not PyTorch.** The previous version loaded a `.pt` and ran
  ultralytics in eager mode. Measured on an Orin NX 8GB, eager costs ~39 ms per
  frame against ~5 ms for the same network as a TensorRT fp16 engine — the gap
  is kernel-launch overhead, not arithmetic, which is why it barely moved with
  model size. The engine is built offline and shipped as a pinned bundle
  (`utils.model_downloader.ensure_vop_model`), the same way OCR ships its own.

* **The vocabulary is frozen.** Ultralytics bakes the open-vocabulary class list
  into the weights at export time; on an exported model `set_classes()` raises.
  So the runtime `set_classes` action and the per-instance `classes` config that
  this plugin used to offer cannot work against an engine. Both now fail with an
  explicit message naming the baked vocabulary rather than silently doing
  nothing — a card that quietly stops honouring its configured classes is far
  worse than one that says why.

The vocabulary travels with the engine as `vocab.json`; it is never hardcoded
here, so what the plugin reports and what the engine can actually detect cannot
drift apart.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

log = logging.getLogger(__name__)

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

DEFAULT_MODEL = "yoloe-26s-seg"

# Names that already exist in deployed canvas cards and yaml. A card saved when
# this plugin ran YOLOv8-World must keep loading after an upgrade: the engine it
# gets is the current one, but the name it asks for still resolves. Dropping the
# alias would turn every such card into `state: error` on its next restart.
_MODEL_ALIASES = {
    "yolov8s-worldv2": DEFAULT_MODEL,
    "yolov8s-world": DEFAULT_MODEL,
    "yoloe-26s": DEFAULT_MODEL,
}


def canonical_model_name(name: str) -> str:
    """Map a configured model name onto the one model this build ships."""
    base = (name or "").strip()
    if base.endswith(".pt") or base.endswith(".engine"):
        base = base.rsplit(".", 1)[0]
    return _MODEL_ALIASES.get(base, base or DEFAULT_MODEL)

TOOLS = [
    {
        "name": "vop",
        "type": "processor",
        "multiInstance": True,
        # `set_classes` is deliberately absent from the enum: the engine's
        # vocabulary is frozen at export time, so advertising the action would
        # promise the agent a capability that always fails. dispatch() still
        # answers it — with an explanation — because deployed cards and older
        # conversations can still send it.
        "description": "Video Object Perception — detect objects in camera feed (fixed open-vocabulary set, see info)",
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
                "confidence": {"type": "number", "description": "Detection confidence threshold (0-1)", "default": 0.3, "scope": "instance"},
                "fps":        {"type": "integer", "description": "Max inference frames per second", "default": 5, "scope": "instance"},
            },
        },
        "topic_in":  [{"format": "image/jpeg", "desc": "camera image input"}],
        "topic_out": [{"format": "data/json",  "desc": "detected objects with positions"}],
    }
]


# ── ROS2 Node (one per instance/topic) ────────────────────────────────────────

class _VOPNode(Node):
    """Per-topic YOLO inference node."""

    def __init__(self, input_topic: str, model, confidence: float, fps: float,
                 node_suffix: str, vocabulary: Optional[list] = None):
        super().__init__(f"vop_{node_suffix}")
        self._input_topic = input_topic
        self._output_topic = f"{input_topic}/objects"
        self._model = model
        self._vocabulary = list(vocabulary or [])
        self._confidence = confidence
        self._fps = fps
        self._frame_interval = 1.0 / max(fps, 0.1)

        self._pub = self.create_publisher(String, self._output_topic, _PUB_QOS)
        self._sub: Optional[object] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._last_inference_time = 0.0
        self._detect_count = 0
        # Serializes start/stop on this node. The plugin calls both from HTTP
        # handler threads, and the canvas routinely does config→start→stop→start
        # within seconds; without this two starts can both pass the running
        # check and the second overwrites the first's subscription, orphaning a
        # live worker nothing can stop. Same rule as plugins/ocr.py.
        self._lifecycle_lock = threading.RLock()

    def request_stop(self) -> None:
        """Signal the worker to wind down without taking the lifecycle lock.

        stop() has to be able to cancel a start() that is still in progress, so
        the cancellation flag must be settable while that start holds the lock.
        """
        self._stop_event.set()

    def start(self) -> dict:
        with self._lifecycle_lock:
            if self._sub is not None:
                return {"state": "running", "input": self._input_topic, "output": self._output_topic}
            self._stop_event.clear()
            self._sub = self.create_subscription(
                CompressedImage, self._input_topic, self._image_cb, _LOW_LAT_QOS
            )
            self._worker = threading.Thread(target=self._inference_worker, daemon=True,
                                            name=f"vop_worker_{self._input_topic}")
            self._worker.start()
            log.info(f"[vop] started: {self._input_topic} → {self._output_topic}")
            return {"state": "running", "input": self._input_topic, "output": self._output_topic}

    def stop(self) -> dict:
        # Flag first, lock second: a start() holding the lock will see the flag
        # as soon as it releases, instead of this call queueing behind it.
        self._stop_event.set()
        with self._lifecycle_lock:
            if self._sub is not None:
                self.destroy_subscription(self._sub)
                self._sub = None
            if self._worker and self._worker.is_alive():
                self._worker.join(timeout=3.0)
            self._worker = None
            log.info(f"[vop] stopped: {self._input_topic}")
            return {"state": "idle", "input": self._input_topic}

    def _image_cb(self, msg: CompressedImage):
        now = time.monotonic()
        if now - self._last_inference_time < self._frame_interval:
            return
        self._last_inference_time = now
        # Drop old frame if queue full (no backpressure)
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
        from plugins.vision_runtime import decode_detections

        while not self._stop_event.is_set():
            try:
                jpeg_bytes = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                frame = cv2.imdecode(
                    np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR
                )
                if frame is None:
                    continue
                outputs, meta = self._model.infer(frame)
                boxes, scores, classes = decode_detections(
                    outputs[0], meta, self._confidence
                )
                self._publish_objects(self._extract_objects(boxes, scores, classes, frame.shape))
            except Exception as e:
                log.error(f"[vop] inference error: {e}", exc_info=True)

    def _class_name(self, cls_id: int) -> str:
        """Resolve a class id through the engine's frozen vocabulary.

        vocab.json is written by the exporter from the same list, in the same
        order, that was baked into the weights. An id outside it stays numeric
        rather than becoming a confidently wrong word.
        """
        if 0 <= cls_id < len(self._vocabulary):
            return self._vocabulary[cls_id]
        return str(cls_id)

    def _extract_objects(self, boxes, scores, classes, shape) -> list:
        H, W = shape[:2]
        half_w, half_h = W / 2.0, H / 2.0
        objects = []
        for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, classes):
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            objects.append({
                "name": self._class_name(int(cls_id)),
                "position": [
                    round(float((cx - half_w) / half_w), 3),
                    round(float((cy - half_h) / half_h), 3),
                ],
                "confidence": round(float(score), 2),
            })
        return objects

    def _publish_objects(self, objects: list):
        self._detect_count += 1
        msg = String()
        msg.data = json.dumps({
            "timestamp": time.time(),
            "objects": objects,
        }, ensure_ascii=False)
        self._pub.publish(msg)


# ── Plugin class ──────────────────────────────────────────────────────────────

class VideoObjectPerceptionPlugin:
    PREFIX = "vop"

    def __init__(self, plugin_cfg: dict, namespace: str, executor):
        self._namespace = namespace
        self._executor = executor
        self._confidence = float(plugin_cfg.get("confidence", 0.3))
        self._fps = int(plugin_cfg.get("fps", 5))
        self._model_name = canonical_model_name(plugin_cfg.get("model", DEFAULT_MODEL))
        self._vocabulary: list[str] = []      # filled from the bundle's vocab.json
        self._model = None  # lazy load
        self._model_loading = False
        self._model_load_error = None
        self._model_lock = threading.Lock()
        self._nodes: dict[str, _VOPNode] = {}
        self._instance_configs: dict[str, dict] = {}  # per-instance config overrides
        # Guards _nodes and _instance_configs. Every dispatch() runs on its own
        # ThreadingHTTPServer thread, so an unguarded read-modify-write of
        # _nodes can leave a started node unreachable — see
        # perception/README.md § "Plugin Concurrency". Never held across
        # node.start()/stop() or a model load.
        self._nodes_lock = threading.RLock()
        # A configured `classes` list cannot be honoured any more (the engine's
        # vocabulary is frozen). Remember that it was asked for, so info() and
        # the next dispatch can say so instead of pretending it took effect.
        self._rejected_classes: list[str] = list(plugin_cfg.get("classes") or [])
        if self._rejected_classes:
            log.warning(
                "[vop] ignoring configured classes %s — this build runs a "
                "TensorRT engine with a frozen vocabulary; see info action",
                self._rejected_classes,
            )

    def _frozen_vocab_error(self, requested) -> str:
        """The one explanation both rejection paths give."""
        head = ", ".join(self._vocabulary[:12])
        more = f" (+{len(self._vocabulary) - 12} more)" if len(self._vocabulary) > 12 else ""
        return (
            f"This build runs a prebuilt TensorRT engine whose open-vocabulary "
            f"class list was frozen when the engine was exported, so classes "
            f"cannot be changed at runtime. Requested: {list(requested)}. "
            f"The engine detects {len(self._vocabulary)} classes: {head}{more}. "
            f"To detect something outside that list, rebuild the engine with "
            f"tools/export_vision_engines.py and republish the bundle."
        )

    def _ensure_model(self):
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return

            # No YOLO_CONFIG_DIR / TORCH_HOME setup any more: ultralytics is not
            # in this path at all — the engine is run directly through
            # utils.tensorrt_runtime. Only the cv2 repair below still matters,
            # because the decode and letterbox use it.

            # Fix broken system cv2 on Jetson (circular import in mat_wrapper)
            # and patch missing imshow for headless environments
            try:
                import cv2
                # Test if cv2 is functional
                _ = cv2.IMREAD_COLOR
            except (ImportError, AttributeError):
                import importlib.util, sys as _sys
                import glob as _glob
                # Find the .so directly
                _so_candidates = _glob.glob("/usr/lib/python*/dist-packages/cv2/python-*/cv2.cpython-*.so")
                if _so_candidates:
                    _spec = importlib.util.spec_from_file_location("cv2", _so_candidates[0])
                    _mod = importlib.util.module_from_spec(_spec)
                    _spec.loader.exec_module(_mod)
                    _sys.modules["cv2"] = _mod
                    import cv2
                else:
                    import cv2  # let it fail naturally

            if not hasattr(cv2, 'imshow'):
                cv2.imshow = lambda *a, **k: None
                cv2.waitKey = lambda *a, **k: 0
                cv2.destroyAllWindows = lambda *a, **k: None

            from plugins.vision_runtime import VisionEngineSession

            engine_path, vocab = self._resolve_engine()
            log.info(f"[vop] loading engine: {engine_path}")
            self._model = VisionEngineSession(engine_path)
            # The engine's own metadata wins over the bundled vocab.json: it was
            # written by the export that baked the classes into the weights, so
            # it cannot be stale or out of order. vocab.json only covers an
            # engine built without names.
            self._vocabulary = self._model.class_names() or vocab
            if not self._vocabulary:
                log.warning("[vop] engine carries no class names and no vocab.json "
                            "was readable — detections will be labelled by index")
            log.info(f"[vop] engine loaded: {self._model_name} "
                     f"input={self._model.input_size}, "
                     f"{len(self._vocabulary)} classes")

    def _resolve_engine(self) -> tuple[str, list[str]]:
        """Return (engine path, frozen vocabulary) for the configured model.

        A path given in config wins, so a dev box can point at a locally built
        engine; otherwise the pinned bundle for this machine's TensorRT is
        fetched. There is no `.pt` fallback: silently dropping back to PyTorch
        would cost 8x per frame and look like nothing was wrong.
        """
        configured = (self._model_name or "").strip()
        if configured.endswith(".engine") and os.path.isfile(configured):
            return configured, self._read_vocab(
                os.path.join(os.path.dirname(configured), "vocab.json")
            )

        from utils.model_downloader import ensure_vop_model

        model_dir = os.environ.get("VOP_MODEL_DIR", "/models/vop")
        paths = ensure_vop_model(model_dir)
        engine = next(p for name, p in paths.items() if name.endswith(".engine"))
        return engine, self._read_vocab(paths.get("vocab.json"))

    @staticmethod
    def _read_vocab(path: Optional[str]) -> list[str]:
        """Read the class list the engine was exported with.

        Missing or unreadable is not fatal: the engine still detects exactly
        what it was built for, and the only thing lost is this plugin's ability
        to name those classes in info() and in rejection messages. Detection
        results carry their own names from the engine metadata.
        """
        if not path or not os.path.isfile(path):
            log.warning("[vop] no vocab.json beside the engine; class list unknown")
            return []
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError) as exc:
            log.warning(f"[vop] unreadable vocab.json ({exc}); class list unknown")
            return []
        names = data.get("classes") if isinstance(data, dict) else data
        return [str(n) for n in names] if isinstance(names, list) else []

    def _start_node(self, node_key: str, input_topic: str):
        """Create and start a VOPNode for the given topic.

        Registers the node before starting it so a concurrent stop can always
        find and cancel it; the lock covers only the registration, never the
        start itself, so a stop is not queued behind a start it is trying to
        abort (perception/README.md § "Plugin Concurrency").
        """
        with self._nodes_lock:
            if node_key in self._nodes:
                return
            icfg = self._instance_configs.get(node_key, {})
            confidence = float(icfg.get("confidence", self._confidence))
            fps = int(icfg.get("fps", self._fps))
            suffix = node_key.replace("/", "_").replace("-", "_").lstrip("_")
            node = _VOPNode(input_topic, self._model, confidence, fps,
                            node_suffix=suffix, vocabulary=self._vocabulary)
            self._executor.add_node(node)
            self._nodes[node_key] = node
        node.start()
        log.info(f"[vop] node started (background): {input_topic}")

    def _retire_node(self, node_key: str) -> Optional[dict]:
        """Stop, unregister and destroy one node. Returns its stop() result."""
        with self._nodes_lock:
            node = self._nodes.pop(node_key, None)
        if node is None:
            return None
        node.request_stop()
        result = node.stop()
        self._executor.remove_node(node)
        # destroy_node(), not just remove_node(): otherwise the publisher and
        # the ROS node name leak, and the next start on the same topic trips
        # rclpy's "Publisher already registered for provided node name".
        node.destroy_node()
        return result

    def get_tools(self) -> list:
        return TOOLS

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action", name)
        instance_id = args.get("instance_id", "")

        if action == "info":
            # Report loading/error state
            if self._model_loading:
                return {
                    "name": "VideoObjectPerception", "manufacture": "Embodied", "model": self._model_name,
                    "state": "loading",
                    "desc": "Loading YOLO model...",
                }
            if self._model_load_error:
                return {
                    "name": "VideoObjectPerception", "manufacture": "Embodied", "model": self._model_name,
                    "state": "error",
                    "desc": f"Model load failed: {self._model_load_error}",
                }
            with self._nodes_lock:
                nodes = dict(self._nodes)
            instances = {}
            for key, node in nodes.items():
                instances[key] = {
                    "input": node._input_topic,
                    "output": node._output_topic,
                    "confidence": node._confidence,
                    "fps": node._fps,
                    "detect_count": node._detect_count,
                }
            # Determine topic info: from running instance, args, or empty
            input_topic = args.get("input_topic", "")
            if not input_topic:
                topics_list = args.get("input_topics") or []
                if topics_list:
                    input_topic = topics_list[0]
            # If instance_id specified and running, use its topics
            if instance_id and instance_id in nodes:
                input_topic = nodes[instance_id]._input_topic
            # If no explicit topic but there are running instances, use first one
            elif not input_topic and nodes:
                input_topic = next(iter(nodes.values()))._input_topic
            topics_in = [{"topic": input_topic, "format": "image/jpeg"}] if input_topic else []
            topics_out = [{"topic": f"{input_topic}/objects", "format": "data/json"}] if input_topic else []
            state = "running" if instances else "idle"
            info = {
                "name": "VideoObjectPerception", "manufacture": "Embodied", "model": self._model_name,
                "state": state,
                "vocabulary_frozen": True,
                "total_classes": len(self._vocabulary),
                "classes": self._vocabulary,
                "instances": instances,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": "YOLOE-26 open-vocabulary object detection (TensorRT, fixed class list)",
            }
            # Surface a `classes` that yaml asked for and this build cannot
            # honour. Without this the card looks perfectly healthy while
            # silently detecting a different set than its config states.
            if self._rejected_classes:
                info["ignored_config_classes"] = self._rejected_classes
                info["warning"] = self._frozen_vocab_error(self._rejected_classes)
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
                        return {"state": "loading", "message": "Model is still loading, please wait..."}
                    if self._model_load_error:
                        return {"state": "error", "message": f"Model failed to load: {self._model_load_error}"}
                    # Model not loaded yet — start loading in background
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
                            log.error(f"[vop] model load failed: {e}", exc_info=True)
                    threading.Thread(target=_bg_start, daemon=True, name="vop_model_load").start()
                    return {"state": "loading", "input": input_topic, "output": f"{input_topic}/objects",
                            "message": "Model loading in background, will start automatically"}
                self._start_node(node_key, input_topic)
                with self._nodes_lock:
                    running = self._nodes.get(node_key)
                if running is None:
                    # A concurrent stop retired it between start and lookup.
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

        elif action == "set_classes":
            # Kept reachable although it is no longer advertised in the tool
            # schema: deployed cards and in-flight conversations can still send
            # it, and an explicit refusal is worth far more than "unknown
            # action" or a success that changes nothing.
            raise ValueError(self._frozen_vocab_error(args.get("classes") or []))

        elif action == "config":
            cfg = {k: v for k, v in args.items() if k not in ('action', 'instance_id') and v is not None and v != ''}
            # An old card still carrying `classes` reaches here. Refuse the
            # whole config call rather than applying confidence/fps and
            # dropping classes on the floor: a half-applied config is the
            # failure mode this is meant to prevent.
            if cfg.get("classes"):
                self._rejected_classes = list(cfg["classes"])
                raise ValueError(self._frozen_vocab_error(cfg["classes"]))
            if instance_id:
                with self._nodes_lock:
                    self._instance_configs[instance_id] = cfg
                    running = instance_id in self._nodes
                # If instance is running, retire it; the next start picks up
                # the new config.
                if running:
                    self._retire_node(instance_id)
                return {"status": "configured", "instance_id": instance_id, "config": cfg}
            else:
                # Update global defaults
                if "confidence" in cfg:
                    self._confidence = float(cfg["confidence"])
                if "fps" in cfg:
                    self._fps = int(cfg["fps"])
                return {"status": "configured", "config": cfg}

        return None

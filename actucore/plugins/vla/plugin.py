"""The VLA card — one card, a pluggable provider, and whatever it is wired to.

Two axes, kept orthogonal on purpose, because changing the model and changing
the robot are unrelated events and a design that couples them needs an edit per
(model × robot) pair:

  **provider** — which model, and where it runs. Discovered from
  `providers/`, never enumerated here (see providers/__init__.py).

  **embodiment** — which robot. Not configured at all: the card drives whatever
  is connected to its output on the canvas. agent-core asks that card for its
  action space and hands it here as `control_interface` on `start`, the same way
  `canvas_binding.py` decides which MCPs the agent may reach. There is no
  `embodiment: "g1_dex1"` string to get wrong, and no URDF to misread as an
  action interface.

The card's own job is small: negotiate once, then pace a chunk out one command
at a time onto a `control/*` topic. Everything that keeps an arm safe lives in
the driver — `ControlSink` checks each command, holds on silence, and stops on
force. That split is deliberate: this process can crash, be OOM-killed, or lose
its network, and none of those must be what stops the robot.

Design: phanthymotus/docs/vla-integration.md
"""

from __future__ import annotations

import json
import logging
import threading
import time

from . import negotiate
from .message import build as build_message
from .providers import discover

log = logging.getLogger(__name__)

DEFAULT_TOPIC = "/actucore/vla/cmd"
# Which topic format to publish. Only the ones a driver can consume today; the
# card refuses anything else rather than publishing into a void.
FORMATS = {"joint_position": "control/joint",
           "joint_velocity": "control/joint-velocity",
           "joint_torque": "control/joint-torque",
           "twist": "control/velocity",
           "eef_pose": "control/waypoint"}


class VLAPlugin:
    PREFIX = "vla"        # no underscore — dispatch routes on partition("_")

    def __init__(self, plugin_cfg: dict, executor, namespace: str = ""):
        self._cfg = dict(plugin_cfg or {})
        self._executor = executor
        self._namespace = (namespace or "").strip("/")
        self._topic = self._cfg.get("topic") or DEFAULT_TOPIC

        # start/stop/config arrive on separate threads (ThreadingHTTPServer).
        # The lock guards bookkeeping only — never a provider construction, an
        # inference, or a node start/stop, or a stop would queue behind the
        # start it is meant to cancel.
        self._lock = threading.RLock()
        self._provider = None
        self._descriptor: dict = {}
        self._capabilities: dict = {}
        self._node = None
        self._publisher = None
        self._timer = None
        self._running = False
        self._task = ""
        self._rate_hz = 30.0
        self._ttl_ms = 100
        self._seq = 0
        self._chunk: list = []
        self._chunk_index = 0
        self._chunk_obs_ms = 0
        self._last_error = ""
        self._published = 0

    # ── tools ────────────────────────────────────────────────────────────────

    def get_tools(self) -> list:
        available = sorted(discover())
        # Names of the checkpoints staged for local providers. Offered as an
        # enum so the operator picks one rather than typing a name that has to
        # match a config key exactly — and left absent for a remote provider,
        # whose model names live on the server and are not ours to enumerate.
        local_models = sorted(self._cfg.get("models") or {})
        default_model = self._cfg.get("model_name") or (
            local_models[0] if local_models else "")
        return [{
            "name": "vla",
            "type": "processor",
            "multiInstance": False,
            "description": "语言指令驱动的端到端策略；把动作流发到连上的驱动命令卡片",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string",
                               "enum": ["start", "stop", "info", "config"]},
                    "task": {"type": "string", "description": "自然语言指令"},
                    # Handed over by agent-core from the card wired downstream.
                    # Not operator-editable: it is a reading of the other card,
                    # and a hand-typed copy is a copy that goes stale.
                    "control_interface": {"type": "object"},
                },
                "required": ["action"],
                # Long-running and cancellable, but never "complete": a policy
                # runs until stopped. Declaring x-completion would hold an ACP
                # pending open for the life of the card and block every other
                # actuator behind the barrier.
                "x-hooks": {"on_interrupt_all": {"action": "stop"},
                            "on_interrupt_motion": {"action": "stop"}},
                "x-is-dangerous": True,
                "x-resource": self._resources(),
            },
            "configSchema": {
                "type": "object",
                "properties": {
                    # Built from what was discovered, not written down — adding
                    # a provider file is the whole of adding a provider.
                    "provider": {"type": "string", "enum": available,
                                 "default": "mock" if "mock" in available else
                                 (available[0] if available else ""),
                                 "scope": "shared"},
                    # Which checkpoint. For a local provider this selects one
                    # of the staged models under `models:`; for vla_cloud it is
                    # the name the server knows it by. One field either way —
                    # the question "which model" is the same question, and
                    # splitting it would give the same thing two names.
                    #
                    # The published checkpoints keep their upstream names
                    # (`smolvla_base`); a version fine-tuned for a particular
                    # robot gets a name that says so (`smolvla_tianyi`,
                    # `smolvla_q5`), because the action space it fits is the
                    # thing an operator has to get right.
                    # No `description`/`title`: the form renders
                    # `title || description || key` as the label, with no
                    # separate hint element, so prose here would replace the
                    # field name rather than accompany it. The explanation
                    # belongs in config.yaml where it can be read in full.
                    "model_name": {"type": "string", "default": default_model,
                                   "scope": "shared",
                                   **({"enum": local_models} if local_models else {})},
                    # Only vla_cloud has anywhere to send a request. Hiding
                    # these for a local provider is not cosmetic: a filled-in
                    # endpoint beside `provider: smolvla` reads as configured
                    # and is ignored, which is the kind of thing an operator
                    # spends an afternoon on.
                    "endpoint": {"type": "string", "scope": "shared",
                                 "x-show-when": {"provider": "vla_cloud"}},
                    "api_key": {"type": "string", "scope": "shared",
                                "x-show-when": {"provider": "vla_cloud"}},
                    "timeout_ms": {"type": "number", "default": 500,
                                   "scope": "shared",
                                   "x-show-when": {"provider": "vla_cloud"}},
                    "topic": {"type": "string", "default": DEFAULT_TOPIC,
                              "scope": "instance"},
                    "rate_hz": {"type": "number", "scope": "instance"},
                    "priority": {"type": "integer", "default": 50,
                                 "scope": "instance"},
                },
                "required": [],
            },
            "topic_out": [{"format": self._format(), "desc": "motus.control/1 命令流"}],
        }]

    def dispatch(self, name: str, args: dict):
        action = args.get("action") or name
        if action == "start":
            return self._start(args)
        if action == "stop":
            return self._stop()
        if action == "info":
            return self._info()
        if action == "config":
            return self._config(args)
        return None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        """Bundle lifecycle. Deliberately inert — same rule as the driver card.

        A policy must not resume because a container restarted. It starts when
        someone wires it and asks.
        """

    def stop(self):
        self._stop()

    # ── actions ──────────────────────────────────────────────────────────────

    def _start(self, args: dict):
        descriptor = args.get("control_interface") or {}
        if not descriptor:
            return self._error(
                "没有拿到下游动作空间 —— 请把本卡片的输出连到一张驱动命令卡片"
                "（control/* 端口），agent-core 会在启动时把它的 descriptor 传过来")

        provider_name = self._cfg.get("provider") or "mock"
        providers = discover()
        if provider_name not in providers:
            detail = discover.errors.get(provider_name)
            return self._error(
                f"provider {provider_name!r} 不可用"
                + (f"：{detail}" if detail else f"，可用的有 {sorted(providers)}"))

        try:
            provider = providers[provider_name](descriptor, self._cfg)
            capabilities = provider.capabilities()
        except Exception as error:      # noqa: BLE001
            return self._error(f"provider {provider_name} 初始化失败：{error}")

        problems = negotiate.check(capabilities, descriptor)
        if problems:
            self._close(provider)
            # Refused rather than started and left to fail per command: at
            # 30 Hz the second outcome is a stopped robot with no reason given.
            return self._error("模型与下游动作空间不匹配：" + "；".join(problems))

        mode = descriptor.get("mode")
        if mode not in FORMATS:
            self._close(provider)
            return self._error(f"下游 mode {mode!r} 还没有对应的 topic 格式")

        rate = negotiate.effective_rate(capabilities, descriptor,
                                        self._cfg.get("rate_hz"))
        with self._lock:
            if self._running:
                return self._error("already running")
            self._provider = provider
            self._descriptor = descriptor
            self._capabilities = capabilities
            self._task = args.get("task") or self._cfg.get("task") or ""
            self._rate_hz = rate
            self._ttl_ms = negotiate.ttl_ms(rate, descriptor)
            self._seq = 0
            self._chunk = []
            self._chunk_index = 0
            self._published = 0
            self._last_error = ""
            self._running = True

        try:
            self._open_publisher(mode)
        except Exception as error:      # noqa: BLE001
            with self._lock:
                self._running = False
                self._provider = None
            self._close(provider)
            return self._error(f"publisher 启动失败：{error}")

        log.info("vla started: provider=%s topic=%s %.1f Hz ttl=%d ms task=%r",
                 provider_name, self._topic, rate, self._ttl_ms, self._task)
        result = {"state": self._state(), "topic": self._topic, "rate_hz": rate,
                  "ttl_ms": self._ttl_ms, "provider": provider_name,
                  "capabilities": capabilities}
        if result["state"] == "loading":
            # agent-core keeps the card visibly starting and polls info() until
            # it settles. Reporting ready here is how an operator ends up with a
            # card that claims ready seconds before it can act.
            result["message"] = "模型加载中，就绪后自动开始发布"
        return result

    def _stop(self):
        with self._lock:
            node, self._node = self._node, None
            timer, self._timer = self._timer, None
            provider, self._provider = self._provider, None
            was_running, self._running = self._running, False
            self._publisher = None

        if timer is not None and node is not None:
            try:
                node.destroy_timer(timer)
            except Exception:           # noqa: BLE001 — teardown is best effort
                pass
        if node is not None:
            try:
                if self._executor is not None:
                    self._executor.remove_node(node)
            finally:
                # destroy_node, not only remove_node: otherwise the publisher
                # and the ROS node name leak and a restart collides with itself.
                node.destroy_node()
        self._close(provider)
        if was_running:
            log.info("vla stopped (%s)", self._topic)
        # Stopping publishing is not stopping the robot. The driver's watchdog
        # is what brings it to rest, which is the point of it being there.
        return {"state": "idle"}

    def _info(self):
        with self._lock:
            return {
                "state": self._state(),
                "topic": self._topic,
                "format": self._format(),
                "provider": self._cfg.get("provider") or "mock",
                "providers_available": sorted(discover()),
                "providers_unavailable": dict(discover.errors),
                "task": self._task,
                "rate_hz": self._rate_hz,
                "ttl_ms": self._ttl_ms,
                "published": self._published,
                "capabilities": dict(self._capabilities),
                "control_interface": dict(self._descriptor),
                "error": self._last_error,
            }

    def _config(self, args: dict):
        for key, value in (args or {}).items():
            if key in ("action", "instance_id"):
                continue
            self._cfg[key] = value
        if "topic" in (args or {}):
            self._topic = args["topic"] or DEFAULT_TOPIC
        return {"state": "running" if self._running else "idle", "config": dict(self._cfg)}

    # ── publishing ───────────────────────────────────────────────────────────

    def _open_publisher(self, mode: str):
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            # One deep: a queued command is a stale command, and the receiver
            # would drop it on ttl anyway.
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        node = Node("actucore_vla")
        publisher = node.create_publisher(String, self._topic, qos)
        timer = node.create_timer(1.0 / self._rate_hz, self._tick)
        if self._executor is not None:
            self._executor.add_node(node)
        with self._lock:
            self._node, self._publisher, self._timer = node, publisher, timer

    def _tick(self):
        publisher = self._publisher
        if publisher is None or not self._running:
            return
        provider = self._provider
        # A provider that loads its weights in the background is not an error
        # while it does so — it simply has nothing to say yet. Publishing
        # nothing keeps the receiver's watchdog holding the arm, which is the
        # right state for "no policy is driving".
        if provider is not None and not provider.health():
            return
        try:
            message = self.next_command()
        except Exception as error:      # noqa: BLE001
            with self._lock:
                self._last_error = f"{type(error).__name__}: {error}"
            log.warning("vla inference failed: %s", error)
            # Publishing nothing is the correct failure: the receiver's watchdog
            # brings the arm to rest, whereas a repeated last command would keep
            # driving it on a policy that is no longer running.
            return
        from std_msgs.msg import String
        payload = String()
        payload.data = json.dumps(message, ensure_ascii=False)
        publisher.publish(payload)
        with self._lock:
            self._published += 1

    def next_command(self) -> dict:
        """One command, refilling the chunk when it runs out.

        Public and ROS-free so the whole of the wire format — sequence numbers,
        the two timestamps, chunk indices — is testable without a node. `_tick`
        is only this plus a publish.
        """
        if self._chunk_index >= len(self._chunk):
            # The observation timestamp belongs to the moment the chunk was
            # computed, and every command paced out of it carries that same
            # value while its own stamp advances. That is what makes an ageing
            # chunk visible downstream instead of looking perpetually fresh.
            self._chunk_obs_ms = int(time.time() * 1000)
            self._chunk = list(self._provider.infer(None) or [])
            self._chunk_index = 0
            if not self._chunk:
                raise RuntimeError("provider returned an empty chunk")

        values = self._chunk[self._chunk_index]
        index, size = self._chunk_index, len(self._chunk)
        self._chunk_index += 1
        self._seq += 1
        return build_message(
            seq=self._seq,
            values=values,
            mode=self._descriptor.get("mode", "joint_position"),
            dof=int(self._descriptor.get("dof") or len(values)),
            source=f"mcp__actucore__{self.PREFIX}",
            stamp_ms=int(time.time() * 1000),
            obs_stamp_ms=self._chunk_obs_ms,
            ttl_ms=self._ttl_ms,
            priority=int(self._cfg.get("priority", 50)),
            chunk_index=index,
            chunk_size=size,
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _state(self) -> str:
        """idle | loading | running.

        `loading` is a real state, not a nicety: a local provider reads its
        checkpoint's config in milliseconds and its weights in seconds, and
        agent-core has a watcher for exactly this (`api/config.py`
        `_settle_loading_item`). Skipping it would report ready while the card
        still cannot produce a command.
        """
        if not self._running:
            return "idle"
        provider = self._provider
        if provider is not None and not provider.health():
            return "loading"
        return "running"

    def _format(self) -> str:
        return FORMATS.get(self._descriptor.get("mode"), "control/joint")

    def _resources(self) -> list:
        """Physical channels this card occupies, for the ACP barrier.

        Taken from the negotiated descriptor's `groups` once there is one: only
        the downstream driver knows what it actually owns. Tianyi's action space
        is four channels (both arms, both hands); a card that kept claiming a
        single configured `arm` would let something else drive the hands while a
        policy was moving them.

        Before `start` there is no descriptor — the tool list is fetched long
        before anything is wired — so the configured value stands in. agent-core
        re-reads the schema on heartbeat (`api/mcp_manage.py` extracts
        `x-resource` there as well as at registration), so the negotiated set
        replaces it shortly after the card starts.
        """
        groups = self._descriptor.get("groups") or []
        negotiated = []
        for group in groups:
            resource = (group or {}).get("resource")
            if resource and resource not in negotiated:
                negotiated.append(resource)
        if negotiated:
            return negotiated

        configured = self._cfg.get("resource") or "arm"
        return configured if isinstance(configured, list) else [configured]

    def _error(self, message: str) -> dict:
        with self._lock:
            self._last_error = message
        log.warning("vla: %s", message)
        return {"state": "error", "message": message}

    @staticmethod
    def _close(provider):
        if provider is None:
            return
        try:
            provider.close()
        except Exception as error:      # noqa: BLE001
            log.warning("vla provider close failed: %s", error)

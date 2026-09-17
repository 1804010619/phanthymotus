"""SmolVLA on this robot, via LeRobot.

**One provider per model, named after the model** — the same shape
`perception/plugins/` has, where `asr.py`, `tts.py` and `vop.py` each own their
weights, their download and their load. A single `local` provider would have had
to grow a switch over model families, and every model's quirks would have piled
up behind it: SmolVLA's action padding, π0's JAX stack, UnifoLM's flash-attn
build. They do not belong in one file.

What is shared lives in the card and in `providers/__init__.py`: discovery, the
four-method protocol, and the negotiation against the arm. What is specific to
*this checkpoint family* lives here.

This is also the only kind of provider whose resource use lands on the robot's
own budget, which is why almost everything below is about *when* things are
loaded rather than about inference.

Three rules from the design (phanthymotus/docs/vla-integration.md § 4.5), and
each is a thing that goes wrong if skipped:

**Lazy import.** torch and lerobot are imported inside the functions that need
them. A card nobody selected must not pull a GPU stack into the process, and
`providers/__init__.py` reports an import failure rather than raising — so an
actucore image without lerobot still offers `mock` and still starts.

**Lazy download.** Weights come from COS with size and SHA256 pinned, through
perception's `model_downloader`, which already handles the parts that are easy
to get wrong: a check file that must not appear until the model behind it is
complete, a lock so concurrent starts fetch one copy, archives staged before
they are merged. Not reimplemented here.

**Lazy load, and load off the calling thread.** `__init__` reads the
checkpoint's *config* — cheap, and enough to answer `capabilities()` so the card
can negotiate against the arm — then loads the weights in the background.
`health()` is False until they are in. Blocking `start` on a multi-second load
is how a card reports ready before it can act, which the TTS card already taught
this project once.

**What this cannot do for you.** A SmolVLA checkpoint is trained for a specific
robot, and its action dimension is that robot's. Pointing it at a 26-DOF
humanoid will fail negotiation, correctly and immediately — that is a
fine-tuning or retargeting problem, not a configuration one, and the error says
so rather than letting the mismatch reach a motor.
"""

from __future__ import annotations

import json
import logging
import os
import threading

log = logging.getLogger(__name__)

# The checkpoint layout LeRobot writes. `config.json` is read directly rather
# than through the library so that capabilities are available before torch is
# imported at all.
CONFIG_FILE = "config.json"


class SmolVLAProvider:
    """A SmolVLA checkpoint loaded in this process.

    Config keys:
        model_dir      where the checkpoint lives (default /models/vla/smolvla)
        model_id       upstream id, for the record — ModelScope first (see
                       docs/vla-integration.md § 4.5.4). Not fetched from
                       directly: the robot pulls from COS.
        weights        optional {base_url, files:{name:{size,sha256}}} manifest;
                       fetched into model_dir when the checkpoint is absent
        device         "cuda" | "cpu" (default cuda, falling back to cpu)
        feature_map    {our observation name: the policy's input key}, e.g.
                       {"main": "observation.images.top", "state":
                       "observation.state"}
        chunk_size     override the checkpoint's action horizon
    """

    def __init__(self, descriptor: dict, config: dict | None = None):
        config = dict(config or {})
        self._descriptor = descriptor or {}
        self._model_dir = config.get("model_dir") or "/models/vla/smolvla"
        self._device = str(config.get("device") or "cuda")
        self._feature_map = dict(config.get("feature_map") or {})
        self._weights = config.get("weights") or {}
        self._chunk_override = config.get("chunk_size")

        self._policy = None
        self._error = ""
        self._lock = threading.RLock()
        self._closed = False

        self._require_lerobot()
        self._ensure_checkpoint()
        self._config = self._read_config()
        # Weights in the background: `capabilities()` is answerable from the
        # config alone, so negotiation can fail fast on a mismatched action
        # space without having waited for several gigabytes to load.
        self._loader = threading.Thread(target=self._load, name="vla-local-load",
                                        daemon=True)
        self._loader.start()

    # ── provider protocol ────────────────────────────────────────────────────

    def capabilities(self) -> dict:
        features = self._config.get("input_features") or {}
        image_keys = [k for k in features if "image" in k]
        return {
            "model": self._config.get("type") or "smolvla",
            "action_dim": self._action_dim(),
            "chunk_size": self._chunk_size(),
            "control_hz": float(self._config.get("fps") or 30.0),
            "needs_state": any("state" in k for k in features),
            "n_cameras": len(image_keys),
            "image_size": self._image_size(features, image_keys),
            # LeRobot ships RTC for the flow-matching policies, but this
            # provider does not implement the prefix conditioning it needs, so
            # it must not claim it — a card that believed this would hand it an
            # inference_delay nothing acts on.
            "supports_rtc": False,
            "ready": self._policy is not None,
            "error": self._error,
        }

    def infer(self, observation=None, inference_delay: int = 0) -> list:
        policy = self._policy
        if policy is None:
            raise RuntimeError(self._error or "weights are still loading")

        batch = self._batch(observation)
        chunk = self._predict(policy, batch)
        return [[float(v) for v in step] for step in chunk]

    def health(self) -> bool:
        return self._policy is not None and not self._error

    def close(self) -> None:
        with self._lock:
            self._closed = True
            policy, self._policy = self._policy, None
        if policy is None:
            return
        del policy
        # CUDA context is never returned to the OS; freeing the cache is all
        # that can be done from inside the process, which is why the design
        # would rather run this provider in its own process on a small board.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:       # noqa: BLE001 — teardown is best effort
            pass

    # ── loading ──────────────────────────────────────────────────────────────

    @staticmethod
    def _require_lerobot():
        """Refuse at start on an image that cannot have lerobot, and say why.

        `find_spec` rather than an import: the point of this file is that torch
        and lerobot are only imported when something is actually going to run,
        and a presence check must not undo that.

        This is the normal state on the JetPack 5.11 image line, and it is not
        a packaging oversight — that line is CUDA 11.4, lerobot needs
        torch >= 2.2.1, and no torch >= 2.2 supports CUDA 11.4. Local inference
        cannot exist there, so the honest thing is to fail the start with the
        reason instead of loading in the background and reporting `unhealthy`
        forever.
        """
        import importlib.util

        if importlib.util.find_spec("lerobot") is None:
            raise ModuleNotFoundError(
                "lerobot is not installed in this image. On JetPack 5.11 that is "
                "expected and permanent: CUDA 11.4 cannot host torch >= 2.2.1, "
                "which lerobot requires — use a remote provider there. On "
                "JetPack 6.1 the actucore base image carries it; check the image "
                "was built from jetson-base-actucore."
            )

    def _ensure_checkpoint(self):
        """Fetch the checkpoint from COS if it is not already on disk.

        Absent a `weights` manifest this only checks: pinning a size and a
        SHA256 for a file nobody has staged yet would be inventing them, and an
        unpinned download of something that drives motors is not an improvement
        on a clear error.
        """
        if os.path.exists(os.path.join(self._model_dir, CONFIG_FILE)):
            return
        manifest = self._weights
        base_url, files = manifest.get("base_url"), manifest.get("files")
        if not base_url or not files:
            raise FileNotFoundError(
                f"no checkpoint at {self._model_dir} and no `weights` manifest "
                f"configured. Stage the checkpoint on COS (ModelScope first — "
                f"see docs/vla-integration.md § 4.5.4) and put its base_url plus "
                f"per-file size/sha256 in the card's `weights` config."
            )
        # perception's downloader: existing → size → sha256 → reuse, otherwise
        # lock, re-check, download with retry, verify, atomic replace.
        from model_downloader import ensure_verified_bundle

        ensure_verified_bundle("vla-local", self._model_dir, base_url, files)

    def _read_config(self) -> dict:
        path = os.path.join(self._model_dir, CONFIG_FILE)
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as error:      # noqa: BLE001
            raise RuntimeError(
                f"could not read {path}: {error}. capabilities() is derived "
                f"from it, and without it the card cannot check the model "
                f"against the arm before moving anything."
            ) from error

    def _load(self):
        """Background weight load. Failures are recorded, never raised here."""
        try:
            policy = self._build_policy()
        except Exception as error:      # noqa: BLE001 — reported via health()
            self._error = f"{type(error).__name__}: {error}"
            log.warning("smolvla provider failed to load: %s", self._error)
            return
        with self._lock:
            if self._closed:            # stopped while we were loading
                return
            self._policy = policy
        log.info("smolvla provider ready: %s on %s",
                 self._config.get("type"), self._device)

    def _build_policy(self):
        """Construct the LeRobot policy. The one version-sensitive call here.

        Kept to a couple of lines on purpose: everything else in this file works
        against plain dicts and is tested without torch, so when LeRobot's API
        moves this is the only thing to fix.
        """
        from lerobot.policies.factory import make_policy_config  # noqa: F401
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        policy = SmolVLAPolicy.from_pretrained(self._model_dir)
        policy.to(self._resolved_device())
        policy.eval()
        return policy

    def _resolved_device(self) -> str:
        if self._device != "cuda":
            return self._device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:       # noqa: BLE001
            pass
        log.warning("cuda unavailable; smolvla provider falling back to cpu")
        return "cpu"

    # ── inference ────────────────────────────────────────────────────────────

    def _batch(self, observation) -> dict:
        """Our observation, in the keys this checkpoint was trained on.

        The mapping is configuration rather than convention because the keys
        travel with the training dataset — `observation.images.top` on one
        checkpoint and `observation.images.cam_high` on the next — and guessing
        produces a policy acting on a black image rather than an error.
        """
        import torch

        if observation is None:
            raise ValueError("smolvla provider needs an observation")

        expected = set(self._config.get("input_features") or {})
        batch, missing = {}, []
        device = self._resolved_device()

        for name, image in (getattr(observation, "images", None) or {}).items():
            key = self._feature_map.get(name)
            if not key:
                continue
            batch[key] = torch.as_tensor(image).to(device)

        state = getattr(observation, "state", None)
        if state is not None:
            key = self._feature_map.get("state", "observation.state")
            batch[key] = torch.as_tensor(state).to(device)

        for key in expected:
            if key not in batch:
                missing.append(key)
        if missing:
            raise KeyError(
                f"this checkpoint expects {sorted(missing)} and the card's "
                f"feature_map does not supply them. feature_map maps our "
                f"observation names to the policy's input keys."
            )

        batch["task"] = getattr(observation, "prompt", "") or ""
        return batch

    @staticmethod
    def _predict(policy, batch) -> list:
        """One chunk out of the policy. The second version-sensitive call.

        `predict_action_chunk` is what LeRobot's async and RTC paths use;
        `select_action` is the single-step fallback for a policy that has no
        chunk method, wrapped so the caller always sees a chunk.
        """
        import torch

        with torch.no_grad():
            if hasattr(policy, "predict_action_chunk"):
                chunk = policy.predict_action_chunk(batch)
            else:
                chunk = policy.select_action(batch)

        chunk = chunk.detach().to("cpu")
        # (B, T, D) → (T, D); (T, D) stays; (D,) becomes one step.
        if chunk.ndim == 3:
            chunk = chunk[0]
        elif chunk.ndim == 1:
            chunk = chunk.unsqueeze(0)
        return chunk.tolist()

    # ── capabilities, from the checkpoint's own config ───────────────────────

    def _action_dim(self):
        """The checkpoint's action width, not the network's.

        SmolVLA pads its action tensor to a fixed internal maximum, so the
        network's width says nothing about the robot it was trained on. What
        matters is the `action` output feature's shape, which carries the
        dataset's real dimension — and that is the number negotiation must
        compare against the arm.
        """
        features = self._config.get("output_features") or {}
        action = features.get("action") or {}
        shape = action.get("shape")
        if isinstance(shape, (list, tuple)) and shape:
            return int(shape[-1])
        return None

    def _chunk_size(self):
        if self._chunk_override:
            return int(self._chunk_override)
        for key in ("n_action_steps", "chunk_size", "horizon"):
            value = self._config.get(key)
            if isinstance(value, int) and value > 0:
                return value
        return 1

    @staticmethod
    def _image_size(features: dict, image_keys: list):
        for key in image_keys:
            shape = (features.get(key) or {}).get("shape")
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                return int(shape[-1])
        return 0


def PROVIDER(descriptor: dict, config: dict | None = None) -> SmolVLAProvider:
    return SmolVLAProvider(descriptor, config)


# Discovery checks the four methods on whatever PROVIDER is; a factory function
# has none of them, so they are advertised here.
for _name in ("capabilities", "infer", "health", "close"):
    setattr(PROVIDER, _name, getattr(SmolVLAProvider, _name))

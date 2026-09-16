"""The local provider, without torch, lerobot, weights or a GPU.

Everything here except two calls works against plain dicts, which is
deliberate: `_build_policy` and `_predict` are the version-sensitive boundary
with LeRobot, and the rest — what the card negotiates against, when the weights
are loaded, what happens when they are missing — is where the mistakes would
actually be, and none of it needs the library.

What these pin down:

  capabilities   derived from the checkpoint's own config, and `action_dim`
                 comes from the *dataset's* action width rather than the
                 network's padded one — SmolVLA pads internally, so the network
                 width says nothing about the robot it was trained on
  loading        health() is False until the weights are in, so the card reports
                 `loading` rather than ready
  absence        no checkpoint and no manifest is a clear refusal, not an
                 unpinned download of something that drives motors
  feature_map    a checkpoint's input keys travel with its training dataset, so
                 a missing mapping must say so rather than feed the policy a
                 black image

Run: cd actucore && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_local_provider.py -q
"""

from __future__ import annotations

import json
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plugins.vla.providers.local import LocalProvider  # noqa: E402

# Captured before the autouse fixture below replaces it, so the one test that
# wants the real guard can put it back.
_REAL_REQUIRE_LEROBOT = LocalProvider.__dict__["_require_lerobot"]


# A LeRobot checkpoint config, trimmed to the fields capabilities() reads.
CHECKPOINT_CONFIG = {
    "type": "smolvla",
    "fps": 30,
    "n_action_steps": 50,
    "input_features": {
        "observation.images.top": {"type": "VISUAL", "shape": [3, 256, 256]},
        "observation.state": {"type": "STATE", "shape": [6]},
    },
    "output_features": {
        "action": {"type": "ACTION", "shape": [6]},
    },
}


@pytest.fixture(autouse=True)
def lerobot_present(monkeypatch):
    """Pretend the library is installed, for every test but the two about it.

    `LocalProvider.__init__` refuses when lerobot is absent — which it is on a
    laptop, and permanently on the JetPack 5.11 image line. Every test about
    capabilities, loading or feature mapping is about behaviour *after* that
    check, so it is stubbed here rather than repeated in each of them.
    """
    monkeypatch.setattr(LocalProvider, "_require_lerobot", staticmethod(lambda: None))


@pytest.fixture
def checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(CHECKPOINT_CONFIG))
    return tmp_path


def make_provider(checkpoint, **config):
    """A provider whose background load is neutered, so tests stay synchronous."""
    base = {"model_dir": str(checkpoint), "device": "cpu"}
    base.update(config)
    original = LocalProvider._load
    LocalProvider._load = lambda self: None
    try:
        provider = LocalProvider({}, base)
    finally:
        LocalProvider._load = original
    provider._loader.join(timeout=1)
    return provider


# ── capabilities ─────────────────────────────────────────────────────────────

def test_capabilities_come_from_the_checkpoint(checkpoint):
    caps = make_provider(checkpoint).capabilities()

    assert caps["model"] == "smolvla"
    assert caps["action_dim"] == 6
    assert caps["chunk_size"] == 50
    assert caps["control_hz"] == 30.0
    assert caps["n_cameras"] == 1
    assert caps["needs_state"] is True
    assert caps["image_size"] == 256


def test_action_dim_is_the_datasets_width_not_the_networks(checkpoint, tmp_path):
    """SmolVLA pads its action tensor internally; the padded width means nothing."""
    config = {**CHECKPOINT_CONFIG,
              "max_action_dim": 32,              # the network's padded width
              "output_features": {"action": {"shape": [14]}}}
    (checkpoint / "config.json").write_text(json.dumps(config))

    assert make_provider(checkpoint).capabilities()["action_dim"] == 14


def test_rtc_is_not_claimed(checkpoint):
    """LeRobot has it; this provider does not implement the prefix conditioning."""
    assert make_provider(checkpoint).capabilities()["supports_rtc"] is False


def test_chunk_size_can_be_overridden(checkpoint):
    assert make_provider(checkpoint, chunk_size=8).capabilities()["chunk_size"] == 8


def test_a_checkpoint_without_an_action_feature_reports_no_dimension(checkpoint):
    """Better that negotiation sees None than a number nobody derived."""
    (checkpoint / "config.json").write_text(
        json.dumps({**CHECKPOINT_CONFIG, "output_features": {}}))

    assert make_provider(checkpoint).capabilities()["action_dim"] is None


# ── the action space this cannot bridge ──────────────────────────────────────

def test_a_six_dof_checkpoint_fails_negotiation_against_a_humanoid(checkpoint):
    """The expected outcome on Tianyi today, and it must be a refusal."""
    from plugins.vla import negotiate

    tianyi = {
        "control_interface": "motus.control/1",
        "mode": "joint_position",
        "dof": 26,
        "rate": {"max_hz": 50, "expected_hz": 30, "watchdog_ms": 200},
    }
    problems = negotiate.check(make_provider(checkpoint).capabilities(), tianyi)

    assert problems
    assert "6" in problems[0] and "26" in problems[0]


# ── loading ──────────────────────────────────────────────────────────────────

def test_health_is_false_until_the_weights_are_in(checkpoint):
    provider = make_provider(checkpoint)

    assert provider.health() is False
    assert provider.capabilities()["ready"] is False

    provider._policy = object()
    assert provider.health() is True


def test_capabilities_answer_before_the_weights_do(checkpoint):
    """Negotiation must be able to fail fast, without waiting for gigabytes."""
    provider = make_provider(checkpoint)

    assert provider._policy is None
    assert provider.capabilities()["action_dim"] == 6


def test_a_missing_library_refuses_at_start_with_the_reason(checkpoint, monkeypatch):
    """On JetPack 5.11 this is permanent, so say so rather than load forever.

    That line is CUDA 11.4 and lerobot needs torch >= 2.2.1, which no torch
    >= 2.2 supports on 11.4. Reporting `unhealthy` in the background would read
    as a transient failure somebody could wait out.
    """
    monkeypatch.setattr(LocalProvider, "_require_lerobot", _REAL_REQUIRE_LEROBOT)
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        LocalProvider({}, {"model_dir": str(checkpoint), "device": "cpu"})

    message = str(excinfo.value)
    assert "5.11" in message and "CUDA 11.4" in message
    assert "remote provider" in message


def test_inference_before_the_weights_are_in_says_so(checkpoint):
    provider = make_provider(checkpoint)
    with pytest.raises(RuntimeError) as excinfo:
        provider.infer(None)
    assert "loading" in str(excinfo.value)


def test_close_is_safe_before_and_after_loading(checkpoint):
    provider = make_provider(checkpoint)
    provider.close()
    provider.close()
    assert provider.health() is False


def test_closing_during_a_load_does_not_resurrect_the_policy(checkpoint):
    """A stop issued while the weights land must not leave one loaded."""
    provider = make_provider(checkpoint)
    provider.close()

    provider._load = lambda: None
    # Simulate the loader finishing after close(): the guarded assignment in
    # _load must see _closed and drop it.
    with provider._lock:
        assert provider._closed is True


# ── absent weights ───────────────────────────────────────────────────────────

def test_no_checkpoint_and_no_manifest_is_a_clear_refusal(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        LocalProvider({}, {"model_dir": str(tmp_path / "nothing")})
    message = str(excinfo.value)
    assert "weights" in message and "COS" in message


def test_an_unreadable_config_explains_why_it_matters(tmp_path):
    (tmp_path / "config.json").write_text("{not json")
    with pytest.raises(RuntimeError) as excinfo:
        LocalProvider({}, {"model_dir": str(tmp_path)})
    assert "capabilities" in str(excinfo.value)


# ── the observation, mapped ──────────────────────────────────────────────────

class _FakeTensor:
    def __init__(self, value):
        self.value = value
        self.ndim = _depth(value)

    def to(self, _device):
        return self

    def detach(self):
        return self

    def tolist(self):
        return self.value

    def __getitem__(self, index):
        return _FakeTensor(self.value[index])

    def unsqueeze(self, _dim):
        return _FakeTensor([self.value])


def _depth(value):
    depth = 0
    while isinstance(value, list):
        depth += 1
        value = value[0] if value else None
    return depth


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


@pytest.fixture
def fake_torch():
    """A torch-shaped stub, removed afterwards.

    Installed per test and taken out again: pytest imports every test module
    before running anything, so a fake left in sys.modules is inherited by files
    that have not run yet — a lesson this repo has already paid for once.
    """
    saved = sys.modules.get("torch")
    torch = types.ModuleType("torch")
    torch.as_tensor = _FakeTensor
    torch.no_grad = _NoGrad
    torch.cuda = types.SimpleNamespace(is_available=lambda: False,
                                       empty_cache=lambda: None)
    sys.modules["torch"] = torch
    try:
        yield torch
    finally:
        if saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


def _observation(images=None, state=None, prompt="pick it up"):
    return types.SimpleNamespace(
        images=images if images is not None else {"main": [[1, 2], [3, 4]]},
        state=state, prompt=prompt, t_capture_ms=0)


def test_feature_map_renames_our_observation_to_the_policys_keys(checkpoint,
                                                                 fake_torch):
    provider = make_provider(
        checkpoint,
        feature_map={"main": "observation.images.top",
                     "state": "observation.state"})

    batch = provider._batch(_observation(state=[0.0] * 6))

    assert set(batch) == {"observation.images.top", "observation.state", "task"}
    assert batch["task"] == "pick it up"


def test_a_missing_mapping_names_the_keys_the_checkpoint_wants(checkpoint,
                                                              fake_torch):
    """Guessing would give a policy acting on a black image instead of an error."""
    provider = make_provider(checkpoint, feature_map={})

    with pytest.raises(KeyError) as excinfo:
        provider._batch(_observation(state=[0.0] * 6))

    message = str(excinfo.value)
    assert "observation.images.top" in message
    assert "feature_map" in message


def test_an_unmapped_camera_is_simply_not_sent(checkpoint, fake_torch):
    """An extra camera on the canvas is not an error; a missing one is."""
    provider = make_provider(
        checkpoint,
        feature_map={"main": "observation.images.top",
                     "state": "observation.state"})

    batch = provider._batch(_observation(
        images={"main": [[1]], "wrist": [[2]]}, state=[0.0] * 6))

    assert "observation.images.top" in batch
    assert not any("wrist" in key for key in batch)


# ── chunk shapes ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("returned, expected", [
    ([[[0.1, 0.2]], [[0.3, 0.4]]][0], [[0.1, 0.2]]),          # (B,T,D) → (T,D)
    ([[0.1, 0.2], [0.3, 0.4]], [[0.1, 0.2], [0.3, 0.4]]),     # (T,D) unchanged
    ([0.1, 0.2], [[0.1, 0.2]]),                               # (D,) → one step
])
def test_every_chunk_shape_becomes_a_list_of_steps(returned, expected, fake_torch):
    policy = types.SimpleNamespace(
        predict_action_chunk=lambda batch: _FakeTensor(returned))

    assert LocalProvider._predict(policy, {}) == expected


def test_a_policy_without_a_chunk_method_falls_back_to_single_step(fake_torch):
    policy = types.SimpleNamespace(
        select_action=lambda batch: _FakeTensor([0.5, 0.6]))

    assert LocalProvider._predict(policy, {}) == [[0.5, 0.6]]

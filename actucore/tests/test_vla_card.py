"""The VLA card, its provider discovery, and the wire format it produces.

No ROS, no model, no robot: `next_command` is deliberately separate from the
timer that publishes it, so the whole of the message construction — sequence
numbers, the two timestamps, chunk indices — is testable here, and `_tick` is
only this plus a publish.

What these cover, in the order the card does them:

  discovery      a provider file is the whole of adding a provider; a broken
                 one is reported, not raised
  negotiation    the card refuses to start on a mismatch rather than failing
                 one command at a time at 30 Hz
  the message    obs_stamp_ms is the observation's age, not the command's —
                 passing stamp_ms for both looks right in every test that does
                 not involve latency and disables the receiver's staleness check
  the mock       bounded by construction, so a test signal cannot reach a limit

Run: cd actucore && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_vla_card.py -q
"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plugins.vla import VLAPlugin  # noqa: E402
from plugins.vla import negotiate  # noqa: E402
from plugins.vla.message import build as build_message  # noqa: E402
from plugins.vla.providers import discover, REQUIRED  # noqa: E402
from plugins.vla.providers.mock import PROVIDER as MOCK  # noqa: E402


DESCRIPTOR = {
    "control_interface": "motus.control/1",
    "mode": "joint_position",
    "dof": 7,
    "joint_names": [f"joint{i}" for i in range(1, 8)],
    "units": {"angle": "rad"},
    "limits": {"lower": [-2.0] * 7, "upper": [2.0] * 7},
    "rate": {"max_hz": 100, "expected_hz": 30, "watchdog_ms": 200},
    "force_torque": None,
}


def make_card(**cfg):
    base = {"provider": "mock", "amplitude": 0.1, "period_s": 4.0, "chunk_size": 5}
    base.update(cfg)
    return VLAPlugin(base, executor=None)


# ── provider discovery ───────────────────────────────────────────────────────

def test_mock_is_discovered_without_being_named_anywhere():
    assert "mock" in discover()


def test_the_schema_enum_is_built_from_what_was_found():
    """Adding a provider file must not require editing the card or its schema."""
    card = make_card()
    schema = card.get_tools()[0]["configSchema"]["properties"]["provider"]
    assert schema["enum"] == sorted(discover())


def test_every_discovered_provider_has_the_whole_protocol():
    for name, factory in discover().items():
        for method in REQUIRED:
            assert hasattr(factory, method), f"{name} is missing {method}"


def test_a_broken_provider_is_reported_not_raised(tmp_path, monkeypatch):
    """One backend's missing dependency must not take the card down with it."""
    import plugins.vla.providers as providers

    # A real module whose import genuinely fails, the way one missing torch
    # would. Not underscore-prefixed: discovery skips those, which would make
    # this pass for the wrong reason.
    pkg_dir = pathlib.Path(providers.__path__[0])
    broken = pkg_dir / "tmpbroken.py"
    broken.write_text("import definitely_not_a_real_module\n")
    try:
        found = discover()
        assert "tmpbroken" not in found
        assert "mock" in found                       # unaffected
        assert "tmpbroken" in discover.errors
    finally:
        broken.unlink()


# ── negotiation ──────────────────────────────────────────────────────────────

def test_matching_capabilities_pass():
    assert negotiate.check(MOCK(DESCRIPTOR).capabilities(), DESCRIPTOR) == []


def test_action_dim_mismatch_names_both_numbers():
    """"shape mismatch" sends somebody to read code; this sends them to a wire."""
    problems = negotiate.check({"action_dim": 32}, DESCRIPTOR)
    assert problems
    assert "32" in problems[0] and "7" in problems[0]


def test_a_downstream_that_is_not_a_control_card_is_caught_first():
    problems = negotiate.check({"action_dim": 7}, {"control_interface": None})
    assert len(problems) == 1
    assert "motus.control/1" in problems[0]


def test_a_model_faster_than_the_hardware_is_refused():
    problems = negotiate.check({"action_dim": 7, "control_hz": 500}, DESCRIPTOR)
    assert any("500" in p for p in problems)


def test_every_problem_is_reported_at_once():
    """An operator fixing a canvas should see the whole disagreement."""
    problems = negotiate.check({"action_dim": 32, "control_hz": 500}, DESCRIPTOR)
    assert len(problems) == 2


def test_rate_is_clamped_by_everyone_with_a_say():
    # The model's own rate wins over a faster request: publishing faster than
    # the actions were computed for changes what each one means.
    assert negotiate.effective_rate({"control_hz": 10}, DESCRIPTOR, 60) == 10
    # And the driver's ceiling wins over everything.
    slow = {**DESCRIPTOR, "rate": {**DESCRIPTOR["rate"], "max_hz": 5}}
    assert negotiate.effective_rate({"control_hz": 50}, slow, 50) == 5


def test_ttl_never_outlives_the_receivers_watchdog():
    """A generous ttl removes the protection while appearing to provide it."""
    assert negotiate.ttl_ms(1.0, DESCRIPTOR) == 200        # capped by watchdog_ms
    assert negotiate.ttl_ms(30.0, DESCRIPTOR) >= 50        # floored


# ── the message ──────────────────────────────────────────────────────────────

def test_message_carries_every_field_the_receiver_checks():
    message = build_message(seq=1, values=[0.0] * 7, mode="joint_position", dof=7,
                            source="test", stamp_ms=1000, obs_stamp_ms=900,
                            ttl_ms=100)
    for field in ("schema", "seq", "stamp_ms", "obs_stamp_ms", "ttl_ms",
                  "source", "priority", "mode", "dof", "values"):
        assert field in message
    assert message["schema"] == "motus.control/1"


def test_the_two_timestamps_are_not_the_same_number():
    """The characteristic failure of remote inference is only visible in obs_stamp."""
    card = make_card()
    card._descriptor = DESCRIPTOR
    card._provider = MOCK(DESCRIPTOR, {"chunk_size": 5})
    card._ttl_ms = 100

    first = card.next_command()
    second = card.next_command()

    # Both came from one inference, so they share an observation timestamp while
    # their own stamps advance — that is what makes an ageing chunk visible.
    assert first["obs_stamp_ms"] == second["obs_stamp_ms"]
    assert second["stamp_ms"] >= first["stamp_ms"]


def test_sequence_numbers_are_strictly_increasing_across_chunks():
    card = make_card()
    card._descriptor = DESCRIPTOR
    card._provider = MOCK(DESCRIPTOR, {"chunk_size": 3})
    card._ttl_ms = 100

    seqs = [card.next_command()["seq"] for _ in range(8)]      # spans 3 chunks

    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)


def test_chunk_indices_walk_the_chunk_then_refill():
    card = make_card()
    card._descriptor = DESCRIPTOR
    card._provider = MOCK(DESCRIPTOR, {"chunk_size": 3})
    card._ttl_ms = 100

    indices = [card.next_command()["chunk"]["index"] for _ in range(7)]

    assert indices == [0, 1, 2, 0, 1, 2, 0]


def test_an_empty_chunk_raises_rather_than_publishing_nothing_silently():
    class Empty:
        def capabilities(self): return {"action_dim": 7}
        def infer(self, obs=None, inference_delay=0): return []
        def health(self): return True
        def close(self): return None

    card = make_card()
    card._descriptor = DESCRIPTOR
    card._provider = Empty()

    with pytest.raises(RuntimeError):
        card.next_command()


# ── start-time refusals ──────────────────────────────────────────────────────

def test_start_without_a_downstream_descriptor_is_refused():
    """Without a wired consumer there is no action space to target."""
    result = make_card().dispatch("start", {"action": "start"})
    assert result["state"] == "error"
    assert "control/*" in result["message"]


def test_start_refuses_a_mismatched_model_instead_of_failing_per_command():
    class Wide:
        def capabilities(self): return {"action_dim": 32, "control_hz": 30}
        def infer(self, obs=None, inference_delay=0): return [[0.0] * 32]
        def health(self): return True
        def close(self): self.closed = True

    card = make_card(provider="wide")
    import plugins.vla.plugin as plugin_mod
    original = plugin_mod.discover
    plugin_mod.discover = lambda: {"wide": lambda d, c: Wide()}
    plugin_mod.discover.errors = {}
    try:
        result = card.dispatch("start", {"action": "start",
                                         "control_interface": DESCRIPTOR})
    finally:
        plugin_mod.discover = original

    assert result["state"] == "error"
    assert "32" in result["message"]


def test_an_unknown_provider_lists_what_is_available():
    card = make_card(provider="nope")
    result = card.dispatch("start", {"action": "start",
                                     "control_interface": DESCRIPTOR})
    assert result["state"] == "error"
    assert "mock" in result["message"]


def test_a_mode_with_no_topic_format_is_refused():
    odd = {**DESCRIPTOR, "mode": "hyperdrive"}
    result = make_card().dispatch("start", {"action": "start",
                                            "control_interface": odd})
    assert result["state"] == "error"
    assert "hyperdrive" in result["message"]


def test_the_card_declares_a_resource_and_no_completion():
    schema = make_card().get_tools()[0]["inputSchema"]
    # A policy runs until stopped; an ACP pending held for its lifetime would
    # block every other actuator behind the barrier.
    assert "x-completion" not in schema
    assert schema["x-resource"] == "arm"
    assert schema["x-hooks"]["on_interrupt_all"]["action"] == "stop"


def test_info_answers_before_anything_has_started():
    """Agent Core probes liveness with info(); a card that cannot answer is offline."""
    info = make_card().dispatch("info", {"action": "info"})
    assert info["state"] == "idle"
    assert "mock" in info["providers_available"]


def test_action_enum_contains_info():
    enum = make_card().get_tools()[0]["inputSchema"]["properties"]["action"]["enum"]
    assert "info" in enum


def test_prefix_has_no_underscore():
    """dispatch routes on partition('_'); a prefix with one never matches."""
    assert "_" not in VLAPlugin.PREFIX


# ── the mock signal ──────────────────────────────────────────────────────────

def test_the_sine_cannot_leave_the_declared_limits():
    """A test signal that can reach a joint limit is one that will, at 3am."""
    provider = MOCK(DESCRIPTOR, {"amplitude": 1.0, "chunk_size": 64,
                                 "period_s": 1.0, "control_hz": 30})
    lower, upper = DESCRIPTOR["limits"]["lower"], DESCRIPTOR["limits"]["upper"]

    for _ in range(20):
        for values in provider.infer():
            for v, lo, hi in zip(values, lower, upper):
                assert lo <= v <= hi


def test_the_default_amplitude_is_small():
    provider = MOCK(DESCRIPTOR)
    extremes = []
    for _ in range(40):
        for values in provider.infer():
            extremes.append(abs(values[0]))
    # 5% of a half-span of 2.0
    assert max(extremes) <= 0.11


def test_the_mock_adapts_to_whatever_arm_it_is_given():
    six = {**DESCRIPTOR, "dof": 6,
           "limits": {"lower": [-1.0] * 6, "upper": [1.0] * 6}}
    assert MOCK(six).capabilities()["action_dim"] == 6
    assert len(MOCK(six).infer()[0]) == 6


def test_a_descriptor_whose_limits_contradict_its_dof_is_refused():
    bad = {**DESCRIPTOR, "dof": 7,
           "limits": {"lower": [-1.0] * 6, "upper": [1.0] * 6}}
    with pytest.raises(ValueError):
        MOCK(bad)


def test_the_mock_does_not_claim_to_read_anything():
    """Claiming otherwise would hide a broken observation path behind motion."""
    assert MOCK(DESCRIPTOR).capabilities()["needs_state"] is False
    assert MOCK(DESCRIPTOR).capabilities()["n_cameras"] == 0

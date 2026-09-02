"""Regression: `speak` → `finish` must not end the turn before the audio plays.

Observed on G1 in an exhibition tour: a ~47s narration was cut off because the
model called `finish` about 4s after `speak`. The reported root cause — perception
firing ACP `completed` at push-EOF instead of playback end — accounts for about
1s (perception's 500ms prebuffer plus the speaker's 300ms PREFILL_BYTES and 240ms
MAX_LEAD_S); the push loop itself is paced to the audio clock, so pushing 47s of
audio takes 47s.

The real hole is that `finish` never reached the ACP barrier. `_dispatch` matches
`name in self._sys_tools` *before* `name.startswith('mcp__')`, and the only
`await_pending` call lived in the latter branch — `_needs_barrier` also returns
False for any non-`mcp__` name. There is no end-of-turn barrier either, so
`speak` → `finish` waited for nothing regardless of when `completed` arrived.

Run: cd agent-core && python3 -m pytest tests/test_acp_barrier_finish.py
"""
import asyncio
import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import mcp_client  # noqa: E402
from event.llm import (  # noqa: E402
    _acp_barrier,
    _acp_barrier_log,
    _sys_tool_needs_barrier,
)

SPEAK_ID = 'speak-deadbeef'


@pytest.fixture(autouse=True)
def clean_pending():
    """Every test starts with an empty ACP pending table and leaves one behind."""
    for d in (mcp_client._pending_actions, mcp_client._pending_results,
              mcp_client._pending_timeouts, mcp_client._pending_tools):
        d.clear()
    yield
    for d in (mcp_client._pending_actions, mcp_client._pending_results,
              mcp_client._pending_timeouts, mcp_client._pending_tools):
        d.clear()


def _arm(action_id=SPEAK_ID, timeout=30.0, tool='tts'):
    """Register a pending ACP action the way mcp_client.call_tool would."""
    mcp_client._pending_actions[action_id] = asyncio.Event()
    mcp_client._pending_tools[action_id] = tool
    mcp_client._pending_timeouts[action_id] = timeout
    return mcp_client._pending_actions[action_id]


def _complete(action_id=SPEAK_ID, status='completed'):
    """What POST /api/acp/complete does to unblock the barrier."""
    mcp_client._pending_results[action_id] = {'action_id': action_id, 'status': status}
    mcp_client._pending_actions[action_id].set()


# ── which tools the barrier gates ────────────────────────────────────────────

def test_finish_is_gated():
    assert _sys_tool_needs_barrier('finish')


@pytest.mark.parametrize('name', [
    'task_update',      # 每站都要调；挡住它就等于每站多等一整段音频
    'task_create',
    'task_done',
    'update_memory',
    'subagent_status',
])
def test_other_system_tools_are_not_gated(name):
    assert not _sys_tool_needs_barrier(name)


# ── the barrier itself ───────────────────────────────────────────────────────

def test_no_pending_does_not_block():
    """The common case — a turn with no audio in flight must not pay anything."""
    async def scenario():
        return await asyncio.wait_for(_acp_barrier('finish', None), timeout=1)

    assert asyncio.run(scenario()) is None


def test_finish_waits_until_playback_completes():
    """The regression: finish must not return while a speak is still pending."""
    async def scenario():
        ev = _arm()
        barrier = asyncio.create_task(_acp_barrier('finish', None))

        # Give the barrier a chance to run and block.
        await asyncio.sleep(0.05)
        assert not barrier.done(), 'finish returned before the speak completed'
        assert not ev.is_set()

        _complete()
        return await asyncio.wait_for(barrier, timeout=1)

    result = asyncio.run(scenario())
    assert result['status'] == 'completed'
    assert result['actions'] == [SPEAK_ID]
    assert not mcp_client._pending_actions, 'pending table must be cleared'


def test_barge_in_cancels_the_wait():
    """"别说了" must still cut through — the wait honours cancel_event."""
    async def scenario():
        _arm()
        cancel = asyncio.Event()
        barrier = asyncio.create_task(_acp_barrier('finish', cancel))

        await asyncio.sleep(0.05)
        assert not barrier.done()

        cancel.set()
        return await asyncio.wait_for(barrier, timeout=1)

    result = asyncio.run(scenario())
    assert result['status'] == 'cancelled'
    assert not mcp_client._pending_actions


def test_missing_acp_callback_times_out_and_releases():
    """A completion that never arrives must not wedge the turn forever.

    This is the ACP-callback-failed path (self-signed cert, wrong
    AGENT_CORE_URL): perception logs a warning and moves on, so the barrier is
    all that is left. It has to release, and say so.
    """
    async def scenario():
        _arm(timeout=0.1)
        return await asyncio.wait_for(_acp_barrier('finish', None), timeout=2)

    result = asyncio.run(scenario())
    assert result['status'] == 'timeout'
    assert result['actions'] == [SPEAK_ID]
    assert not mcp_client._pending_actions


def test_barrier_uses_the_longest_pending_timeout():
    """Two utterances in flight: the barrier must outlast the slower one."""
    async def scenario():
        _arm('speak-short', timeout=0.05)
        _arm('speak-long', timeout=5.0)
        barrier = asyncio.create_task(_acp_barrier('finish', None))

        # Past the short action's own timeout — the barrier must still be waiting,
        # because effective_timeout is max(...) over every pending action.
        await asyncio.sleep(0.3)
        assert not barrier.done()

        _complete('speak-short')
        _complete('speak-long')
        return await asyncio.wait_for(barrier, timeout=1)

    assert asyncio.run(scenario())['status'] == 'completed'


# ── attribution ──────────────────────────────────────────────────────────────

def test_timeout_is_logged(capsys):
    _acp_barrier_log('finish', {'status': 'timeout', 'actions': [SPEAK_ID]})
    out = capsys.readouterr().out
    assert 'barrier timeout before finish' in out
    assert SPEAK_ID in out


def test_cancelled_is_logged_without_a_trailing_none(capsys):
    """await_pending's cancel branch returns no `actions` key."""
    _acp_barrier_log('finish', {'status': 'cancelled'})
    out = capsys.readouterr().out
    assert 'barrier cancelled before finish' in out
    assert 'None' not in out


@pytest.mark.parametrize('result', [
    {'status': 'completed', 'actions': [SPEAK_ID]},
    {'status': 'no_pending'},
    None,
    'not-a-dict',
])
def test_normal_results_stay_quiet(result, capsys):
    _acp_barrier_log('finish', result)
    assert capsys.readouterr().out == ''

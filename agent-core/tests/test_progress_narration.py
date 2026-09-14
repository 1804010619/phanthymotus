"""
test_progress_narration.py — 框架级主动播报 + 沉默计时状态机。

背景：主动播报原本完全建立在「LLM 自愿写 content，系统自动播出去」之上。两个实测问题：
prompt 约束不住，多轮工具调用期间模型经常一句话不写；写了也不合适，那是内部推理而不是
说给等着的人听的进度汇报。现在改成框架保证，content 自动播报同时废除（回归断言见
test_on_notify_barrier.py）。

计时部分经过一轮简化：**定时器的存在性就是状态**，没有时钟变量，也就不需要「播放期间
冻结」——正在说话时根本不存在定时器。

    事件                              动作
    ───────────────────────────────────────────────────────────────
    输出被派发（开始说话）             停止；该输出若无 ACP 跟踪则当场重新计时
    打断                               停止
    说完了（完成 / 超时 / 取消）       有活 → 重新计时；无活 → 停止
    任务开始                           开始计时（已在计时则不动）
    活全干完                           停止
    轮边界                             不在说话且没在计时 → 开始计时（不变式兜底）
    到点                               有活且 gate 全过 → 汇报；没播成 → 重新计时

本文件里标了「防永久静音」的用例是风险最高的一类：新模型下失败模式不是刷屏而是**彻底
没声**，日志里什么都看不到。每条都要能抓到回归。

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_progress_narration.py
"""
import asyncio
import os
import pathlib
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import client  # noqa: E402
import config  # noqa: E402
import hooks  # noqa: E402
import mcp_client  # noqa: E402
import subagent  # noqa: E402
import event.skills as skills_tools  # noqa: E402
# event/__init__.py rebinds the `event.skills` / `event.llm` *attributes* to instances, so
# those names give you the methods but not the module globals — those must come from
# sys.modules. Same reason event/llm.py reaches get_notify_override through sys.modules.
skills_mod = sys.modules['event.skills']
ell = sys.modules['event.llm']

from event.llm import (  # noqa: E402
    _build_narration_messages,
    _build_system_tools,
    _narration_feedback_message,
    _narration_thresholds,
    _notify_fire_spoke,
    _restricted_channel_tool_allowed,
)

MOUTH_META = {
    'type': 'actuator',
    'action_enum': None,
    'has_config_schema': False,
    'completion': {'actions': ['speak'], 'timeout': 180},
    'resource': frozenset({'mouth'}),
}


class _Fixture(unittest.TestCase):
    """隔离全局注册表、pending、两个运行时覆盖和计时器。"""

    def setUp(self):
        self._saved_registry = dict(mcp_client.registry)
        self._saved_hooks = dict(hooks._registry)
        mcp_client.registry.clear()
        hooks._registry.clear()
        self._saved_pending = tuple(
            dict(d) for d in (mcp_client._pending_actions, mcp_client._pending_tools,
                              mcp_client._pending_resources, mcp_client._pending_timeouts))
        for d in (mcp_client._pending_actions, mcp_client._pending_tools,
                  mcp_client._pending_resources, mcp_client._pending_timeouts):
            d.clear()
        self._saved_notify = skills_mod._notify_override
        self._saved_report = skills_mod._report_override
        skills_mod._notify_override = None
        skills_mod._report_override = None
        self._saved_event = config.main.get('event', {})
        self._saved_mgr = subagent._manager_instance
        self._saved_inst = ell._event_instance
        ell._stop_countdown()
        ell._last_report_text_global = ''
        ell._last_turn_restricted = False
        del ell._pending_narration_feedback[:]

    def tearDown(self):
        ell._stop_countdown()
        del ell._pending_narration_feedback[:]
        mcp_client.registry.clear()
        mcp_client.registry.update(self._saved_registry)
        hooks._registry.clear()
        hooks._registry.update(self._saved_hooks)
        for d, saved in zip((mcp_client._pending_actions, mcp_client._pending_tools,
                             mcp_client._pending_resources, mcp_client._pending_timeouts),
                            self._saved_pending):
            d.clear()
            d.update(saved)
        skills_mod._notify_override = self._saved_notify
        skills_mod._report_override = self._saved_report
        config.main['event'] = self._saved_event
        subagent._manager_instance = self._saved_mgr
        ell._event_instance = self._saved_inst

    # -- helpers ---------------------------------------------------------
    def _register_mouth(self, *, resource=frozenset({'mouth'})):
        mcp_client.registry['dev1'] = {
            'name': 'dev1', 'url': 'http://dev1', 'online': True, 'tools': ['tts'],
            'schemas': {'mcp__dev1__tts__speak': {'name': 'mcp__dev1__tts__speak'}},
            'tool_meta': {'mcp__dev1__tts__speak': {**MOUTH_META, 'resource': resource}},
            'split_map': {'mcp__dev1__tts__speak': {'tool': 'tts', 'action': 'speak'}},
            'tool_groups': {}, 'input_schemas': {},
        }
        hooks.register('dev1', 'tts', {'on_notify': {'action': 'speak'}})

    def _cfg(self, **kw):
        ev = dict(config.main.get('event', {}))
        llm = dict(ev.get('llm', {}))
        llm.update(kw)
        ev['llm'] = llm
        config.main['event'] = ev

    def _speaking(self, aid='spk', resource=frozenset({'mouth'})):
        """模拟一次正在播放的输出。返回它的 Event。"""
        evt = asyncio.Event()
        mcp_client._pending_actions[aid] = evt
        mcp_client._pending_resources[aid] = resource
        return evt

    def _work(self, running=True):
        class _S:
            id, status, rounds_completed, goal = 'ab12', 'running' if running else 'completed', 3, '长任务'

        class _M:
            def list_active(self_):
                return [_S()]
        subagent._manager_instance = _M()


# ── 纯谓词 ────────────────────────────────────────────────────────────────

class TestNotifyFireSpoke(unittest.TestCase):
    """hooks.fire 的返回值里，到底有没有一个绑定真把话送到设备上。"""

    def test_empty_is_not_spoken(self):
        self.assertFalse(_notify_fire_spoke([]))
        self.assertFalse(_notify_fire_spoke(None))

    def test_resource_busy_skip_is_not_spoken(self):
        self.assertFalse(_notify_fire_spoke([{'result': {'skipped': 'resource busy'}}]))

    def test_error_is_not_spoken(self):
        self.assertFalse(_notify_fire_spoke([{'error': 'boom'}]))
        self.assertFalse(_notify_fire_spoke([{'result': {'error': 'device offline'}}]))

    def test_real_result_is_spoken(self):
        self.assertTrue(_notify_fire_spoke([{'result': {'action_id': 'speak-1'}}]))

    def test_mixed_counts_as_spoken(self):
        self.assertTrue(_notify_fire_spoke([
            {'result': {'skipped': 'resource busy'}},
            {'result': {'action_id': 'speak-1'}}]))


class TestBuildNarrationMessages(unittest.TestCase):
    FROZEN = {'role': 'system', 'content': 'you are a robot'}

    def _build(self, context, last='', budget=6000):
        return _build_narration_messages(frozen_system=self.FROZEN, context=context,
                                         last_report_text=last, budget_chars=budget)

    def test_shape_and_system_reuse(self):
        msgs = self._build('ctx')
        self.assertEqual(len(msgs), 2)
        # 同一个对象：汇报继承人设，且与主循环逐字节相同 ⇒ 命中前缀缓存。
        self.assertIs(msgs[0], self.FROZEN)
        self.assertEqual(msgs[1]['role'], 'user')

    def test_keeps_tail_drops_head(self):
        # 与 _compress_turns 的 text[:30000] 相反：进度汇报关心的是近况。
        body = self._build('HEAD' + 'x' * 20000 + '\nTAILMARKER', budget=500)[1]['content']
        self.assertIn('TAILMARKER', body)
        self.assertNotIn('HEAD', body)
        self.assertIn('(前略)', body)

    def test_last_report_text_included(self):
        body = self._build('ctx', last='我已经找完客厅了')[1]['content']
        self.assertIn('我已经找完客厅了', body)

    def test_no_last_report_line_when_empty(self):
        self.assertNotIn('不要重复你上次', self._build('ctx')[1]['content'])


class TestFeedbackMessage(unittest.TestCase):
    def test_shape(self):
        m = _narration_feedback_message('已经找完客厅了，接下来去卧室')
        self.assertEqual(m['role'], 'user')
        self.assertIn('source=narration', m['content'])
        self.assertIn('已经找完客厅了，接下来去卧室', m['content'])


# ── 三个原语 ──────────────────────────────────────────────────────────────

class TestCountdownPrimitives(_Fixture):
    def setUp(self):
        super().setUp()
        self._cfg(narration_silence_seconds=30)

    def test_start_creates_when_idle(self):
        async def go():
            self.assertIsNone(ell._silence_countdown)
            ell._start_countdown()
            return ell._silence_countdown
        self.assertIsNotNone(asyncio.run(go()))

    def test_start_does_not_replace_a_running_countdown(self):
        """「已存在不更新」——否则新任务会把上一件事已经攒下的沉默一笔勾销。"""
        async def go():
            ell._start_countdown()
            first = ell._silence_countdown
            ell._start_countdown()
            return first, ell._silence_countdown
        first, second = asyncio.run(go())
        self.assertIs(first, second)
        self.assertFalse(first.cancelled())

    def test_restart_always_replaces(self):
        async def go():
            ell._restart_countdown()
            first = ell._silence_countdown
            ell._restart_countdown()
            second = ell._silence_countdown
            await asyncio.sleep(0)
            return first, second
        first, second = asyncio.run(go())
        self.assertIsNot(first, second)
        self.assertTrue(first.cancelled() or first.done())

    def test_stop_clears(self):
        async def go():
            ell._start_countdown()
            t = ell._silence_countdown
            ell._stop_countdown()
            await asyncio.sleep(0)
            return t
        t = asyncio.run(go())
        self.assertTrue(t.cancelled() or t.done())
        self.assertIsNone(ell._silence_countdown)

    def test_threshold_zero_means_no_countdown(self):
        self._cfg(narration_silence_seconds=0)

        async def go():
            ell._start_countdown()
            ell._restart_countdown()
            return ell._silence_countdown
        self.assertIsNone(asyncio.run(go()))

    def test_no_event_loop_does_not_raise(self):
        """启动早期 / 同步测试里没有运行中的 loop，开始计时不能把调用方炸掉。"""
        ell._start_countdown()
        self.assertIsNone(ell._silence_countdown)


# ── 事件 → 动作 ───────────────────────────────────────────────────────────

class TestStateMachine(_Fixture):
    def setUp(self):
        super().setUp()
        self._register_mouth()
        self._cfg(narration_silence_seconds=30, auto_notify=True)

    def test_speaking_started_stops_countdown_while_tracked(self):
        async def go():
            ell._start_countdown()
            self._speaking()                 # 有 ACP 跟踪，嘴现在忙
            ell._on_speaking_started()
            return ell._silence_countdown
        self.assertIsNone(asyncio.run(go()))

    def test_speaking_started_without_acp_tracking_restarts(self):
        """**防永久静音**：没有 ACP 跟踪的输出永远等不到「说完」事件。

        设备不声明 x-completion 是合法的，那样 call_tool_hook 不注册 pending，嘴从来
        不「忙」，也就永远不会有「说完」来重新计时。只停不启的话播报就此静音。
        """
        async def go():
            ell._start_countdown()
            ell._on_speaking_started()       # 嘴不忙 ⇒ 这次输出已经结束
            return ell._silence_countdown
        self.assertIsNotNone(asyncio.run(go()))

    def test_speaking_finished_restarts_when_work_remains(self):
        self._work()

        async def go():
            evt = self._speaking()
            ell._on_speaking_started()
            evt.set()                        # 播完
            ell._on_speaking_finished()
            return ell._silence_countdown
        self.assertIsNotNone(asyncio.run(go()))

    def test_speaking_finished_stops_when_nothing_left(self):
        async def go():
            evt = self._speaking()
            ell._on_speaking_started()
            evt.set()
            ell._on_speaking_finished()
            return ell._silence_countdown
        self.assertIsNone(asyncio.run(go()))

    def test_speaking_finished_ignored_while_another_mouth_still_playing(self):
        """多设备：第一个说完不等于说完了。"""
        self._work()

        async def go():
            a = self._speaking('spk-a')
            self._speaking('spk-b')          # 第二个还在播
            ell._on_speaking_started()
            a.set()
            ell._on_speaking_finished()
            return ell._silence_countdown
        self.assertIsNone(asyncio.run(go()))

    def test_settle_listener_fires_for_user_facing_action(self):
        """完成回调 → 走到 _on_speaking_finished。两处完成入口都经过 mark_action_complete。"""
        self._work()

        async def go():
            self._speaking('spk-9')
            ell._on_speaking_started()
            mcp_client._pending_actions['spk-9'].set()
            ok = mcp_client.mark_action_complete('spk-9', {'status': 'completed'})
            return ok, ell._silence_countdown
        ok, timer = asyncio.run(go())
        self.assertTrue(ok)
        self.assertIsNotNone(timer)

    def test_settle_listener_ignores_non_output_action(self):
        """走路走完了不是「说完话了」，不该触发计时。"""
        self._work()

        async def go():
            mcp_client._pending_actions['move-1'] = asyncio.Event()
            mcp_client._pending_resources['move-1'] = frozenset({'legs'})
            ell._stop_countdown()
            mcp_client.mark_action_complete('move-1', {'status': 'completed'})
            return ell._silence_countdown
        self.assertIsNone(asyncio.run(go()))

    def test_timeout_and_cancel_also_settle(self):
        """**防永久静音**：超时/取消走 _forget_pending，不经过 mark_action_complete。

        不在那里通知的话，一次超时的播报之后再没有任何事件会让计时重新开始。
        """
        self._work()

        async def go():
            self._speaking('spk-t')
            ell._on_speaking_started()
            self.assertIsNone(ell._silence_countdown)
            mcp_client._forget_pending(['spk-t'], 'timeout')
            return ell._silence_countdown
        self.assertIsNotNone(asyncio.run(go()))

    def test_task_started_and_all_work_done(self):
        async def go():
            ell._on_task_started()
            started = ell._silence_countdown
            ell._on_all_work_done()
            return started, ell._silence_countdown
        started, after = asyncio.run(go())
        self.assertIsNotNone(started)
        self.assertIsNone(after)

    def test_interrupt_only_stops(self):
        """打断 = 用户在说话，不是沉默的起点。紧接着的新任务才重新计时。"""
        async def go():
            ell._start_countdown()
            ell._stop_countdown()            # 打断路径做的事
            after_interrupt = ell._silence_countdown
            ell._on_task_started()           # 新请求进来
            return after_interrupt, ell._silence_countdown
        after_interrupt, after_task = asyncio.run(go())
        self.assertIsNone(after_interrupt)
        self.assertIsNotNone(after_task)

    def test_round_boundary_invariant_restores(self):
        """轮边界兜底：不在说话却没在计时 ⇒ 补回来。正在说话时**不**补。"""
        async def go():
            ell._stop_countdown()
            if not ell._mouth_busy():        # 轮循环里那两行
                ell._start_countdown()
            restored = ell._silence_countdown

            ell._stop_countdown()
            self._speaking()
            if not ell._mouth_busy():
                ell._start_countdown()
            return restored, ell._silence_countdown
        restored, while_speaking = asyncio.run(go())
        self.assertIsNotNone(restored)
        self.assertIsNone(while_speaking)


# ── 汇报本体 ──────────────────────────────────────────────────────────────

class TestReportProgress(_Fixture):
    def setUp(self):
        super().setUp()
        self._register_mouth()
        self._cfg(auto_notify=True, narration_silence_seconds=30,
                  narration_timeout_s=5, narration_context_chars=6000)
        self._work()
        self._saved_call = client.call
        self._saved_fire = hooks.fire
        self._saved_push = ell.push_event
        # system 段的拼装不是这组用例的主题，而且它要读 prompt_system.md 和一堆 config
        # 键 —— 让状态机测试依赖那些，只会在别的模块改了全局 config 时莫名其妙地挂。
        self._saved_build = ell.prompt_mod.build_system
        ell.prompt_mod.build_system = lambda *a, **k: {'role': 'system', 'content': 'sys'}
        self.fired = []
        self.pushed = []

        async def _fire(hook_id, params=None, **kw):
            self.fired.append((hook_id, params, kw))
            return [{'result': {'action_id': 'speak-1'}}]

        async def _push(ev):
            self.pushed.append(ev)

        hooks.fire = _fire
        ell.push_event = _push
        self.inst = ell.Event.__new__(ell.Event)
        self.inst._turns = []
        self.inst._current_turn = []
        ell._event_instance = self.inst

    def tearDown(self):
        client.call = self._saved_call
        hooks.fire = self._saved_fire
        ell.push_event = self._saved_push
        ell.prompt_mod.build_system = self._saved_build
        super().tearDown()

    def _stub(self, content=None, *, raises=None, hang=False, record=None):
        async def _call(message_list, tool_list, **kw):
            if record is not None:
                record.append({'messages': message_list, 'tools': tool_list, 'kw': kw})
            if hang:
                await asyncio.sleep(30)
            if raises:
                raise raises
            return {'content': content}
        client.call = _call

    def _run(self):
        return asyncio.run(self.inst._report_progress())

    def test_happy_path(self):
        self._stub('客厅找完了，接下来去卧室')
        self._run()
        self.assertEqual(self.fired[0][0], 'on_notify')
        self.assertEqual(self.fired[0][1], {'text': '客厅找完了，接下来去卧室'})
        # 必须 barrier-aware：既不能盖过正在播的音频，也不能被下一个工具盖掉。
        self.assertTrue(self.fired[0][2].get('barrier_aware'))
        self.assertTrue(any(e['type'] == 'narration' for e in self.pushed))
        self.assertEqual(ell._last_report_text_global, '客厅找完了，接下来去卧室')

    def test_feedback_is_queued_not_appended(self):
        """回灌必须排队：定时器可能在 assistant(tool_calls) 已入列、tool 结果还没入列
        的窗口触发，当场 append 会造出 provider 会拒的消息序列。"""
        self._stub('好了')
        self._run()
        self.assertEqual(self.inst._current_turn, [])
        self.assertEqual(ell._pending_narration_feedback, ['好了'])

    def test_toolless_main_model_call(self):
        rec = []
        self._stub('好了', record=rec)
        self._run()
        self.assertEqual(rec[0]['tools'], [])
        self.assertEqual(rec[0]['messages'][0]['role'], 'system')
        self.assertIsNone(rec[0]['kw'].get('model_override'))
        self.assertIsNone(rec[0]['kw'].get('reconsider_event'))

    def test_out_of_turn_context_carries_subagent_state(self):
        rec = []
        self._stub('好了', record=rec)
        self._run()
        self.assertIn('ab12', rec[0]['messages'][1]['content'])
        self.assertIn('长任务', rec[0]['messages'][1]['content'])

    def test_skip_does_not_fire_but_restarts(self):
        """**防永久静音**：SKIP 也要重新计时，否则这条路就此断掉。"""
        self._stub('SKIP')
        ell._last_report_text_global = '上次说的'
        self._run()
        self.assertEqual(self.fired, [])
        self.assertEqual(ell._last_report_text_global, '上次说的')
        self.assertIsNotNone(ell._silence_countdown)

    def test_empty_response_restarts(self):
        self._stub('')
        self._run()
        self.assertEqual(self.fired, [])
        self.assertIsNotNone(ell._silence_countdown)

    def test_llm_failure_restarts(self):
        """**防永久静音**：失败也要重新计时。"""
        self._stub(raises=RuntimeError('endpoint down'))
        self._run()
        self.assertEqual(self.fired, [])
        self.assertIsNotNone(ell._silence_countdown)

    def test_timeout_is_bounded_and_restarts(self):
        """**防永久静音**：超时也要重新计时。"""
        self._cfg(narration_timeout_s=1)
        self._stub(hang=True)
        t0 = time.time()
        self._run()
        self.assertLess(time.time() - t0, 5)
        self.assertEqual(self.fired, [])
        self.assertIsNotNone(ell._silence_countdown)

    def test_busy_mouth_restarts_without_spending_a_call(self):
        called = []
        self._stub('不该被调用', record=called)
        self._speaking()
        self._run()
        self.assertEqual(called, [])
        self.assertEqual(self.fired, [])
        self.assertIsNotNone(ell._silence_countdown)

    def test_all_bindings_skipped_restarts(self):
        """**防永久静音**：gate 时嘴空着、真要说时被占了。"""
        self._stub('说点什么')

        async def _fire(hook_id, params=None, **kw):
            self.fired.append((hook_id, params, kw))
            return [{'result': {'skipped': 'resource busy'}}]
        hooks.fire = _fire
        ell._last_report_text_global = '旧的'
        self._run()
        self.assertEqual(ell._last_report_text_global, '旧的')
        self.assertEqual(ell._pending_narration_feedback, [])
        self.assertIsNotNone(ell._silence_countdown)

    def test_overlong_report_is_truncated(self):
        self._stub('啊' * 500)
        self._run()
        # 这段话会注册成 ACP pending，超时按长度算，下一个工具调用都得等它播完。
        self.assertLessEqual(len(self.fired[0][1]['text']), ell._NARRATION_MAX_CHARS)

    def test_no_work_stops_and_spends_nothing(self):
        """活干完了就该安静 —— 而且连 LLM 调用都不该花。"""
        called = []
        self._stub('不该被调用', record=called)
        subagent._manager_instance = None
        self._run()
        self.assertEqual(called, [])
        self.assertIsNone(ell._silence_countdown)

    def test_auto_notify_off_stops(self):
        called = []
        self._stub('不该被调用', record=called)
        skills_mod._notify_override = False
        self._run()
        self.assertEqual(called, [])
        self.assertIsNone(ell._silence_countdown)

    def test_no_bindings_stops(self):
        called = []
        self._stub('不该被调用', record=called)
        hooks._registry.clear()
        self._run()
        self.assertEqual(called, [])
        self.assertIsNone(ell._silence_countdown)

    def test_restricted_turn_work_is_not_narrated(self):
        """不可信 bot / viewer 的 turn 本来就不播任何东西，它派出去的活也不该。"""
        called = []
        self._stub('不该被调用', record=called)
        ell._last_turn_restricted = True
        self._run()
        self.assertEqual(called, [])


# ── set_progress_report ───────────────────────────────────────────────────

class TestSetProgressReport(_Fixture):
    def setUp(self):
        super().setUp()
        self._cfg(narration_silence_seconds=25)

    def _call(self, **kw):
        return asyncio.run(skills_tools.set_progress_report(**kw))

    def test_seconds_override(self):
        self._call(seconds=15)
        self.assertEqual(_narration_thresholds()[1], 15)

    def test_no_args_changes_nothing(self):
        out = self._call()
        self.assertIn('没有改动', out)
        self.assertIsNone(skills_mod.get_report_override())

    def test_zero_disables_and_stops_countdown(self):
        """**防意外静音的反面**：关掉之后没有事件会再启动计时，必须当场停掉。"""
        async def go():
            ell._start_countdown()
            out = await skills_tools.set_progress_report(seconds=0)
            return out, ell._silence_countdown
        out, timer = asyncio.run(go())
        self.assertIn('已关闭', out)
        self.assertEqual(_narration_thresholds()[1], 0)
        self.assertIsNone(timer)

    def test_reenabling_restarts_countdown(self):
        """从关闭改回非 0：没有别的事件会来启动它，工具自己要负责。"""
        async def go():
            await skills_tools.set_progress_report(seconds=0)
            await skills_tools.set_progress_report(seconds=20)
            return ell._silence_countdown
        self.assertIsNotNone(asyncio.run(go()))

    def test_restore_default(self):
        self._call(seconds=1)
        out = self._call(restore_default=True)
        self.assertIn('恢复默认', out)
        self.assertIsNone(skills_mod.get_report_override())
        self.assertEqual(_narration_thresholds()[1], 25)

    def test_negative_clamps_to_zero(self):
        self._call(seconds=-5)
        self.assertEqual(_narration_thresholds()[1], 0)

    def test_return_string_states_effective_value(self):
        self.assertIn('15 秒', self._call(seconds=15))

    def test_registers_without_raising(self):
        """_build_system_tools 的类型映射只认 {str,int,float,bool}；联合类型会在
        **启动阶段** KeyError 挂掉整个进程。"""
        d = _build_system_tools([('set_progress_report', skills_tools.set_progress_report)])
        schema = d['set_progress_report']['schema']
        self.assertEqual(schema['parameters']['required'], [])
        self.assertEqual(set(schema['parameters']['properties']), {'seconds', 'restore_default'})
        self.assertIn('主动进展汇报', schema['description'])

    def test_not_reachable_from_restricted_turns(self):
        # bot 身份可伪造；任何群里的 bot 都能关掉进度播报是不可接受的。
        self.assertFalse(_restricted_channel_tool_allowed('set_progress_report', bot_restricted=True))
        self.assertFalse(_restricted_channel_tool_allowed('set_progress_report', bot_restricted=False))


class TestSkillsNoLongerPredeclareNarration(_Fixture):
    """技能不再预先声明要不要播报 —— 播不播由 agent-core 运行时自己判断。"""

    def test_skill_toggle_does_not_clobber_set_auto_notify(self):
        asyncio.run(skills_tools.set_auto_notify(False))
        saved = config.main.get('skills', {})
        try:
            config.main['skills'] = {'installed': [{
                'slug': 'chess', 'name': '下棋', 'active': True,
                'instruction': 'x', 'oneLiner': 'y', 'narrationDefault': True}]}
            asyncio.run(skills_tools.activate_skill(slug='chess'))
            self.assertFalse(ell._auto_notify_enabled())
            asyncio.run(skills_tools.deactivate_skill(slug='chess'))
            self.assertFalse(ell._auto_notify_enabled())
        finally:
            config.main['skills'] = saved

    def test_recompute_helper_is_gone(self):
        self.assertFalse(hasattr(skills_mod, '_recompute_notify_override'))

    def test_legacy_narration_default_field_is_ignored_not_fatal(self):
        saved = config.main.get('skills', {})
        try:
            config.main['skills'] = {'installed': [{
                'slug': 'old', 'name': '旧技能', 'active': True,
                'instruction': 'x', 'oneLiner': 'y', 'narrationDefault': False}]}
            self.assertEqual([s['slug'] for s in skills_mod.visible_skills()], ['old'])
        finally:
            config.main['skills'] = saved


class TestRoundsArmIsGone(unittest.TestCase):
    """轮数维度已删除 —— 纯时间触发，定时器全局负责。"""

    def test_helpers_removed(self):
        for name in ('_narration_decision', '_narration_threshold_hit',
                     '_maybe_report_progress', 'silent_seconds', '_note_user_output'):
            self.assertFalse(hasattr(ell, name), f'{name} 应该已经删掉')

    def test_config_key_removed(self):
        src = pathlib.Path(__file__).resolve().parents[1] / 'src'
        for rel in ('config.py', 'start.py', 'api/mcp_manage.py'):
            self.assertNotIn('narration_silence_rounds', (src / rel).read_text(),
                             f'{rel} 里还有轮数配置项')


if __name__ == '__main__':
    unittest.main()

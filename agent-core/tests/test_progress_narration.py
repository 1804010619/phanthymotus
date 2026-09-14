"""
test_progress_narration.py — 框架级主动播报（progress narration）。

背景：主动播报原本完全建立在「LLM 自愿写 content，系统自动播出去」之上
（event/llm.py 里那段 hooks.fire('on_notify', {'text': content})）。两个实测问题：
prompt 约束不住，多轮工具调用期间模型经常一句话不写；写了也不合适，那是内部推理而
不是说给等着的人听的进度汇报。

现在改成框架保证：turn 内统计连续无面向用户输出的**轮数/秒数**，越过阈值就跑一次
跳出 agent loop 的一次性 toolless LLM 调用，专门生成一句进度汇报，再经同一个
on_notify 播出去，并回灌进 turn_messages 免得主 LLM 再说一遍。content 自动播报同时
废除，因此另加一条哑火护栏（见 test_on_notify_barrier.py 里的回归断言）。

每个判断都下沉成模块级纯函数，理由和 _turn_ends_on_finish 在 llm.py 里记的一样：
不用起 prompt / client / registry / DB 就能测。

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_progress_narration.py
"""
import asyncio
import os
import pathlib
import sys
import tempfile
import time
import typing
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import config  # noqa: E402
import hooks  # noqa: E402
import mcp_client  # noqa: E402
import event.skills as skills_tools  # noqa: E402
# event/__init__.py rebinds the `event.skills` *attribute* to a Tools() instance, so the
# name above gives you the tool methods but not the module globals — those must come from
# sys.modules. Same reason event/llm.py reaches get_notify_override through sys.modules.
skills_mod = sys.modules['event.skills']
from event.llm import (  # noqa: E402
    _auto_notify_enabled,
    _build_narration_messages,
    _build_system_tools,
    _narration_decision,
    _narration_feedback_message,
    _narration_threshold_hit,
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


class _NarrationFixture(unittest.TestCase):
    """Swap the global registry/hooks/pendings and the two runtime overrides."""

    def setUp(self):
        self._saved_registry = dict(mcp_client.registry)
        self._saved_hooks = dict(hooks._registry)
        mcp_client.registry.clear()
        hooks._registry.clear()
        self._saved_pending = (
            dict(mcp_client._pending_actions), dict(mcp_client._pending_tools),
            dict(mcp_client._pending_resources), dict(mcp_client._pending_timeouts),
        )
        for d in (mcp_client._pending_actions, mcp_client._pending_tools,
                  mcp_client._pending_resources, mcp_client._pending_timeouts):
            d.clear()
        self._saved_notify = skills_mod._notify_override
        self._saved_report = skills_mod._report_override
        skills_mod._notify_override = None
        skills_mod._report_override = None
        self._saved_event = config.main.get('event', {})

    def tearDown(self):
        mcp_client.registry.clear()
        mcp_client.registry.update(self._saved_registry)
        hooks._registry.clear()
        hooks._registry.update(self._saved_hooks)
        targets = (mcp_client._pending_actions, mcp_client._pending_tools,
                   mcp_client._pending_resources, mcp_client._pending_timeouts)
        for d, saved in zip(targets, self._saved_pending):
            d.clear()
            d.update(saved)
        skills_mod._notify_override = self._saved_notify
        skills_mod._report_override = self._saved_report
        config.main['event'] = self._saved_event

    # -- helpers ---------------------------------------------------------
    def _register_mouth(self, *, resource=frozenset({'mouth'})):
        mcp_client.registry['dev1'] = {
            'name': 'dev1', 'url': 'http://dev1', 'online': True,
            'tools': ['tts'],
            'schemas': {'mcp__dev1__tts__speak': {'name': 'mcp__dev1__tts__speak'}},
            'tool_meta': {'mcp__dev1__tts__speak': {**MOUTH_META, 'resource': resource}},
            'split_map': {'mcp__dev1__tts__speak': {'tool': 'tts', 'action': 'speak'}},
            'tool_groups': {}, 'input_schemas': {},
        }
        hooks.register('dev1', 'tts', {'on_notify': {'action': 'speak'}})

    def _set_llm_cfg(self, **kw):
        ev = dict(config.main.get('event', {}))
        llm = dict(ev.get('llm', {}))
        llm.update(kw)
        ev['llm'] = llm
        config.main['event'] = ev


# ── 纯谓词 ────────────────────────────────────────────────────────────────

class TestNotifyFireSpoke(unittest.TestCase):
    """hooks.fire 的返回值里，到底有没有一个绑定真把话送到设备上。"""

    def test_empty_is_not_spoken(self):
        self.assertFalse(_notify_fire_spoke([]))
        self.assertFalse(_notify_fire_spoke(None))

    def test_resource_busy_skip_is_not_spoken(self):
        # barrier_aware=True 在嘴被占用时返回这个字面量 —— 用户没听到新东西。
        self.assertFalse(_notify_fire_spoke([{'result': {'skipped': 'resource busy'}}]))

    def test_error_is_not_spoken(self):
        self.assertFalse(_notify_fire_spoke([{'error': 'boom'}]))
        self.assertFalse(_notify_fire_spoke([{'result': {'error': 'device offline'}}]))

    def test_real_result_is_spoken(self):
        self.assertTrue(_notify_fire_spoke([{'result': {'action_id': 'speak-1'}}]))

    def test_mixed_counts_as_spoken(self):
        self.assertTrue(_notify_fire_spoke([
            {'result': {'skipped': 'resource busy'}},
            {'result': {'action_id': 'speak-1'}},
        ]))


class TestThresholdHit(unittest.TestCase):
    """轮数 / 秒数，先到者触发。两个维度互不替代。"""

    def test_rounds_arm_alone(self):
        self.assertFalse(_narration_threshold_hit(3, 0, 4, 0))
        self.assertTrue(_narration_threshold_hit(4, 0, 4, 0))

    def test_seconds_arm_alone(self):
        self.assertFalse(_narration_threshold_hit(1, 24, 0, 25))
        self.assertTrue(_narration_threshold_hit(1, 25, 0, 25))

    def test_seconds_fires_when_rounds_far_from_threshold(self):
        # 「一步就很久」：2 轮 × 40 秒，轮数阈值 4 永远够不着，秒数阈值兜住。
        self.assertTrue(_narration_threshold_hit(2, 80, 4, 25))

    def test_rounds_fires_when_seconds_far_from_threshold(self):
        # 「步骤多但每步快」：4 轮 × 3 秒，才 12 秒，但用户已经看不到动静了。
        self.assertTrue(_narration_threshold_hit(4, 12, 4, 25))

    def test_both_zero_never_fires(self):
        self.assertFalse(_narration_threshold_hit(999, 9999, 0, 0))

    def test_negative_thresholds_never_fire(self):
        self.assertFalse(_narration_threshold_hit(999, 9999, -1, -1))


class TestAutoNotifyEnabled(_NarrationFixture):
    def test_follows_config_when_no_override(self):
        self._set_llm_cfg(auto_notify=True)
        self.assertTrue(_auto_notify_enabled())
        self._set_llm_cfg(auto_notify=False)
        self.assertFalse(_auto_notify_enabled())

    def test_override_dominates_config(self):
        self._set_llm_cfg(auto_notify=True)
        skills_mod._notify_override = False
        self.assertFalse(_auto_notify_enabled())
        self._set_llm_cfg(auto_notify=False)
        skills_mod._notify_override = True
        self.assertTrue(_auto_notify_enabled())

    def test_none_override_falls_through(self):
        self._set_llm_cfg(auto_notify=False)
        skills_mod._notify_override = None
        self.assertFalse(_auto_notify_enabled())


class TestNarrationThresholds(_NarrationFixture):
    def test_db_only(self):
        self._set_llm_cfg(narration_silence_rounds=4, narration_silence_seconds=25)
        self.assertEqual(_narration_thresholds(), (4, 25))

    def test_override_rounds_only_keeps_db_seconds(self):
        self._set_llm_cfg(narration_silence_rounds=4, narration_silence_seconds=25)
        skills_mod._report_override = {'rounds': 2}
        self.assertEqual(_narration_thresholds(), (2, 25))

    def test_override_both(self):
        self._set_llm_cfg(narration_silence_rounds=4, narration_silence_seconds=25)
        skills_mod._report_override = {'rounds': 10, 'seconds': 120}
        self.assertEqual(_narration_thresholds(), (10, 120))

    def test_cleared_override_falls_back_to_db(self):
        self._set_llm_cfg(narration_silence_rounds=7, narration_silence_seconds=70)
        skills_mod._report_override = {'rounds': 1}
        skills_mod._report_override = None
        self.assertEqual(_narration_thresholds(), (7, 70))


class TestHookQueries(_NarrationFixture):
    def test_has_bindings(self):
        self.assertFalse(hooks.has_bindings('on_notify'))
        self._register_mouth()
        self.assertTrue(hooks.has_bindings('on_notify'))

    def test_only_interrupt_bindings_is_not_notify(self):
        hooks.register('dev1', 'tts', {'on_interrupt_all': {'action': 'interrupt'}})
        self.assertFalse(hooks.has_bindings('on_notify'))

    def test_busy_when_mouth_held(self):
        self._register_mouth()
        self.assertFalse(hooks.notify_resource_busy('on_notify'))
        mcp_client._pending_actions['speak-1'] = asyncio.Event()
        mcp_client._pending_resources['speak-1'] = frozenset({'mouth'})
        self.assertTrue(hooks.notify_resource_busy('on_notify'))

    def test_binding_without_declared_resource_is_never_busy(self):
        # call_tool_hook 会无条件派发这种绑定，这里声称它忙就会白白压掉一次能被
        # 听到的播报。
        self._register_mouth(resource=None)
        mcp_client._pending_actions['speak-1'] = asyncio.Event()
        mcp_client._pending_resources['speak-1'] = frozenset({'mouth'})
        self.assertFalse(hooks.notify_resource_busy('on_notify'))

    def test_no_bindings_is_not_busy(self):
        self.assertFalse(hooks.notify_resource_busy('on_notify'))

    def test_completed_but_unreaped_pending_is_not_busy(self):
        """Orin5 实测的那个 bug。

        完成回调只做 `_pending_actions[aid].set()`，故意不删这一项（晚到的 waiter 还要
        读结果），真正的删除要等某个 barrier 来回收。所以一句早就播完的话会一直挂在
        _pending_actions 里。按 key 判断"忙不忙"的话，嘴会被锁到下一次 barrier 为止 ——
        现场是 17:03:24 播完、17:05:58 才回收，中间 2 分 34 秒的自动播报全被静默跳过。
        """
        self._register_mouth()
        ev = asyncio.Event()
        mcp_client._pending_actions['speak-1'] = ev
        mcp_client._pending_resources['speak-1'] = frozenset({'mouth'})
        self.assertTrue(hooks.notify_resource_busy('on_notify'))   # 播放中
        ev.set()                                                   # 完成回调到达
        self.assertFalse(hooks.notify_resource_busy('on_notify'))  # 嘴已经空了
        # 该项仍在 dict 里 —— 回收是 barrier 的事，不是这个判断的事。
        self.assertIn('speak-1', mcp_client._pending_actions)

    def test_resource_actually_busy_ignores_completed(self):
        want = frozenset({'mouth'})
        ev = asyncio.Event()
        mcp_client._pending_actions['a'] = ev
        mcp_client._pending_resources['a'] = want
        self.assertTrue(mcp_client.resource_actually_busy(want))
        self.assertEqual(mcp_client.conflicting_pending(want), ['a'])
        ev.set()
        self.assertFalse(mcp_client.resource_actually_busy(want))
        # conflicting_pending 的语义不变 —— barrier 仍要靠它找到该等/该回收的项。
        self.assertEqual(mcp_client.conflicting_pending(want), ['a'])

    def test_one_in_flight_among_completed_is_still_busy(self):
        want = frozenset({'mouth'})
        done, live = asyncio.Event(), asyncio.Event()
        done.set()
        mcp_client._pending_actions.update({'done': done, 'live': live})
        mcp_client._pending_resources.update({'done': want, 'live': want})
        self.assertTrue(mcp_client.resource_actually_busy(want))


class TestBuildNarrationMessages(unittest.TestCase):
    FROZEN = {'role': 'system', 'content': 'you are a robot'}

    def _turn(self, n):
        return '\n'.join(f'[user] msg{i} ' + 'x' * 200 for i in range(n))

    def test_shape_and_system_reuse(self):
        msgs = _build_narration_messages(self.FROZEN, self._turn(2), '', 4, 30.0, 6000)
        self.assertEqual(len(msgs), 2)
        # 同一个对象：汇报继承人设，且与主循环逐字节相同 ⇒ 命中前缀缓存。
        self.assertIs(msgs[0], self.FROZEN)
        self.assertEqual(msgs[1]['role'], 'user')

    def test_keeps_tail_drops_head(self):
        # 与 _compress_turns 的 text[:30000] 相反：进度汇报关心的是近况。
        turn = 'HEAD' + 'x' * 20000 + '\nTAILMARKER'
        body = _build_narration_messages(self.FROZEN, turn, '', 4, 30.0, 500)[1]['content']
        self.assertIn('TAILMARKER', body)
        self.assertNotIn('HEAD', body)
        self.assertIn('(前略)', body)

    def test_last_report_text_included(self):
        body = _build_narration_messages(
            self.FROZEN, self._turn(1), '我已经找完客厅了', 4, 30.0, 6000)[1]['content']
        self.assertIn('我已经找完客厅了', body)

    def test_no_last_report_line_when_empty(self):
        body = _build_narration_messages(self.FROZEN, self._turn(1), '', 4, 30.0, 6000)[1]['content']
        self.assertNotIn('不要重复你上次', body)

    def test_rounds_and_seconds_rendered(self):
        body = _build_narration_messages(self.FROZEN, self._turn(1), '', 7, 42.6, 6000)[1]['content']
        self.assertIn('7 轮', body)
        self.assertIn('42', body)


class TestFeedbackMessage(unittest.TestCase):
    def test_shape(self):
        m = _narration_feedback_message('已经找完客厅了，接下来去卧室')
        self.assertEqual(m['role'], 'user')
        self.assertIn('source=narration', m['content'])
        self.assertIn('已经找完客厅了，接下来去卧室', m['content'])


# ── Gate 表 ───────────────────────────────────────────────────────────────

class TestNarrationDecision(unittest.TestCase):
    BASE = dict(now=1000.0, rounds_thr=4, seconds_thr=25, tool_restricted=False,
                auto_notify=True, has_bindings=True, busy=False, cancelled=False)

    def _d(self, rounds=4, seconds=0.0, **kw):
        return _narration_decision(rounds, seconds, **{**self.BASE, **kw})

    def test_reports_when_all_gates_pass(self):
        self.assertEqual(self._d(), 'report')

    def test_disabled_when_both_thresholds_zero(self):
        self.assertEqual(self._d(rounds_thr=0, seconds_thr=0), 'skip')

    def test_tool_restricted_skips(self):
        # 受限 turn（不可信 bot / viewer）本来就不播任何东西。
        self.assertEqual(self._d(tool_restricted=True), 'skip')

    def test_auto_notify_off_skips(self):
        self.assertEqual(self._d(auto_notify=False), 'skip')

    def test_no_bindings_skips(self):
        self.assertEqual(self._d(has_bindings=False), 'skip')

    def test_cancelled_skips(self):
        self.assertEqual(self._d(cancelled=True), 'skip')

    def test_busy_is_its_own_outcome(self):
        # 与普通 skip 区分：一次 LLM 调用都没花，计数器该原样留着下轮再试。
        self.assertEqual(self._d(busy=True), 'skip_busy')

    def test_rounds_boundary(self):
        self.assertEqual(self._d(rounds=3, seconds=0), 'skip')
        self.assertEqual(self._d(rounds=4, seconds=0), 'report')

    def test_seconds_boundary(self):
        self.assertEqual(self._d(rounds=1, seconds=24.9), 'skip')
        self.assertEqual(self._d(rounds=1, seconds=25.0), 'report')

    def test_seconds_arm_fires_while_rounds_arm_has_not(self):
        """本次新增语义的关键用例：慢轮场景靠秒数兜底。"""
        self.assertEqual(self._d(rounds=2, seconds=80), 'report')

    def test_busy_checked_after_threshold(self):
        # 阈值没到时嘴忙不忙都无所谓，不该报成 skip_busy。
        self.assertEqual(self._d(rounds=1, seconds=0, busy=True), 'skip')


# ── set_progress_report 工具 ──────────────────────────────────────────────

class TestSetProgressReportTool(_NarrationFixture):
    def setUp(self):
        super().setUp()
        self._set_llm_cfg(narration_silence_rounds=4, narration_silence_seconds=25)

    def _call(self, **kw):
        return asyncio.run(skills_tools.set_progress_report(**kw))

    def test_rounds_only(self):
        self._call(rounds=2)
        self.assertEqual(_narration_thresholds(), (2, 25))

    def test_accumulates_rather_than_replaces(self):
        # 只传 seconds 不该把之前设好的 rounds 冲回默认。
        self._call(rounds=2)
        self._call(seconds=120)
        self.assertEqual(_narration_thresholds(), (2, 120))

    def test_no_args_changes_nothing(self):
        out = self._call()
        self.assertIn('没有改动', out)
        self.assertIsNone(skills_mod.get_report_override())
        self.assertEqual(_narration_thresholds(), (4, 25))

    def test_zero_zero_disables(self):
        out = self._call(rounds=0, seconds=0)
        self.assertIn('已关闭', out)
        self.assertEqual(_narration_thresholds(), (0, 0))
        self.assertEqual(
            _narration_decision(999, 9999.0, now=0.0, rounds_thr=0, seconds_thr=0,
                                tool_restricted=False, auto_notify=True,
                                has_bindings=True, busy=False, cancelled=False),
            'skip')

    def test_restore_default(self):
        self._call(rounds=1, seconds=2)
        out = self._call(restore_default=True)
        self.assertIn('恢复默认', out)
        self.assertIsNone(skills_mod.get_report_override())
        self.assertEqual(_narration_thresholds(), (4, 25))

    def test_negative_clamps_to_zero(self):
        self._call(rounds=-5)
        self.assertEqual(_narration_thresholds()[0], 0)

    def test_return_string_states_effective_values(self):
        # 模型要能直接转述给用户，不用再跑一轮去查。
        out = self._call(rounds=2, seconds=15)
        self.assertIn('2 轮', out)
        self.assertIn('15 秒', out)

    def test_registers_without_raising(self):
        """_build_system_tools 的类型映射只认 {str,int,float,bool}。

        写成 `int | None` 会在注册时 KeyError —— 那发生在**启动阶段**，整个进程起不来，
        所以这条护栏必须有。
        """
        d = _build_system_tools([('set_progress_report', skills_tools.set_progress_report)])
        schema = d['set_progress_report']['schema']
        self.assertEqual(schema['parameters']['required'], [])
        self.assertEqual(set(schema['parameters']['properties']),
                         {'rounds', 'seconds', 'restore_default'})
        self.assertEqual(schema['parameters']['properties']['rounds']['type'], 'integer')
        self.assertIn('主动进展汇报', schema['description'])

    def test_not_reachable_from_restricted_turns(self):
        # bot 身份可伪造；任何群里的 bot 都能关掉进度播报是不可接受的。
        self.assertFalse(_restricted_channel_tool_allowed('set_progress_report',
                                                          bot_restricted=True))
        self.assertFalse(_restricted_channel_tool_allowed('set_progress_report',
                                                          bot_restricted=False))


class TestSkillsNoLongerPredeclareNarration(_NarrationFixture):
    """技能不再预先声明要不要播报 —— 播不播由 agent-core 运行时自己判断。"""

    def test_skill_toggle_does_not_clobber_set_auto_notify(self):
        # 以前 _recompute_notify_override 会在技能激活/停用时重算这个覆盖，把模型
        # 刚设好的闭麦状态冲掉。
        asyncio.run(skills_tools.set_auto_notify(False))
        saved = config.main.get('skills', {})
        try:
            config.main['skills'] = {'installed': [{
                'slug': 'chess', 'name': '下棋', 'active': True,
                'instruction': 'x', 'oneLiner': 'y', 'narrationDefault': True,
            }]}
            asyncio.run(skills_tools.activate_skill(slug='chess'))
            self.assertFalse(_auto_notify_enabled())
            asyncio.run(skills_tools.deactivate_skill(slug='chess'))
            self.assertFalse(_auto_notify_enabled())
        finally:
            config.main['skills'] = saved

    def test_recompute_helper_is_gone(self):
        self.assertFalse(hasattr(skills_tools, '_recompute_notify_override'))

    def test_legacy_narration_default_field_is_ignored_not_fatal(self):
        """库里存量技能仍带这个键 —— 加载时忽略即可，不能抛。"""
        saved = config.main.get('skills', {})
        try:
            config.main['skills'] = {'installed': [{
                'slug': 'old', 'name': '旧技能', 'active': True,
                'instruction': 'x', 'oneLiner': 'y', 'narrationDefault': False,
            }]}
            self.assertEqual([s['slug'] for s in skills_mod.visible_skills()], ['old'])
        finally:
            config.main['skills'] = saved


# ── 汇报器（异步） ────────────────────────────────────────────────────────

class _FakeEvent:
    """真正的 Event 实例，只是不跑 __init__ —— 汇报这条路只用到 _turns 和两个方法。"""

    def __init__(self):
        import sys as _sys
        ell = _sys.modules['event.llm']          # the *module*; the attribute is an Event()
        self._ev = ell.Event.__new__(ell.Event)
        self._ev._turns = []
        self._ell = ell

    def run(self, turn_messages, **kw):
        frozen = {'role': 'system', 'content': 'sys'}
        ell = self._ell
        # 时钟是模块级的：把上次输出推到很久以前，让秒数臂满足。
        ell._last_output_ts = time.time() - kw.pop('silent_for', 100)
        ell._last_report_text_global = kw.pop('last_report_text', '')
        if kw.pop('threshold_not_reached', False):
            ell._last_output_ts = time.time()
        return asyncio.run(self._ev._maybe_report_progress(
            turn_messages, frozen, kw.pop('silent_rounds', 4),
            tool_restricted=kw.pop('tool_restricted', False),
            cancel_event=kw.pop('cancel_event', None)))


class TestReporter(_NarrationFixture):
    def setUp(self):
        super().setUp()
        self._register_mouth()
        self._set_llm_cfg(auto_notify=True, narration_silence_rounds=4,
                          narration_silence_seconds=25, narration_timeout_s=5,
                          narration_context_chars=6000)
        import client
        self._ell = sys.modules['event.llm']
        self._client = client
        self._saved_call = client.call
        self._saved_fire = hooks.fire
        self._saved_push = self._ell.push_event
        self._saved_ts = self._ell._last_output_ts
        self._saved_txt = self._ell._last_report_text_global
        self.fired = []
        self.pushed = []

        async def _fire(hook_id, params=None, **kw):
            self.fired.append((hook_id, params, kw))
            return [{'result': {'action_id': 'speak-1'}}]

        async def _push(ev):
            self.pushed.append(ev)

        hooks.fire = _fire
        self._ell.push_event = _push

    def tearDown(self):
        self._ell._cancel_narration_timer()
        self._client.call = self._saved_call
        hooks.fire = self._saved_fire
        self._ell.push_event = self._saved_push
        self._ell._last_output_ts = self._saved_ts
        self._ell._last_report_text_global = self._saved_txt
        super().tearDown()

    def _stub_call(self, content=None, *, raises=None, hang=False, record=None):
        async def _call(message_list, tool_list, **kw):
            if record is not None:
                record.append({'messages': message_list, 'tools': tool_list, 'kw': kw})
            if hang:
                await asyncio.sleep(30)
            if raises:
                raise raises
            return {'content': content}
        self._client.call = _call

    def test_happy_path_fires_and_feeds_back(self):
        self._stub_call('客厅找完了，接下来去卧室')
        msgs = []
        self.assertTrue(_FakeEvent().run(msgs))
        self.assertEqual(self.fired[0][0], 'on_notify')
        self.assertEqual(self.fired[0][1], {'text': '客厅找完了，接下来去卧室'})
        # 必须走 barrier-aware：既不能盖过正在播的音频，也不能被下一个工具盖掉。
        self.assertTrue(self.fired[0][2].get('barrier_aware'))
        # 回灌，否则主 LLM 下一轮很可能显式调播报工具再说一遍。
        self.assertEqual(len(msgs), 1)
        self.assertIn('source=narration', msgs[0]['content'])
        self.assertTrue(any(e['type'] == 'narration' for e in self.pushed))
        self.assertEqual(self._ell._last_report_text_global, '客厅找完了，接下来去卧室')

    def test_clock_is_not_reset_at_broadcast_start(self):
        """播报刚发出时沉默还没开始 —— 起点是它**播完**的那一刻。

        在这里就把时钟清零的话，一段 134 秒的播报刚说完就已经"沉默 0 秒"，下一次判断
        会立刻又满足阈值。真正的锚点是 ACP 完成回调。
        """
        self._stub_call('我在找')

        # 模拟 call_tool_hook：播报发出后注册一个 ACP pending（设备声明了 x-completion）。
        async def _fire(hook_id, params=None, **kw):
            self.fired.append((hook_id, params, kw))
            mcp_client._pending_actions['spk'] = asyncio.Event()
            mcp_client._pending_resources['spk'] = frozenset({'mouth'})
            return [{'result': {'action_id': 'spk'}}]
        hooks.fire = _fire

        self.assertTrue(_FakeEvent().run([], silent_for=100))
        # 正在播 ⇒ 沉默还没开始，时钟冻结在 0，而不是被"开始播"清零后继续跑。
        self.assertEqual(self._ell.silent_seconds(), 0.0)
        mcp_client.mark_action_complete('spk', {'status': 'completed'})
        # 播完那一刻才是沉默的起点。
        self.assertLess(self._ell.silent_seconds(), 5)

    def test_toolless_main_model_call(self):
        """跳出 agent loop 的一次性调用：无工具、system 段复用、不指定别的模型。"""
        rec = []
        self._stub_call('好了', record=rec)
        _FakeEvent().run([])
        self.assertEqual(rec[0]['tools'], [])
        self.assertEqual(rec[0]['messages'][0]['role'], 'system')
        self.assertIsNone(rec[0]['kw'].get('model_override'))
        # steer 不该作废一次本来就很短的汇报；它下一轮会被排空。
        self.assertIsNone(rec[0]['kw'].get('reconsider_event'))

    def test_skip_does_not_fire_and_backs_off(self):
        self._stub_call('SKIP')
        msgs = []
        self.assertFalse(_FakeEvent().run(msgs, last_report_text='上次说的'))
        self.assertEqual(self.fired, [])
        self.assertEqual(msgs, [])
        self.assertEqual(self._ell._last_report_text_global, '上次说的')   # 别丢
        self.assertLess(self._ell.silent_seconds(), 5)                    # 退避一整个间隔

    def test_empty_response_does_not_fire(self):
        self._stub_call('')
        self.assertFalse(_FakeEvent().run([]))
        self.assertEqual(self.fired, [])

    def test_llm_failure_backs_off_a_full_interval(self):
        """失败后必须退满一个间隔 —— 否则定时器会被 max(1.0,...) 兜成每秒重试。"""
        self._stub_call(raises=RuntimeError('endpoint down'))
        self.assertFalse(_FakeEvent().run([]))
        self.assertEqual(self.fired, [])
        self.assertLess(self._ell.silent_seconds(), 5)

    def test_timeout_is_bounded(self):
        self._set_llm_cfg(narration_timeout_s=1)
        self._stub_call(hang=True)
        t0 = time.time()
        self.assertFalse(_FakeEvent().run([]))
        self.assertLess(time.time() - t0, 5)
        self.assertEqual(self.fired, [])

    def test_overlong_report_is_truncated(self):
        self._stub_call('啊' * 500)
        self.assertTrue(_FakeEvent().run([]))
        # 这段话会注册成 ACP pending，超时按长度算，下一个工具调用都得等它播完。
        self.assertLessEqual(len(self.fired[0][1]['text']), self._ell._NARRATION_MAX_CHARS)

    def test_busy_mouth_skips_before_spending_an_llm_call(self):
        called = []
        self._stub_call('不该被调用', record=called)
        mcp_client._pending_actions['speak-1'] = asyncio.Event()
        mcp_client._pending_resources['speak-1'] = frozenset({'mouth'})
        self.assertFalse(_FakeEvent().run([]))
        self.assertEqual(called, [])
        self.assertEqual(self.fired, [])

    def test_all_bindings_skipped_is_not_spoken(self):
        """竞态：前置检查时嘴还空着，真要说的时候被占了。"""
        self._stub_call('说点什么')

        async def _fire(hook_id, params=None, **kw):
            self.fired.append((hook_id, params, kw))
            return [{'result': {'skipped': 'resource busy'}}]
        hooks.fire = _fire
        msgs = []
        self.assertFalse(_FakeEvent().run(msgs, last_report_text='旧的'))
        self.assertEqual(self._ell._last_report_text_global, '旧的')
        self.assertEqual(msgs, [])      # 没播出去就不该回灌

    def test_threshold_not_reached_does_nothing(self):
        called = []
        self._stub_call('不该被调用', record=called)
        self.assertFalse(_FakeEvent().run([], silent_rounds=1, threshold_not_reached=True))
        self.assertEqual(called, [])

    def test_auto_notify_off_does_nothing(self):
        called = []
        self._stub_call('不该被调用', record=called)
        skills_mod._notify_override = False
        self.assertFalse(_FakeEvent().run([]))
        self.assertEqual(called, [])

    def test_tool_restricted_does_nothing(self):
        called = []
        self._stub_call('不该被调用', record=called)
        self.assertFalse(_FakeEvent().run([], tool_restricted=True))
        self.assertEqual(called, [])

    def test_no_bindings_does_nothing(self):
        called = []
        self._stub_call('不该被调用', record=called)
        hooks._registry.clear()
        self.assertFalse(_FakeEvent().run([]))
        self.assertEqual(called, [])

    def test_verbal_retune_takes_effect_immediately(self):
        """「多汇报一点」发生在当前 turn 的某一轮，下一轮就得按新阈值判。"""
        called = []
        self._stub_call('好', record=called)
        self.assertFalse(_FakeEvent().run([], silent_rounds=1, threshold_not_reached=True))
        asyncio.run(skills_tools.set_progress_report(rounds=1))
        self.assertTrue(_FakeEvent().run([], silent_rounds=1, threshold_not_reached=True))


class TestSilenceWatchdog(_NarrationFixture):
    """以播报为锚的定时器 —— 轮循环那套采样不到的两种情况。

    Orin5 实测：主 agent 17:22:31 派了异步子代理，17:22:48 就 finish 了，子代理独自跑了
    2 分半 12 轮，期间零播报 —— 计数器是 _one_turn 的局部变量，没有 turn 就没有轮，那段
    检查一次都不执行。另一次是单轮跑了 3 分半，轮间隙压根没到来。
    """

    def setUp(self):
        super().setUp()
        self._ell = sys.modules['event.llm']
        self._saved_ts = self._ell._last_output_ts
        self._saved_report = self._ell._last_report_text_global

    def tearDown(self):
        self._ell._cancel_narration_timer()
        self._ell._last_output_ts = self._saved_ts
        self._ell._last_report_text_global = self._saved_report
        super().tearDown()

    def test_note_output_resets_the_clock(self):
        self._ell._last_output_ts = time.time() - 300
        self.assertGreater(self._ell.silent_seconds(), 290)
        asyncio.run(self._noop_note())
        self.assertLess(self._ell.silent_seconds(), 5)

    async def _noop_note(self):
        self._ell._note_user_output()

    def test_arming_twice_keeps_only_the_latest(self):
        """"新播报取消旧定时" —— 否则几次输出会排出一串定时器，各自到点各播一句。"""
        async def go():
            self._ell._arm_narration_timer()
            first = self._ell._narration_timer
            self._ell._arm_narration_timer()
            second = self._ell._narration_timer
            await asyncio.sleep(0)
            return first, second
        first, second = asyncio.run(go())
        self.assertIsNot(first, second)
        self.assertTrue(first.cancelled() or first.done())

    def test_no_active_work_spends_nothing(self):
        """活干完了就该安静 —— 而且连 LLM 调用都不该花。下次出声会重新上表。"""
        self._register_mouth()
        self._set_llm_cfg(auto_notify=True, narration_silence_seconds=15)
        self.assertEqual(self._ell._active_work_summary(), [],
                         '这个 fixture 里不该有活在跑')

        import client
        called = []
        saved_call, saved_fire = client.call, hooks.fire

        async def _call(**kw):
            called.append(1)
            return {'content': 'x'}

        async def _fire(*a, **k):
            called.append('fire')
            return []

        client.call, hooks.fire = _call, _fire
        try:
            inst = self._ell.Event.__new__(self._ell.Event)
            inst._turns = []
            asyncio.run(inst._report_progress_out_of_turn())
        finally:
            client.call, hooks.fire = saved_call, saved_fire
        self.assertEqual(called, [])

    def test_active_work_lists_running_subagents(self):
        class _S:
            id, status, rounds_completed, goal = 'ab12', 'running', 7, '写一份英伟达投研报告'

        import subagent
        saved = subagent._manager_instance
        class _M:
            def list_active(self_): return [_S()]
        subagent._manager_instance = _M()
        try:
            work = self._ell._active_work_summary()
        finally:
            subagent._manager_instance = saved
        self.assertEqual(len(work), 1)
        self.assertIn('ab12', work[0])
        self.assertIn('7 轮', work[0])
        self.assertIn('英伟达', work[0])

    def test_report_text_survives_across_turns(self):
        """turn 结束后子代理接着跑，是同一件事的延续 —— "别重复上次说的"要跨 turn 成立。"""
        self._ell._remember_report('已经查完财报了')
        self.assertEqual(self._ell._last_report_text_global, '已经查完财报了')

    def test_report_without_acp_tracking_does_not_spin(self):
        """播报没注册 ACP pending 时，时钟必须由播报自己锚住。

        设备不声明 x-completion 是合法的，那样 call_tool_hook 就不会注册 pending，
        嘴永远不"忙"，也永远不会有完成回调。如果播完既不锚时钟又指望别人来锚，时钟就
        一直停在"早就超阈"，定时器每次醒来都判定该播 —— 实测 0.6 秒里播了 250 次。
        """
        self._register_mouth()
        self._set_llm_cfg(auto_notify=True, narration_silence_seconds=2,
                          narration_silence_rounds=0, narration_timeout_s=5)
        import client
        saved_call, saved_fire, saved_push = client.call, hooks.fire, self._ell.push_event
        fired = []

        async def _fire(h, p=None, **k):
            fired.append(p['text'])
            return [{'result': {'ok': True}}]      # 没有 action_id ⇒ 不注册 pending

        async def _call(**kw):
            return {'content': '进度'}

        async def _push(e):
            pass

        client.call, hooks.fire, self._ell.push_event = _call, _fire, _push
        try:
            inst = self._ell.Event.__new__(self._ell.Event)
            inst._turns = []
            self._ell._last_output_ts = time.time() - 300
            asyncio.run(inst._emit_progress_report(
                {'role': 'system', 'content': 's'}, 'ctx',
                silent_rounds=0, cancel_event=None, label=''))
        finally:
            client.call, hooks.fire, self._ell.push_event = saved_call, saved_fire, saved_push
        self.assertEqual(len(fired), 1)
        # 关键断言：播完之后时钟被锚在现在，下一次判断不会立刻又满足。
        self.assertLess(self._ell.silent_seconds(), 5)

    def test_clock_is_frozen_while_speaking(self):
        """播放期间沉默恒为 0 —— 沉默是从"说完"开始算的。

        不冻结的话，一段 134 秒的播报（Orin5 上真实出现过）刚说完，silent_seconds()
        已经是 134，下一次判断立刻满足任何阈值，于是话音刚落就又播一句进度。
        """
        self._register_mouth()
        self._ell._last_output_ts = time.time() - 300
        self.assertGreater(self._ell.silent_seconds(), 290)

        mcp_client._pending_actions['speak-1'] = asyncio.Event()   # 开始播
        mcp_client._pending_resources['speak-1'] = frozenset({'mouth'})
        self.assertEqual(self._ell.silent_seconds(), 0.0)

        mcp_client._pending_actions['speak-1'].set()                # 播完
        self.assertLess(self._ell.silent_seconds(), 5,
                        '播完之后沉默应从 0 重新计，而不是接着之前的 300 秒')

    def test_completion_callback_anchors_the_clock(self):
        """ACP 完成回调是沉默起点的权威锚 —— 两处完成入口都要经过它。"""
        self._register_mouth()
        mcp_client._pending_actions['speak-9'] = asyncio.Event()
        mcp_client._pending_resources['speak-9'] = frozenset({'mouth'})
        self._ell._last_output_ts = time.time() - 300
        ok = mcp_client.mark_action_complete('speak-9', {'status': 'completed'})
        self.assertTrue(ok)
        self.assertLess(self._ell.silent_seconds(), 5)

    def test_completion_of_a_non_output_action_does_not_anchor(self):
        """走路走完了不是"跟用户说了话"，不该重置沉默时钟。"""
        self._register_mouth()
        mcp_client._pending_actions['move-1'] = asyncio.Event()
        mcp_client._pending_resources['move-1'] = frozenset({'legs'})
        self._ell._last_output_ts = time.time() - 300
        mcp_client.mark_action_complete('move-1', {'status': 'completed'})
        self.assertGreater(self._ell.silent_seconds(), 290)

    def test_new_request_restarts_the_silence_epoch(self):
        """旧任务的定时器不能对着刚起步的新任务开火。

        场景：上个任务播报完上了表（比如 25s 后到点），任务随即结束、机器闲着，几秒后来
        了个新请求。旧定时器如果还活着，到点时看到的是**新任务**的子代理，就会给一个刚
        跑了几秒的任务播进度 —— 而这时新 turn 的第一轮还在路上，这句罐头汇报会抢在真正
        的回答前面出声。
        """
        self._ell._last_output_ts = time.time() - 24     # 上个任务 24 秒前说过话
        self.assertGreater(self._ell.silent_seconds(), 20)
        self._ell._reset_silence_clock()                  # 新请求进来
        self.assertLess(self._ell.silent_seconds(), 1)    # 纪元从现在重新开始

    def test_reset_is_skipped_for_self_originated_turns(self):
        """子代理完成 / ACP 回调触发的 turn 是同一件事的延续，重置会把沉默一直往后推。"""
        self.assertTrue(self._ell._trigger_is_self_originated(
            {'payload': {'sources': ['subagent:ab12/report']}}))
        self.assertFalse(self._ell._trigger_is_self_originated(
            {'payload': {'sources': ['dds:/remote_control/message']}}))
        # 混在一起时保守处理：有一个来自外部就算外部请求。
        self.assertFalse(self._ell._trigger_is_self_originated(
            {'payload': {'sources': ['acp:speak-1', 'dds:/remote_control/message']}}))

    def test_idle_turn_end_cancels_a_live_timer(self):
        """活干完了要显式取消 —— 只是"不重新上表"会留下一个睡着的表去伏击下个任务。"""
        async def go():
            self._ell._arm_narration_timer()
            t = self._ell._narration_timer
            self.assertIsNotNone(t)
            self._ell._cancel_narration_timer()
            await asyncio.sleep(0)
            return t
        t = asyncio.run(go())
        self.assertTrue(t.cancelled() or t.done())
        self.assertIsNone(self._ell._narration_timer)

    def test_timer_survives_without_event_loop(self):
        """启动早期/测试里没有运行中的 loop，上表不能把调用方炸掉。"""
        self._ell._cancel_narration_timer()
        self._ell._arm_narration_timer()      # 同步上下文，无 loop
        self.assertIsNone(self._ell._narration_timer)


if __name__ == '__main__':
    unittest.main()

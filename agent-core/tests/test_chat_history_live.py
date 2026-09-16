"""历史记录要在 turn 跑的过程中就可见，而不是等它结束。

以前 `save_turn` 只在 turn 结束时调用一次并且是纯 INSERT，一个跑了几分钟的 turn
在这期间在库里完全不存在 —— 历史 modal 手动刷新也刷不出来。现在每轮都写一次、
按 (session_id, turn_index) 覆盖，这几条断言钉住的就是"重复写同一轮不会变成多轮"
以及"分区/排序所依赖的 kind 和最后活动时间是对的"。
"""

import os
import pathlib
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import chat_history  # noqa: E402


class SaveTurnUpsertTest(unittest.TestCase):
    def setUp(self):
        chat_history.clear_all()

    def test_rewriting_a_turn_updates_it_in_place(self):
        sid = chat_history.create_session()
        chat_history.save_turn(sid, 0, [{'role': 'user', 'content': 'hi'}])
        chat_history.save_turn(sid, 0, [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'hello'},
        ])

        turns = chat_history.get_session_turns(sid)
        self.assertEqual(len(turns), 1, 'a rewritten turn must not become a second turn')
        self.assertEqual(len(turns[0]['messages']), 2)

        sessions, _ = chat_history.list_sessions()
        self.assertEqual([s['turn_count'] for s in sessions if s['id'] == sid], [1])

    def test_turn_timestamps_are_exposed(self):
        sid = chat_history.create_session()
        chat_history.save_turn(sid, 0, [{'role': 'user', 'content': 'hi'}])
        started = chat_history.get_session_turns(sid)[0]['started_at']
        time.sleep(0.01)
        chat_history.save_turn(sid, 0, [{'role': 'user', 'content': 'hi'},
                                        {'role': 'assistant', 'content': 'yo'}])
        turn = chat_history.get_session_turns(sid)[0]
        self.assertEqual(turn['started_at'], started, 'start time must not move')
        self.assertGreater(turn['updated_at'], started)

    def test_get_session_messages_still_returns_bare_turns(self):
        sid = chat_history.create_session()
        chat_history.save_turn(sid, 0, [{'role': 'user', 'content': 'hi'}])
        self.assertEqual(chat_history.get_session_messages(sid),
                         [[{'role': 'user', 'content': 'hi'}]])


class SessionOrderingTest(unittest.TestCase):
    def setUp(self):
        chat_history.clear_all()

    def test_kind_round_trips(self):
        main = chat_history.create_session(chat_history.KIND_MAIN)
        bg = chat_history.create_session(chat_history.KIND_BG_SUBAGENT)
        for sid in (main, bg):
            chat_history.save_turn(sid, 0, [{'role': 'user', 'content': 'x'}])
        kinds = {s['id']: s['kind'] for s in chat_history.list_sessions()[0]}
        self.assertEqual(kinds[main], chat_history.KIND_MAIN)
        self.assertEqual(kinds[bg], chat_history.KIND_BG_SUBAGENT)

    def test_ordered_by_last_activity_not_by_start(self):
        old = chat_history.create_session()
        chat_history.save_turn(old, 0, [{'role': 'user', 'content': 'first'}])
        time.sleep(0.01)
        new = chat_history.create_session()
        chat_history.save_turn(new, 0, [{'role': 'user', 'content': 'second'}])
        time.sleep(0.01)
        # 老会话又说话了 —— 它应该回到最前面。
        chat_history.save_turn(old, 1, [{'role': 'user', 'content': 'third'}])

        sessions, _ = chat_history.list_sessions()
        self.assertEqual([s['id'] for s in sessions][:2], [old, new])
        self.assertGreaterEqual(sessions[0]['last_at'], sessions[1]['last_at'])


if __name__ == '__main__':
    unittest.main()

"""
test_config_migrate_narration.py — event.llm 里主动播报相关键的迁移。

两件事，合在同一次 read-modify-write 里做：

1. **补种新键**。_seed_defaults 用的是整行粒度的 `INSERT OR IGNORE`，已部署机器上的
   'event' 行早就存在，新加的默认值永远进不去 —— 结果是阈值读成 0、功能静默不生效。
   这个坑 subagent 那几个键上真踩过（Orin5 实测读出来还是旧值）。
2. **删掉已废弃的键**。轮数维度取消后 narration_silence_rounds 没有任何读者，留在库里
   只会让人对着设置页猜"它还管不管用"。

迁移跑在模块导入期（config.py 末尾的 _migrate()），所以这里不 import config，而是直接
对一个临时 DB 跑同一段逻辑 —— 否则会碰到已经被别的测试模块导入过的那个单例。

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_config_migrate_narration.py
"""
import json
import os
import pathlib
import sqlite3
import subprocess
import sys
import tempfile
import unittest

_SRC = pathlib.Path(__file__).resolve().parents[1] / 'src'


def _run_migrate_on(event_row: dict) -> dict:
    """把 event 行塞进一个全新的 DB，跑一次真实的 config 导入，读回结果。

    用子进程而不是直接 import：_migrate() 只在模块导入期执行一次，而 config 这个单例
    在本次 pytest 里早被别的模块导入过了。
    """
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, 'test.db')
    conn = sqlite3.connect(db)
    conn.execute('CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT)')
    conn.execute("INSERT INTO config (key, value) VALUES ('event', ?)",
                 (json.dumps(event_row),))
    conn.commit()
    conn.close()

    env = {**os.environ, 'DB_PATH': db, 'PYTHONPATH': str(_SRC)}
    subprocess.run([sys.executable, '-c', 'import config'], env=env, check=True,
                   capture_output=True)

    conn = sqlite3.connect(db)
    row = conn.execute("SELECT value FROM config WHERE key='event'").fetchone()
    conn.close()
    return json.loads(row[0])['llm']


class TestNarrationKeyMigration(unittest.TestCase):
    def test_drops_the_retired_rounds_key(self):
        """轮数维度已取消 —— 这个键必须从库里清掉，不是留着当孤儿。"""
        llm = _run_migrate_on({'llm': {
            'narration_silence_rounds': 4,
            'narration_silence_seconds': 25,
            'prompt_system': './resource/memory/prompt_system.md',
        }})
        self.assertNotIn('narration_silence_rounds', llm)
        self.assertEqual(llm['narration_silence_seconds'], 25)

    def test_seeds_missing_keys(self):
        """已部署机器上的 'event' 行早就存在，靠 INSERT OR IGNORE 补不进新默认值。"""
        llm = _run_migrate_on({'llm': {'prompt_system': './resource/memory/prompt_system.md'}})
        self.assertEqual(llm['narration_silence_seconds'], 25)
        self.assertEqual(llm['narration_context_chars'], 6000)
        self.assertEqual(llm['narration_timeout_s'], 20)

    def test_does_not_clobber_a_hand_tuned_value(self):
        """手工调过的值不动 —— 只补缺失的键。"""
        llm = _run_migrate_on({'llm': {
            'narration_silence_seconds': 90,
            'prompt_system': './resource/memory/prompt_system.md',
        }})
        self.assertEqual(llm['narration_silence_seconds'], 90)

    def test_is_idempotent(self):
        """跑第二遍不该再改动任何东西（每次启动都会跑一次）。"""
        base = {'llm': {'narration_silence_rounds': 4,
                        'prompt_system': './resource/memory/prompt_system.md'}}
        once = _run_migrate_on(base)
        twice = _run_migrate_on({'llm': dict(once)})
        self.assertEqual(once, twice)
        self.assertNotIn('narration_silence_rounds', twice)


if __name__ == '__main__':
    unittest.main()

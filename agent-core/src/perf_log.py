"""
perf_log.py — 性能追踪模块。

记录每次 agent turn 各阶段的时间戳，持久化到 SQLite，
提供查询和聚合接口。
"""

import json
import time
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

import config


@dataclass
class PerfTrace:
    """收集一次 turn 各阶段的时间戳。"""
    turn_id: str = ''
    source: str = ''
    trigger_text: str = ''
    # 感知层
    audio_start_ts: Optional[float] = None
    audio_end_ts: Optional[float] = None
    asr_complete_ts: Optional[float] = None
    # Collector
    collector_receive_ts: Optional[float] = None
    trigger_emit_ts: Optional[float] = None
    # Agent Core
    llm_start_ts: Optional[float] = None
    llm_end_ts: Optional[float] = None
    tool_start_ts: Optional[float] = None
    tool_end_ts: Optional[float] = None
    tts_start_ts: Optional[float] = None
    tts_end_ts: Optional[float] = None
    turn_end_ts: Optional[float] = None
    # 元数据
    round_count: int = 1
    tool_names: list = field(default_factory=list)


def _ms(start: Optional[float], end: Optional[float]) -> Optional[int]:
    """计算毫秒差值，任一为 None 则返回 None。"""
    if start is None or end is None:
        return None
    return int((end - start) * 1000)


def commit(trace: PerfTrace):
    """计算派生字段，写入 SQLite。"""
    now = time.time()
    vad_ms = _ms(trace.audio_start_ts, trace.audio_end_ts)
    asr_ms = _ms(trace.audio_end_ts, trace.asr_complete_ts)
    collector_ms = _ms(trace.asr_complete_ts or trace.collector_receive_ts, trace.trigger_emit_ts)
    llm_ms = _ms(trace.llm_start_ts, trace.llm_end_ts)
    tool_ms = _ms(trace.tool_start_ts, trace.tool_end_ts)
    tts_ms = _ms(trace.tts_start_ts, trace.tts_end_ts)
    total_ms = _ms(trace.audio_start_ts or trace.llm_start_ts, trace.turn_end_ts)

    conn = config._get_conn()
    conn.execute(
        '''INSERT INTO perf_turns (
            turn_id, created_at,
            audio_start_ts, audio_end_ts, asr_complete_ts,
            collector_receive_ts, trigger_emit_ts,
            llm_start_ts, llm_end_ts,
            tool_start_ts, tool_end_ts,
            tts_start_ts, tts_end_ts,
            turn_end_ts,
            vad_duration_ms, asr_duration_ms, collector_delay_ms,
            llm_duration_ms, tool_duration_ms, tts_duration_ms, total_duration_ms,
            round_count, tool_names, trigger_text, source
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
        (
            trace.turn_id, now,
            trace.audio_start_ts, trace.audio_end_ts, trace.asr_complete_ts,
            trace.collector_receive_ts, trace.trigger_emit_ts,
            trace.llm_start_ts, trace.llm_end_ts,
            trace.tool_start_ts, trace.tool_end_ts,
            trace.tts_start_ts, trace.tts_end_ts,
            trace.turn_end_ts,
            vad_ms, asr_ms, collector_ms,
            llm_ms, tool_ms, tts_ms, total_ms,
            trace.round_count,
            json.dumps(trace.tool_names, ensure_ascii=False),
            trace.trigger_text[:200],
            trace.source,
        ),
    )
    conn.commit()
    conn.close()


def query(start: float = 0, end: float = 0, limit: int = 50, offset: int = 0) -> dict:
    """查询 perf_turns 记录。"""
    conn = config._get_conn()
    conn.row_factory = sqlite3.Row

    where = 'WHERE 1=1'
    params = []
    if start:
        where += ' AND created_at >= ?'
        params.append(start)
    if end:
        where += ' AND created_at <= ?'
        params.append(end)

    # 总数
    total = conn.execute(f'SELECT COUNT(*) FROM perf_turns {where}', params).fetchone()[0]

    rows = conn.execute(
        f'SELECT * FROM perf_turns {where} ORDER BY created_at DESC LIMIT ? OFFSET ?',
        params + [limit, offset],
    ).fetchall()

    turns = []
    for r in rows:
        d = dict(r)
        # tool_names 存的是 JSON 字符串
        try:
            d['tool_names'] = json.loads(d['tool_names'])
        except (json.JSONDecodeError, TypeError):
            d['tool_names'] = []
        turns.append(d)

    conn.close()
    return {'turns': turns, 'total': total}


def aggregate(start: float = 0, end: float = 0) -> dict:
    """聚合统计：avg 和 p95。"""
    conn = config._get_conn()

    where = 'WHERE 1=1'
    params = []
    if start:
        where += ' AND created_at >= ?'
        params.append(start)
    if end:
        where += ' AND created_at <= ?'
        params.append(end)

    count = conn.execute(f'SELECT COUNT(*) FROM perf_turns {where}', params).fetchone()[0]
    if count == 0:
        conn.close()
        return {'count': 0, 'avg': {}, 'p95': {}}

    fields = ['vad_duration_ms', 'asr_duration_ms', 'collector_delay_ms',
              'llm_duration_ms', 'tool_duration_ms', 'tts_duration_ms', 'total_duration_ms']

    # Average
    avg_exprs = ', '.join(f'AVG({f})' for f in fields)
    row = conn.execute(f'SELECT {avg_exprs} FROM perf_turns {where}', params).fetchone()
    avg = {}
    for i, f in enumerate(fields):
        avg[f] = int(row[i]) if row[i] is not None else None

    # P95 — 取每个字段排序后 95% 位置的值
    p95 = {}
    p95_offset = int(count * 0.95) - 1
    if p95_offset < 0:
        p95_offset = 0
    for f in fields:
        r = conn.execute(
            f'SELECT {f} FROM perf_turns {where} AND {f} IS NOT NULL ORDER BY {f} ASC LIMIT 1 OFFSET ?',
            params + [p95_offset],
        ).fetchone()
        p95[f] = r[0] if r else None

    conn.close()
    return {'count': count, 'avg': avg, 'p95': p95}


def latest(n: int = 20) -> list:
    """获取最近 N 条记录。"""
    result = query(limit=n)
    return result['turns']


def prune(days: int = 7):
    """清理过期记录。"""
    cutoff = time.time() - days * 86400
    conn = config._get_conn()
    conn.execute('DELETE FROM perf_turns WHERE created_at < ?', (cutoff,))
    conn.commit()
    deleted = conn.total_changes
    conn.close()
    if deleted:
        print(f'[perf_log] pruned {deleted} records older than {days} days')


# 模块加载时自动清理
try:
    prune()
except Exception:
    pass  # 表可能还不存在（首次启动前）

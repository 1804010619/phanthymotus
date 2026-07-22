"""
perf_log.py — 开放 Span 式性能追踪。

每个组件（perception、core、driver）上报命名 span，
agent-core 收集后按 trace_id（turn_id）关联存储到 SQLite。

Span 格式：
  {"span": "asr_inference", "component": "perception",
   "start_ts": float, "end_ts": float, "meta": {...}}
"""

import json
import time
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

import config


def _get_conn():
    return config._get_conn()


def commit_spans(trace_id: str, spans: list[dict], source: str = '', trigger_text: str = ''):
    """写入一组 spans 到 perf_spans 表，同时写/更新 perf_turns 索引。"""
    if not spans:
        return
    now = time.time()
    conn = _get_conn()

    # 检查 turn 是否已存在（TTS 等异步 span 会后到）
    existing = conn.execute(
        'SELECT id FROM perf_turns WHERE turn_id=?', (trace_id,)
    ).fetchone()

    if not existing:
        # 计算 total_duration
        starts = [s['start_ts'] for s in spans if s.get('start_ts') and s['start_ts'] > 1e9]
        ends = [s['end_ts'] for s in spans if s.get('end_ts') and s['end_ts'] > 1e9]
        total_ms = int((max(ends) - min(starts)) * 1000) if starts and ends else None
        # 新建 perf_turns 记录
        conn.execute(
            '''INSERT INTO perf_turns (turn_id, created_at, source, trigger_text, total_duration_ms)
               VALUES (?, ?, ?, ?, ?)''',
            (trace_id, now, source, trigger_text[:200], total_ms),
        )

    # 写 perf_spans
    for s in spans:
        start_ts = s.get('start_ts')
        end_ts = s.get('end_ts')
        dur = None
        if start_ts and end_ts and start_ts > 1e9 and end_ts > 1e9:
            dur = int((end_ts - start_ts) * 1000)
        conn.execute(
            '''INSERT INTO perf_spans (trace_id, span, component, start_ts, end_ts, duration_ms, meta, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                trace_id,
                s.get('span', ''),
                s.get('component', ''),
                start_ts,
                end_ts,
                dur,
                json.dumps(s.get('meta', {}), ensure_ascii=False),
                now,
            ),
        )

    conn.commit()
    conn.close()


def query_latest(n: int = 20) -> list:
    """返回最近 N 个 turn，每个 turn 附带其 spans。"""
    conn = _get_conn()
    conn.row_factory = sqlite3.Row

    turns = conn.execute(
        'SELECT * FROM perf_turns ORDER BY created_at DESC LIMIT ?', (n,)
    ).fetchall()

    result = []
    for t in turns:
        td = dict(t)
        trace_id = td['turn_id']
        spans = conn.execute(
            'SELECT span, component, start_ts, end_ts, duration_ms, meta FROM perf_spans WHERE trace_id=? ORDER BY start_ts',
            (trace_id,),
        ).fetchall()
        td['spans'] = []
        for s in spans:
            sd = dict(s)
            try:
                sd['meta'] = json.loads(sd['meta'])
            except (json.JSONDecodeError, TypeError):
                sd['meta'] = {}
            td['spans'].append(sd)
        result.append(td)

    conn.close()
    return result


def query_spans(trace_id: str) -> list:
    """返回单个 turn 的全部 spans。"""
    conn = _get_conn()
    conn.row_factory = sqlite3.Row
    spans = conn.execute(
        'SELECT * FROM perf_spans WHERE trace_id=? ORDER BY start_ts', (trace_id,)
    ).fetchall()
    result = []
    for s in spans:
        sd = dict(s)
        try:
            sd['meta'] = json.loads(sd['meta'])
        except (json.JSONDecodeError, TypeError):
            sd['meta'] = {}
        result.append(sd)
    conn.close()
    return result


def aggregate(start: float = 0, end: float = 0) -> dict:
    """按 span 名称聚合 avg/p95。"""
    conn = _get_conn()

    where = 'WHERE duration_ms IS NOT NULL'
    params = []
    if start:
        where += ' AND created_at >= ?'
        params.append(start)
    if end:
        where += ' AND created_at <= ?'
        params.append(end)

    # 总 turn 数
    turn_count = conn.execute(
        f'SELECT COUNT(DISTINCT trace_id) FROM perf_spans {where}', params
    ).fetchone()[0]

    # 按 span 名称聚合
    rows = conn.execute(
        f'''SELECT span, COUNT(*) as cnt, AVG(duration_ms) as avg_ms
            FROM perf_spans {where}
            GROUP BY span ORDER BY avg_ms DESC''',
        params,
    ).fetchall()

    by_span = {}
    for row in rows:
        span_name = row[0]
        cnt = row[1]
        avg_ms = int(row[2]) if row[2] else 0

        # P95
        p95_offset = max(0, int(cnt * 0.95) - 1)
        p95_row = conn.execute(
            f'SELECT duration_ms FROM perf_spans {where} AND span=? ORDER BY duration_ms ASC LIMIT 1 OFFSET ?',
            params + [span_name, p95_offset],
        ).fetchone()
        p95_ms = p95_row[0] if p95_row else avg_ms

        # 平均开始偏移（相对于每个 trace 中最早 span 的 start_ts）
        offset_row = conn.execute(
            f'''SELECT AVG((s.start_ts - t_min.min_start) * 1000) FROM perf_spans s
                INNER JOIN (SELECT trace_id, MIN(start_ts) as min_start FROM perf_spans GROUP BY trace_id) t_min
                ON s.trace_id = t_min.trace_id
                {where} AND s.span = ?''',
            params + [span_name],
        ).fetchone()
        avg_offset_ms = int(offset_row[0]) if offset_row and offset_row[0] else 0

        by_span[span_name] = {'avg_ms': avg_ms, 'p95_ms': p95_ms, 'count': cnt, 'avg_offset_ms': avg_offset_ms}

    conn.close()
    return {'count': turn_count, 'by_span': by_span}


def prune(days: int = 7):
    """清理过期记录。"""
    cutoff = time.time() - days * 86400
    conn = _get_conn()
    conn.execute('DELETE FROM perf_spans WHERE created_at < ?', (cutoff,))
    conn.execute('DELETE FROM perf_turns WHERE created_at < ?', (cutoff,))
    conn.commit()
    conn.close()


# 模块加载时自动清理
try:
    prune()
except Exception:
    pass

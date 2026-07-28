"""
api/performance.py — 性能分析 API（开放 Span 式）。
"""

import time
from fastapi import APIRouter, Query

import perf_log

router = APIRouter(prefix='/performance', tags=['performance'])


@router.get('/latest')
async def get_latest(n: int = Query(20, ge=1, le=200)):
    return perf_log.query_latest(n=n)


@router.get('/spans')
async def get_spans(trace_id: str = Query(...)):
    return perf_log.query_spans(trace_id=trace_id)


@router.get('/aggregate')
async def get_aggregate(
    start: float = Query(0),
    end: float = Query(0),
):
    return perf_log.aggregate(start=start, end=end)


@router.get('/usage')
async def get_usage(range: str = Query('7d')):
    """Token usage summary + daily breakdown."""
    now = time.time()
    range_map = {
        'today': 86400,
        '7d': 7 * 86400,
        '30d': 30 * 86400,
    }
    seconds = range_map.get(range, 7 * 86400)
    start = now - seconds

    summary = perf_log.query_usage_summary(start=start)
    daily = perf_log.query_usage_daily(start=start)
    return {'summary': summary, 'daily': daily}


@router.delete('/clear')
async def clear_data():
    import config
    conn = config._get_conn()
    conn.execute('DELETE FROM perf_spans')
    conn.execute('DELETE FROM perf_turns')
    conn.execute('DELETE FROM token_usage')
    conn.commit()
    conn.close()
    return {'code': 200, 'message': 'cleared'}

"""
api/performance.py — 性能分析 API（开放 Span 式）。
"""

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

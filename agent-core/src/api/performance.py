"""
api/performance.py — 性能分析 API。
"""

from fastapi import APIRouter, Query

import perf_log

router = APIRouter(prefix='/performance', tags=['performance'])


@router.get('/turns')
async def get_turns(
    start: float = Query(0, description='起始 unix 时间戳'),
    end: float = Query(0, description='结束 unix 时间戳'),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    return perf_log.query(start=start, end=end, limit=limit, offset=offset)


@router.get('/aggregate')
async def get_aggregate(
    start: float = Query(0),
    end: float = Query(0),
):
    return perf_log.aggregate(start=start, end=end)


@router.get('/latest')
async def get_latest(n: int = Query(20, ge=1, le=200)):
    return perf_log.latest(n=n)

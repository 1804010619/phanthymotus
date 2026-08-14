"""API endpoints for the dashboard.

Mounted under `/api` so the static file mount at `/` cannot shadow them, and to
match the project convention that all endpoints live under `/api/...`.

Responses are raw JSON rather than agent-core's `{code, message, data}`
envelope: `deploy.sh status` curls these directly and the README documents the
shapes, so the extra nesting would cost more than the consistency buys.
"""

import logging

from fastapi import APIRouter, HTTPException, Query, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


@router.get("/status")
async def status(request: Request):
    """Live operational state — the dashboard's Overview tab polls this."""
    config = request.app.state.config
    job_queue = request.app.state.job_queue
    poller = request.app.state.poller
    store = request.app.state.store

    return {
        "status": "ok",
        "queue_depth": job_queue.queue_size(),
        "active_jobs": job_queue.active_count(),
        "total_processed": job_queue.total_processed,
        "active": [_summarize_active(j) for j in job_queue.active_jobs()],
        "config": {
            "repos": list(config.repos),
            "max_concurrent_jobs": config.max_concurrent_jobs,
            "job_timeout_seconds": config.job_timeout_seconds,
            "build_timeout_seconds": config.build_timeout_seconds,
            "max_attempts": config.max_attempts,
            "job_history_days": config.job_history_days,
            "mirror": config.mirror,
            "webhook_enabled": config.webhook_enabled,
            "llm_configured": bool(config.llm_base_url and config.llm_api_key),
            "llm_model": config.llm_model if config.llm_base_url else "",
        },
        "poller": poller.stats() if poller else {"enabled": False},
        "history": await store.stats(),
    }


@router.get("/jobs")
async def list_jobs(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: str | None = None,
    repo: str | None = None,
):
    """Paginated job history from SQLite."""
    store = request.app.state.store
    jobs, total = await store.list_jobs(
        limit=limit, offset=offset, status=status, repo=repo
    )
    return {"jobs": jobs, "total": total, "limit": limit, "offset": offset}


@router.get("/jobs/{job_id}")
async def get_job(request: Request, job_id: str):
    """Full job detail: metadata, build results, review text, findings."""
    job = await request.app.state.store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/jobs/{job_id}/log/{idx}")
async def get_job_log(
    request: Request,
    job_id: str,
    idx: int,
    offset: int = Query(0, ge=0),
):
    """Build log bytes from `offset`, for incremental tailing.

    The client passes back the `offset` from the previous response, so a running
    build is followed by repeating the same request. An offset at or past EOF
    returns empty content, which is what a poller sees between writes.
    """
    result = await request.app.state.store.read_log(job_id, idx, offset=offset)
    if result is None:
        raise HTTPException(status_code=404, detail="Log not found")
    return result


def _summarize_active(job) -> dict:
    """Shape an in-flight job for the Overview tab.

    Read from the in-memory queue rather than SQLite so elapsed time and the
    current stage advance between the coarse write-through checkpoints.
    """
    return {
        "id": job.id,
        "repo": job.repo_full_name,
        "pr_number": job.pr_number,
        "head_sha": job.pr_head_sha,
        "requester": job.requester,
        "status": job.status.value,
        "stage": job.stage,
        "stage_detail": job.stage_detail,
        "stage_elapsed": job.stage_elapsed_seconds(),
        "attempt": job.attempt,
        "elapsed": job.elapsed_seconds(),
    }

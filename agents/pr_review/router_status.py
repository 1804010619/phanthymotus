"""Status and monitoring endpoints.

With polling as the trigger there is no inbound webhook traffic to confirm the
agent is alive, so these endpoints are the way to check it — particularly
`poller.last_poll_at` and `poller.last_error`.
"""

import logging

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/status")
async def status(request: Request):
    config = request.app.state.config
    job_queue = request.app.state.job_queue
    poller = request.app.state.poller

    return {
        "status": "ok",
        "queue_depth": job_queue.queue_size(),
        "active_jobs": job_queue.active_count(),
        "total_processed": job_queue.total_processed,
        "config": {
            "repos": list(config.repos),
            "max_concurrent_jobs": config.max_concurrent_jobs,
            "job_timeout_seconds": config.job_timeout_seconds,
            "build_timeout_seconds": config.build_timeout_seconds,
            "max_attempts": config.max_attempts,
            "webhook_enabled": config.webhook_enabled,
            "llm_configured": bool(config.llm_base_url and config.llm_api_key),
        },
        "poller": poller.stats() if poller else {"enabled": False},
    }


@router.get("/jobs")
async def list_jobs(request: Request, limit: int = 20):
    jobs = request.app.state.job_queue.recent_jobs(limit)
    return [
        {
            "id": j.id,
            "repo": j.repo_full_name,
            "pr": j.pr_number,
            "sha": j.pr_head_sha[:7],
            "status": j.status.value,
            "attempt": j.attempt,
            "requester": j.requester,
            "source": j.source,
            "created_at": j.created_at.isoformat(),
            "elapsed": j.elapsed_seconds(),
        }
        for j in jobs
    ]


@router.get("/jobs/{job_id}")
async def get_job(request: Request, job_id: str):
    job = request.app.state.job_queue.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "id": job.id,
        "repo": job.repo_full_name,
        "pr": job.pr_number,
        "sha": job.pr_head_sha,
        "head_ref": job.pr_head_ref,
        "base_ref": job.pr_base_ref,
        "status": job.status.value,
        "attempt": job.attempt,
        "attempt_errors": job.attempt_errors,
        "requester": job.requester,
        "source": job.source,
        "options": {
            "skip_build": job.skip_build,
            "build_only": job.build_only,
            "force_targets": job.force_targets,
        },
        "build_results": [
            {
                "target": r.target.value,
                "driver_path": r.driver_path,
                "success": r.success,
                "image_tag": r.image_tag,
                "log_tail": r.log_tail,
            }
            for r in job.build_results
        ],
        "review_text": job.review_text,
        "error": job.error,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "elapsed": job.elapsed_seconds(),
    }

"""Status and monitoring router."""

import logging

from fastapi import APIRouter, Request

from .models import JobStatus

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/status")
async def status(request: Request):
    job_queue = request.app.state.job_queue
    return {
        "status": "ok",
        "queue_depth": job_queue.queue_size(),
        "active_jobs": job_queue.active_count(),
        "total_processed": job_queue.total_processed,
    }


@router.get("/jobs")
async def list_jobs(request: Request, limit: int = 20):
    job_queue = request.app.state.job_queue
    jobs = job_queue.recent_jobs(limit)
    return [
        {
            "id": j.id,
            "repo": j.repo_full_name,
            "pr": j.pr_number,
            "sha": j.pr_head_sha[:7],
            "status": j.status.value,
            "requester": j.requester,
            "created_at": j.created_at.isoformat(),
            "elapsed": j.elapsed_seconds(),
        }
        for j in jobs
    ]


@router.get("/jobs/{job_id}")
async def get_job(request: Request, job_id: str):
    job_queue = request.app.state.job_queue
    job = job_queue.get_job(job_id)
    if job is None:
        return {"error": "not found"}, 404
    return {
        "id": job.id,
        "repo": job.repo_full_name,
        "pr": job.pr_number,
        "sha": job.pr_head_sha,
        "head_ref": job.pr_head_ref,
        "status": job.status.value,
        "requester": job.requester,
        "skip_build": job.skip_build,
        "build_only": job.build_only,
        "force_targets": job.force_targets,
        "build_results": [
            {
                "target": r.target.value,
                "driver_path": r.driver_path,
                "success": r.success,
                "image_tag": r.image_tag,
            }
            for r in job.build_results
        ],
        "review_text": job.review_text[:500] if job.review_text else "",
        "error": job.error,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "elapsed": job.elapsed_seconds(),
    }

"""Webhook router — receives GitHub issue_comment events."""

import hashlib
import hmac
import logging

from fastapi import APIRouter, Header, HTTPException, Request

from .config import Config
from .models import JobStatus, ReviewJob, parse_trigger_command

logger = logging.getLogger(__name__)

router = APIRouter()


def _verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify GitHub HMAC-SHA256 webhook signature."""
    if not secret:
        return True  # no secret configured, skip verification (dev mode)
    expected = "sha256=" + hmac.HMAC(
        secret.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


@router.post("/webhook")
async def webhook(
    request: Request,
    x_github_event: str = Header(None, alias="X-GitHub-Event"),
    x_hub_signature_256: str = Header("", alias="X-Hub-Signature-256"),
    x_github_delivery: str = Header("", alias="X-GitHub-Delivery"),
):
    config: Config = request.app.state.config
    job_queue = request.app.state.job_queue
    github_client = request.app.state.github_client

    body = await request.body()

    # Verify signature
    if not _verify_signature(body, x_hub_signature_256, config.github_webhook_secret):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Only process issue_comment events
    if x_github_event != "issue_comment":
        return {"status": "ignored", "reason": f"event={x_github_event}"}

    payload = await request.json()

    # Only process new comments on PRs
    if payload.get("action") != "created":
        return {"status": "ignored", "reason": "not a new comment"}

    issue = payload.get("issue", {})
    if "pull_request" not in issue:
        return {"status": "ignored", "reason": "not a PR comment"}

    # Parse trigger command
    comment_body = payload.get("comment", {}).get("body", "")
    trigger = parse_trigger_command(comment_body)
    if trigger is None:
        return {"status": "ignored", "reason": "no trigger command"}

    # Extract info
    repo_full_name = payload["repository"]["full_name"]
    pr_number = issue["number"]
    comment_id = payload["comment"]["id"]
    requester = payload["comment"]["user"]["login"]

    # Check repo is one we handle
    if repo_full_name not in config.repos:
        return {"status": "ignored", "reason": f"repo {repo_full_name} not configured"}

    # Fetch PR details (head SHA, refs)
    pr_info = await github_client.get_pr(repo_full_name, pr_number)
    head_sha = pr_info["head"]["sha"]
    head_ref = pr_info["head"]["ref"]
    base_ref = pr_info["base"]["ref"]

    # Dedup: skip if same repo+PR+SHA already queued or running
    if job_queue.has_pending_job(repo_full_name, pr_number, head_sha):
        logger.info(f"Dedup: job for {repo_full_name}#{pr_number}@{head_sha[:7]} already exists")
        return {"status": "dedup", "pr": pr_number}

    # Create job
    job = ReviewJob(
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        pr_head_sha=head_sha,
        pr_head_ref=head_ref,
        pr_base_ref=base_ref,
        comment_id=comment_id,
        requester=requester,
        skip_build=trigger["skip_build"],
        build_only=trigger["build_only"],
        force_targets=trigger["force_targets"],
    )

    # React to comment with eyes emoji
    await github_client.add_reaction(repo_full_name, comment_id, "eyes")

    # Enqueue
    await job_queue.enqueue(job)
    logger.info(f"Enqueued job {job.id} for {repo_full_name}#{pr_number}@{head_sha[:7]}")

    return {"status": "queued", "job_id": job.id, "pr": pr_number}

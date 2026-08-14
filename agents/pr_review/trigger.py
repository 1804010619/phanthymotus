"""Shared trigger logic — creates jobs from PR comments.

Used by both the webhook receiver and the polling loop, so the two entry
points behave identically and share dedup.
"""

import logging

from . import comments
from .config import Config
from .github_client import GitHubClient
from .models import ReviewJob, parse_trigger_command

logger = logging.getLogger(__name__)


async def create_job_from_comment(
    repo_full_name: str,
    pr_number: int,
    comment_id: int,
    comment_body: str,
    requester: str,
    config: Config,
    github_client: GitHubClient,
    job_queue,
    source: str = "webhook",
) -> ReviewJob | None:
    """Parse a PR comment and enqueue a review job if it contains the trigger.

    Returns the enqueued job, or None if the comment was ignored / deduped.
    """
    trigger = parse_trigger_command(comment_body)
    if trigger is None:
        return None

    if repo_full_name not in config.repos:
        logger.warning(f"Ignoring trigger for unconfigured repo: {repo_full_name}")
        return None

    try:
        pr_info = await github_client.get_pr(repo_full_name, pr_number)
    except Exception as e:
        logger.error(f"Failed to fetch PR {repo_full_name}#{pr_number}: {e}")
        return None

    if pr_info.get("state") != "open":
        logger.info(f"Skipping {repo_full_name}#{pr_number}: PR is not open")
        return None

    head_sha = pr_info["head"]["sha"]

    # Dedup: same repo + PR + SHA already queued or running.
    if job_queue.has_pending_job(repo_full_name, pr_number, head_sha):
        logger.info(
            f"Dedup: job for {repo_full_name}#{pr_number}@{head_sha[:7]} already pending"
        )
        return None

    job = ReviewJob(
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        pr_head_sha=head_sha,
        pr_head_ref=pr_info["head"]["ref"],
        pr_base_ref=pr_info["base"]["ref"],
        comment_id=comment_id,
        requester=requester,
        source=source,
        skip_build=trigger["skip_build"],
        build_only=trigger["build_only"],
        force_targets=trigger["force_targets"],
    )

    # Acknowledge immediately. With polling this matters — the trigger comment
    # may sit for up to one poll interval, and without a reply the requester
    # cannot tell whether the agent saw it. The worker then edits this same
    # comment through the build and result stages.
    await github_client.add_reaction(repo_full_name, comment_id, "eyes")
    try:
        job.progress_comment_id = await github_client.post_comment(
            repo_full_name,
            pr_number,
            comments.format_ack(
                requester=requester,
                head_sha=head_sha,
                skip_build=job.skip_build,
                build_only=job.build_only,
                source=source,
            ),
        )
    except Exception as e:
        # Not fatal — the worker will post a fresh comment when it starts.
        logger.warning(f"Failed to post acknowledgment comment: {e}")

    await job_queue.enqueue(job)
    logger.info(
        f"Enqueued job {job.id} via {source}: "
        f"{repo_full_name}#{pr_number}@{head_sha[:7]} by {requester}"
    )
    return job

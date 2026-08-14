"""Worker — runs the review pipeline for one job, with timeout and retry.

Retry policy: a job that exceeds `job_timeout_seconds` is presumed lost and
retried, as are infrastructure failures (network, git, registry). Conditions
caused by the PR itself are terminal and reported immediately — a merge
conflict or a genuinely broken build is the answer, not something to retry.
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from . import comments
from .build_detector import detect_targets
from .builder import build_core, build_driver, build_perception
from .config import Config
from .git_workspace import GitWorkspaceManager
from .github_client import GitHubClient
from .models import (
    BuildResult,
    BuildTarget,
    JobStatus,
    ReviewError,
    ReviewJob,
)
from .reviewer import llm_review, run_rule_checks

logger = logging.getLogger(__name__)

# Marks a timeout reason string, so the final status can distinguish a lost
# job from a hard error without threading an extra flag through the loop.
TIMEOUT_PREFIX = "Job timed out"


# ── Entry point: retry wrapper ─────────────────────────────────────────────────


async def run_job(
    job: ReviewJob,
    config: Config,
    github_client: GitHubClient,
    workspace_mgr: GitWorkspaceManager,
):
    """Run a job, retrying on timeout or infrastructure failure."""
    job.started_at = datetime.now(timezone.utc)

    for attempt in range(1, config.max_attempts + 1):
        job.attempt = attempt
        job.status = JobStatus.RUNNING

        if attempt > 1:
            logger.info(
                f"Job {job.id}: attempt {attempt}/{config.max_attempts} "
                f"for {job.repo_full_name}#{job.pr_number}"
            )

        try:
            await asyncio.wait_for(
                _run_once(job, config, github_client, workspace_mgr),
                timeout=config.job_timeout_seconds,
            )
            # A terminal state was reached and reported inside the pipeline
            # (review posted, or a real build failure). Either way, done.
            job.finished_at = datetime.now(timezone.utc)
            await _cleanup(job, workspace_mgr)
            return

        except asyncio.TimeoutError:
            minutes = config.job_timeout_seconds // 60
            reason = (
                f"{TIMEOUT_PREFIX}: exceeded {config.job_timeout_seconds}s "
                f"({minutes} min) without completing — presumed lost."
            )
            retryable = True

        except ReviewError as e:
            reason = str(e)
            retryable = e.retryable

        except asyncio.CancelledError:
            # Agent is shutting down — do not retry, do not comment.
            job.status = JobStatus.CANCELLED
            job.finished_at = datetime.now(timezone.utc)
            await _cleanup(job, workspace_mgr)
            raise

        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            retryable = True
            logger.exception(f"Job {job.id} attempt {attempt} raised")

        # This attempt failed. Record it and clean up before deciding.
        job.attempt_errors.append(reason)
        job.error = reason
        await _cleanup(job, workspace_mgr)

        if not retryable:
            job.status = JobStatus.ERROR
            job.finished_at = datetime.now(timezone.utc)
            await _report(
                job, github_client,
                comments.format_error(job.pr_head_sha, reason),
            )
            return

        if attempt < config.max_attempts:
            job.status = JobStatus.RETRYING
            await _report(
                job, github_client,
                comments.format_retrying(
                    job.pr_head_sha, attempt, config.max_attempts,
                    reason, config.retry_backoff_seconds,
                ),
            )
            await asyncio.sleep(config.retry_backoff_seconds)
        else:
            job.status = (
                JobStatus.TIMEOUT
                if reason.startswith(TIMEOUT_PREFIX)
                else JobStatus.ERROR
            )
            job.finished_at = datetime.now(timezone.utc)
            logger.error(
                f"Job {job.id} failed after {config.max_attempts} attempts: {reason}"
            )
            await _report(
                job, github_client,
                comments.format_final_failure(
                    job.pr_head_sha, config.max_attempts, job.attempt_errors,
                ),
            )
            return


# ── The pipeline itself ────────────────────────────────────────────────────────


async def _run_once(
    job: ReviewJob,
    config: Config,
    github_client: GitHubClient,
    workspace_mgr: GitWorkspaceManager,
):
    """One attempt at the full pipeline.

    Raises on failure. Returns normally once a terminal state has been reached
    and reported (review posted, or build failure reported).
    """
    # 1. Fetch latest refs
    await workspace_mgr.fetch(job.repo_full_name)

    # 2. Create an isolated worktree with the PR merged onto main
    worktree = await workspace_mgr.create_worktree(
        job.repo_full_name, job.pr_number, job.pr_head_sha
    )
    job.worktree_path = str(worktree)

    # 3. Determine what changed
    changed_files = await workspace_mgr.get_changed_files(worktree)
    if not changed_files:
        await _report(
            job, github_client, comments.format_no_changes(job.pr_head_sha)
        )
        job.status = JobStatus.REVIEW_DONE
        return

    if job.force_targets:
        targets, driver_paths = _parse_forced_targets(job.force_targets)
    else:
        targets, driver_paths = detect_targets(job.repo_full_name, changed_files)

    # 4. Build
    if not job.skip_build and targets:
        await _report(
            job, github_client,
            comments.format_building(
                job.requester, job.pr_head_sha, targets, driver_paths
            ),
        )

        results = await _execute_builds(targets, driver_paths, worktree, config)
        job.build_results = results

        await _report(
            job, github_client,
            comments.format_build_result(job.pr_head_sha, results),
        )

        if any(not r.success for r in results):
            # A real build failure — terminal and already reported. Not
            # retried: the author needs to fix the code, and rebuilding the
            # same commit twice more would just burn an hour saying the same.
            job.status = JobStatus.BUILD_FAILED
            return

        job.status = JobStatus.BUILD_SUCCESS

    elif not job.skip_build:
        await _report(
            job, github_client,
            comments.format_no_build_needed(job.pr_head_sha),
        )

    # 5. Review
    if not job.build_only:
        diff_stat = await workspace_mgr.get_diff_stat(worktree)
        findings = run_rule_checks(changed_files, diff_stat)

        diff = await workspace_mgr.get_diff(worktree, config.max_diff_lines)
        job.review_text = await llm_review(config, changed_files, diff, findings)

        # The review is its own comment — the progress comment keeps the build
        # result, which stays useful to refer back to.
        await github_client.post_comment(
            job.repo_full_name,
            job.pr_number,
            comments.format_review(findings, job.review_text),
        )

    job.status = JobStatus.REVIEW_DONE


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _execute_builds(
    targets: list[BuildTarget],
    driver_paths: list[str],
    worktree: Path,
    config: Config,
) -> list[BuildResult]:
    """Build each detected target in sequence."""
    results = []
    for target in targets:
        if target == BuildTarget.CORE:
            results.append(await build_core(worktree, config))
        elif target == BuildTarget.PERCEPTION:
            results.append(await build_perception(worktree, config))
        elif target == BuildTarget.DRIVER:
            for dp in driver_paths:
                results.append(await build_driver(worktree, dp, config))
    return results


def _parse_forced_targets(
    force_targets: list[str],
) -> tuple[list[BuildTarget], list[str]]:
    """Resolve user-specified targets from the trigger command."""
    targets: list[BuildTarget] = []
    driver_paths: list[str] = []
    for t in force_targets:
        if t == "core":
            targets.append(BuildTarget.CORE)
        elif t == "perception":
            targets.append(BuildTarget.PERCEPTION)
        elif "/" in t:
            targets.append(BuildTarget.DRIVER)
            driver_paths.append(t)
    return list(dict.fromkeys(targets)), driver_paths


async def _report(job: ReviewJob, github_client: GitHubClient, body: str):
    """Update the job's status comment, falling back to posting a new one.

    Everything funnels through the acknowledgment comment created at trigger
    time, so a PR gets one comment tracking progress rather than one per stage.
    Reporting must never break the pipeline, so failures are logged only.
    """
    try:
        if job.progress_comment_id is not None:
            await github_client.edit_comment(
                job.repo_full_name, job.progress_comment_id, body
            )
        else:
            job.progress_comment_id = await github_client.post_comment(
                job.repo_full_name, job.pr_number, body
            )
    except Exception as e:
        logger.warning(f"Job {job.id}: failed to report status to GitHub: {e}")


async def _cleanup(job: ReviewJob, workspace_mgr: GitWorkspaceManager):
    """Remove the worktree so a retry starts from a clean tree."""
    if not job.worktree_path:
        return
    try:
        await workspace_mgr.remove_worktree(
            job.repo_full_name, Path(job.worktree_path)
        )
    except Exception as e:
        logger.warning(f"Job {job.id}: failed to remove worktree: {e}")
    finally:
        job.worktree_path = ""

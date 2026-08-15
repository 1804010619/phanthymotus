"""Worker — runs the review pipeline for one job, with timeout and retry.

Retry policy: a job that exceeds `job_timeout_seconds` is presumed lost and
retried, as are infrastructure failures (network, git, registry). Conditions
caused by the PR itself are terminal and reported immediately — a merge
conflict or a genuinely broken build is the answer, not something to retry.

Job state is written through to the store at each transition, so the dashboard
survives a restart and shows in-flight work as it happens.
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from . import comments
from .build_detector import detect_targets
from .builder import (
    build_core,
    build_driver,
    build_perception,
    container_name_from_service_yaml,
    log_filename,
    read_service_yaml,
    service_yaml_path,
)
from .config import Config
from .git_workspace import GitWorkspaceManager
from .github_client import GitHubClient
from .models import (
    BuildResult,
    BuildTarget,
    JobStatus,
    ReviewError,
    ReviewJob,
    Stage,
)
from .reviewer import llm_review, run_rule_checks
from .store import JobStore

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
    store: JobStore,
):
    """Run a job, retrying on timeout or infrastructure failure."""
    job.started_at = datetime.now(timezone.utc)

    for attempt in range(1, config.max_attempts + 1):
        job.attempt = attempt
        job.status = JobStatus.RUNNING
        await store.save_job(job)

        if attempt > 1:
            logger.info(
                f"Job {job.id}: attempt {attempt}/{config.max_attempts} "
                f"for {job.repo_full_name}#{job.pr_number}"
            )

        try:
            await asyncio.wait_for(
                _run_once(job, config, github_client, workspace_mgr, store),
                timeout=config.job_timeout_seconds,
            )
            # A terminal state was reached and reported inside the pipeline
            # (review posted, or a real build failure). Either way, done.
            job.finished_at = datetime.now(timezone.utc)
            await _cleanup(job, workspace_mgr)
            await store.save_job(job)
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
            # Agent is shutting down — do not retry, do not comment. The queue's
            # shutdown path notifies the PR and persists the final state.
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
            await store.save_job(job)
            await _report(
                job, github_client,
                comments.format_error(job.pr_head_sha, reason),
            )
            return

        if attempt < config.max_attempts:
            job.status = JobStatus.RETRYING
            await store.save_job(job)
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
            await store.save_job(job)
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
    store: JobStore,
):
    """One attempt at the full pipeline.

    Raises on failure. Returns normally once a terminal state has been reached
    and reported (review posted, or build failure reported).

    Each step advances `job.stage` and persists it. That is what makes a long
    fetch or a long build legible on the dashboard instead of showing "running"
    with nothing else for minutes.
    """
    base_ref = job.pr_base_ref or "main"

    # 1. Fetch just this PR's base branch and head ref
    job.set_stage(Stage.FETCHING, base_ref)
    await store.save_job(job)
    await workspace_mgr.fetch_for_pr(job.repo_full_name, job.pr_number, base_ref)

    # 2. Create an isolated worktree with the PR merged onto the base
    job.set_stage(Stage.WORKTREE, f"{base_ref} + PR #{job.pr_number}")
    await store.save_job(job)
    worktree = await workspace_mgr.create_worktree(
        job.repo_full_name, job.pr_number, job.pr_head_sha, base_ref
    )
    job.worktree_path = str(worktree)

    # 3. Determine what changed
    job.set_stage(Stage.DETECTING)
    await store.save_job(job)
    changed_files = await workspace_mgr.get_changed_files(worktree, base_ref)
    if not changed_files:
        await _report(
            job, github_client, comments.format_no_changes(job.pr_head_sha)
        )
        job.status = JobStatus.REVIEW_DONE
        job.set_stage(Stage.DONE)
        return

    if job.force_targets:
        targets, driver_paths = _parse_forced_targets(job.force_targets)
    else:
        # The worktree is probed for driver.yaml/Dockerfile, so newly added
        # vendors are picked up without editing a provider list.
        targets, driver_paths = detect_targets(
            job.repo_full_name, changed_files, worktree
        )

    # 4. Build
    if not job.skip_build and targets:
        await _report(
            job, github_client,
            comments.format_building(
                job.requester, job.pr_head_sha, targets, driver_paths
            ),
        )

        results = await _execute_builds(
            job, targets, driver_paths, worktree, config, store
        )
        job.build_results = results
        await store.save_job(job)

        await _report(
            job, github_client,
            comments.format_build_result(job.pr_head_sha, results),
        )

        if any(not r.success for r in results):
            # A real build failure — terminal and already reported. Not
            # retried: the author needs to fix the code, and rebuilding the
            # same commit twice more would just burn an hour saying the same.
            job.status = JobStatus.BUILD_FAILED
            job.set_stage(Stage.DONE)
            return

        job.status = JobStatus.BUILD_SUCCESS
        await store.save_job(job)

    elif not job.skip_build:
        await _report(
            job, github_client,
            comments.format_no_build_needed(job.pr_head_sha),
        )

    # 5. Review
    if not job.build_only:
        job.set_stage(Stage.RULE_CHECKS)
        await store.save_job(job)
        diff_stat = await workspace_mgr.get_diff_stat(worktree, base_ref)
        findings = run_rule_checks(changed_files, diff_stat)
        # Kept on the job (not just formatted into the comment) so the
        # dashboard can render them and they survive a restart.
        job.findings = [
            {"severity": f.severity, "file": f.file, "message": f.message}
            for f in findings
        ]

        job.set_stage(Stage.LLM_REVIEW, config.llm_model)
        await store.save_job(job)
        diff = await workspace_mgr.get_diff(
            worktree, base_ref, config.max_diff_lines
        )
        job.review_text = await llm_review(config, changed_files, diff, findings)

        job.set_stage(Stage.POSTING)
        await store.save_job(job)
        # The review is its own comment — the progress comment keeps the build
        # result, which stays useful to refer back to.
        await github_client.post_comment(
            job.repo_full_name,
            job.pr_number,
            comments.format_review(findings, job.review_text),
        )

    job.status = JobStatus.REVIEW_DONE
    job.set_stage(Stage.DONE)


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _execute_builds(
    job: ReviewJob,
    targets: list[BuildTarget],
    driver_paths: list[str],
    worktree: Path,
    config: Config,
    store: JobStore,
) -> list[BuildResult]:
    """Build each detected target in sequence, persisting each result.

    Each build streams to its own log file under the job's log directory, so the
    dashboard can tail an in-progress build and the full output outlives the
    80-line tail that goes into the PR comment.
    """
    log_dir = store.log_dir_for(job.id)
    log_dir.mkdir(parents=True, exist_ok=True)

    plan: list[tuple[BuildTarget, str | None]] = []
    for target in targets:
        if target == BuildTarget.DRIVER:
            plan.extend((target, dp) for dp in driver_paths)
        else:
            plan.append((target, None))

    results = []
    for idx, (target, driver_path) in enumerate(plan):
        label = driver_path or target.value
        log_path = log_dir / log_filename(idx, target, driver_path)

        job.set_stage(Stage.BUILDING, f"{idx + 1}/{len(plan)} {label}")
        await store.save_job(job)

        # Persist a placeholder row *before* the build runs. The dashboard
        # builds its log panes from build_results, so without this there is no
        # pane to tail until the build has already finished — which is exactly
        # when live tailing stops being useful.
        await store.save_build_result(
            job.id, idx,
            BuildResult(
                target=target,
                driver_path=driver_path,
                success=False,
                image_tag="",
                log_tail="",
                log_path=str(log_path),
            ),
        )

        if target == BuildTarget.CORE:
            result = await build_core(worktree, config, log_path)
        elif target == BuildTarget.PERCEPTION:
            result = await build_perception(worktree, config, log_path)
        else:
            result = await build_driver(worktree, driver_path, config, log_path)

        results.append(result)

        # Note which container the target declares, while the worktree still
        # exists. Its presence is what tells the comment that
        # deploy/run-pr-image.sh will work. Drivers and perception ship a
        # fragment; core does not (it self-updates via the web console).
        rel = service_yaml_path(target, driver_path)
        if result.success and result.image_tag and rel:
            svc = read_service_yaml(worktree, rel)
            if svc:
                result.container_name = container_name_from_service_yaml(svc)
            else:
                logger.info(f"{label}: no service.yml at {rel}")

        # Overwrites the placeholder (UNIQUE(job_id, idx) + INSERT OR REPLACE).
        await store.save_build_result(job.id, idx, result)

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

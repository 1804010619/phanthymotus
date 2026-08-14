"""Worker — orchestrates the full PR review pipeline for a single job."""

import logging
from datetime import datetime, timezone
from pathlib import Path

from .build_detector import detect_targets
from .builder import build_core, build_driver, build_perception
from .config import Config
from .github_client import GitHubClient
from .git_workspace import GitWorkspaceManager
from .models import BuildResult, BuildTarget, JobStatus, ReviewJob
from .reviewer import Finding, llm_review, run_rule_checks

logger = logging.getLogger(__name__)


async def run_job(
    job: ReviewJob,
    config: Config,
    github_client: GitHubClient,
    workspace_mgr: GitWorkspaceManager,
):
    """Execute the full review pipeline for a job."""
    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(timezone.utc)

    worktree_path: Path | None = None

    try:
        # 1. Fetch latest refs
        await workspace_mgr.fetch(job.repo_full_name)

        # 2. Create worktree (PR merged onto main)
        worktree_path = await workspace_mgr.create_worktree(
            job.repo_full_name, job.pr_number, job.pr_head_sha
        )
        job.worktree_path = str(worktree_path)

        # 3. Get changed files
        changed_files = await workspace_mgr.get_changed_files(worktree_path)
        if not changed_files:
            await github_client.post_comment(
                job.repo_full_name, job.pr_number,
                _format_no_changes(),
            )
            job.status = JobStatus.REVIEW_DONE
            return

        # 4. Detect build targets
        if job.force_targets:
            targets, driver_paths = _parse_forced_targets(job.force_targets)
        else:
            targets, driver_paths = detect_targets(job.repo_full_name, changed_files)

        # 5. Build phase
        if not job.skip_build and targets:
            # Post progress comment
            progress_body = _format_building(targets, driver_paths)
            job.progress_comment_id = await github_client.post_comment(
                job.repo_full_name, job.pr_number, progress_body
            )

            # Execute builds
            build_results = await _execute_builds(
                targets, driver_paths, worktree_path, config
            )
            job.build_results = build_results

            # Check for failures
            any_failed = any(not r.success for r in build_results)
            if any_failed:
                job.status = JobStatus.BUILD_FAILED
                await github_client.edit_comment(
                    job.repo_full_name, job.progress_comment_id,
                    _format_build_result(build_results, failed=True),
                )
                return

            # Update progress comment with success
            await github_client.edit_comment(
                job.repo_full_name, job.progress_comment_id,
                _format_build_result(build_results, failed=False),
            )
            job.status = JobStatus.BUILD_SUCCESS

        # 6. Review phase
        if not job.build_only:
            # Rule checks
            diff_stat = await workspace_mgr.get_diff_stat(worktree_path)
            findings = run_rule_checks(changed_files, diff_stat)

            # LLM review
            diff = await workspace_mgr.get_diff(worktree_path, config.max_diff_lines)
            review_text = await llm_review(config, changed_files, diff, findings)
            job.review_text = review_text

            # Post review comment
            review_body = _format_review(findings, review_text)
            await github_client.post_comment(
                job.repo_full_name, job.pr_number, review_body
            )

        job.status = JobStatus.REVIEW_DONE

    except RuntimeError as e:
        # Known errors (merge conflict, etc.)
        job.status = JobStatus.ERROR
        job.error = str(e)
        await github_client.post_comment(
            job.repo_full_name, job.pr_number,
            _format_error(str(e)),
        )
    except Exception as e:
        job.status = JobStatus.ERROR
        job.error = str(e)
        logger.exception(f"Job {job.id} failed")
        try:
            await github_client.post_comment(
                job.repo_full_name, job.pr_number,
                _format_error(f"Unexpected error: {e}"),
            )
        except Exception:
            pass
    finally:
        job.finished_at = datetime.now(timezone.utc)
        # Clean up worktree
        if worktree_path:
            try:
                await workspace_mgr.remove_worktree(job.repo_full_name, worktree_path)
            except Exception:
                logger.warning(f"Failed to clean up worktree: {worktree_path}")


def _parse_forced_targets(
    force_targets: list[str],
) -> tuple[list[BuildTarget], list[str]]:
    """Parse user-specified forced targets."""
    targets = []
    driver_paths = []
    for t in force_targets:
        if t == "core":
            targets.append(BuildTarget.CORE)
        elif t == "perception":
            targets.append(BuildTarget.PERCEPTION)
        elif "/" in t:
            # Assume driver path
            targets.append(BuildTarget.DRIVER)
            driver_paths.append(t)
    # Deduplicate
    targets = list(dict.fromkeys(targets))
    return targets, driver_paths


async def _execute_builds(
    targets: list[BuildTarget],
    driver_paths: list[str],
    worktree: Path,
    config: Config,
) -> list[BuildResult]:
    """Execute all builds for detected targets."""
    results = []

    for target in targets:
        if target == BuildTarget.CORE:
            result = await build_core(worktree, config)
            results.append(result)
        elif target == BuildTarget.PERCEPTION:
            result = await build_perception(worktree, config)
            results.append(result)
        elif target == BuildTarget.DRIVER:
            for dp in driver_paths:
                result = await build_driver(worktree, dp, config)
                results.append(result)

    return results


# ── Comment formatting ─────────────────────────────────────────────────────────

BOT_MARKER = "<!-- pr-review-agent -->"


def _format_building(targets: list[BuildTarget], driver_paths: list[str]) -> str:
    target_names = []
    for t in targets:
        if t == BuildTarget.DRIVER:
            target_names.extend(f"driver:{dp}" for dp in driver_paths)
        else:
            target_names.append(t.value)

    return f"""{BOT_MARKER}
## PR Review Agent — Building

Targets: {', '.join(f'`{n}`' for n in target_names)}

Status: building...
"""


def _format_build_result(results: list[BuildResult], failed: bool) -> str:
    rows = []
    for r in results:
        name = r.driver_path or r.target.value
        if r.success:
            rows.append(f"| {name} | :white_check_mark: | `{r.image_tag}` |")
        else:
            rows.append(f"| {name} | :x: Build failed | — |")

    table = "| Target | Status | Image |\n|--------|--------|-------|\n" + "\n".join(rows)

    body = f"""{BOT_MARKER}
## PR Review Agent — Build Result

{table}
"""

    # Append logs for failures
    for r in results:
        if not r.success and r.log_tail:
            name = r.driver_path or r.target.value
            body += f"""
<details><summary>{name} build log (last {len(r.log_tail.splitlines())} lines)</summary>

```
{r.log_tail}
```

</details>
"""

    return body


def _format_review(findings: list[Finding], review_text: str) -> str:
    body = f"""{BOT_MARKER}
## PR Review Agent — Code Review

{review_text}
"""

    if findings:
        body += "\n### Rule Checks\n\n"
        for f in findings:
            icon = {"error": ":x:", "warning": ":warning:", "info": ":information_source:"}.get(
                f.severity, ":grey_question:"
            )
            body += f"- {icon} `{f.file}` — {f.message}\n"

    body += "\n---\n<sub>Generated by PR Review Agent</sub>\n"
    return body


def _format_no_changes() -> str:
    return f"""{BOT_MARKER}
## PR Review Agent

No file changes detected relative to main. Nothing to build or review.
"""


def _format_error(error: str) -> str:
    return f"""{BOT_MARKER}
## PR Review Agent — Error

```
{error}
```
"""

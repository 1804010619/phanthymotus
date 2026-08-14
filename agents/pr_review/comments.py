"""PR comment formatting.

Lives in its own module so both `trigger` (acknowledgment) and `worker`
(progress / results) can format comments without importing each other.
"""

from .models import BuildResult, BuildTarget
from .reviewer import Finding

# Marker prefixed to every bot comment, so bot comments are identifiable.
BOT_MARKER = "<!-- pr-review-agent -->"

MODE_LABELS = {
    (False, False): "Build + Review",
    (True, False): "Review only (build skipped)",
    (False, True): "Build only (review skipped)",
    (True, True): "No-op",
}


def format_ack(
    requester: str,
    head_sha: str,
    skip_build: bool,
    build_only: bool,
    source: str,
) -> str:
    """Immediate acknowledgment posted when a job is accepted.

    This same comment is edited in place to show build progress and results,
    so a PR gets one comment that tracks progress rather than one per stage.
    """
    mode = MODE_LABELS.get((skip_build, build_only), "Build + Review")
    source_label = "polling" if source == "poll" else "webhook"

    return f"""{BOT_MARKER}
## PR Review Agent

Request from @{requester} accepted — starting review.

| | |
|---|---|
| Commit | `{head_sha[:7]}` |
| Mode | {mode} |
| Triggered via | {source_label} |
| Status | Queued |
"""


def format_building(
    requester: str,
    head_sha: str,
    targets: list[BuildTarget],
    driver_paths: list[str],
) -> str:
    """Build-in-progress state."""
    return f"""{BOT_MARKER}
## PR Review Agent

| | |
|---|---|
| Commit | `{head_sha[:7]}` |
| Build targets | {_target_list(targets, driver_paths)} |
| Status | Building... |

Builds usually take 5–20 minutes. This comment will be updated when done.
"""


def format_build_result(head_sha: str, results: list[BuildResult]) -> str:
    """Final build state — success or failure, with logs for failures."""
    rows = []
    for r in results:
        name = r.driver_path or r.target.value
        if r.success:
            image = f"`{r.image_tag}`" if r.image_tag else "built (tag not captured)"
            rows.append(f"| {name} | :white_check_mark: Success | {image} |")
        else:
            rows.append(f"| {name} | :x: Failed | — |")

    table = (
        "| Target | Status | Image |\n"
        "|--------|--------|-------|\n" + "\n".join(rows)
    )

    all_ok = all(r.success for r in results)
    headline = (
        "All builds succeeded. Image tags are listed below."
        if all_ok
        else "Build failed. See the collapsed logs below."
    )

    body = f"""{BOT_MARKER}
## PR Review Agent — Build Result

Commit: `{head_sha[:7]}`

{headline}

{table}
"""

    for r in results:
        if not r.success and r.log_tail:
            name = r.driver_path or r.target.value
            line_count = len(r.log_tail.splitlines())
            body += f"""
<details><summary>{name} build log (last {line_count} lines)</summary>

```
{r.log_tail}
```

</details>
"""

    if not all_ok:
        body += (
            "\nPush a fix and comment `/request_bot_review` again to retrigger.\n"
        )

    return body


def format_no_build_needed(head_sha: str) -> str:
    """No buildable changes detected — proceeding straight to review."""
    return f"""{BOT_MARKER}
## PR Review Agent

| | |
|---|---|
| Commit | `{head_sha[:7]}` |
| Build targets | None (changes do not touch a buildable component) |
| Status | Generating review... |
"""


def format_review(findings: list[Finding], review_text: str) -> str:
    """The substantive code review, posted as its own comment."""
    body = f"""{BOT_MARKER}
## PR Review Agent — Code Review

{review_text}
"""

    if findings:
        body += "\n### Rule Checks\n\n"
        for f in findings:
            icon = {
                "error": ":x:",
                "warning": ":warning:",
                "info": ":information_source:",
            }.get(f.severity, ":grey_question:")
            body += f"- {icon} `{f.file}` — {f.message}\n"

    body += "\n---\n<sub>Generated automatically by PR Review Agent.</sub>\n"
    return body


def format_no_changes(head_sha: str) -> str:
    return f"""{BOT_MARKER}
## PR Review Agent

Commit `{head_sha[:7]}` has no file changes relative to main — nothing to
build or review.
"""


def format_superseded(old_sha: str, new_sha: str) -> str:
    """A queued job dropped because a newer request arrived for the same PR."""
    return f"""{BOT_MARKER}
## PR Review Agent — Superseded

Queued review of `{old_sha[:7]}` was dropped because a newer request arrived
for `{new_sha[:7]}`. See the newer comment for status.
"""


def format_retrying(
    head_sha: str,
    attempt: int,
    max_attempts: int,
    reason: str,
    backoff_seconds: int,
) -> str:
    """Shown between attempts when a job failed for a retryable reason."""
    return f"""{BOT_MARKER}
## PR Review Agent — Retrying

| | |
|---|---|
| Commit | `{head_sha[:7]}` |
| Attempt | {attempt} of {max_attempts} failed |
| Status | Retrying in {backoff_seconds}s... |

<details><summary>Failure reason</summary>

```
{reason}
```

</details>
"""


def format_final_failure(
    head_sha: str,
    max_attempts: int,
    attempt_errors: list[str],
) -> str:
    """All attempts exhausted — a real failure the author needs to see."""
    body = f"""{BOT_MARKER}
## PR Review Agent — Failed

Commit `{head_sha[:7]}` did not complete after {max_attempts} attempts.

Likely causes: build environment problem, network failure, unreachable
registry, or a build that exceeds the per-job time limit.
"""

    for i, err in enumerate(attempt_errors, start=1):
        body += f"""
<details><summary>Attempt {i} failure reason</summary>

```
{err}
```

</details>
"""

    body += "\nOnce resolved, comment `/request_bot_review` again to retrigger.\n"
    return body


def format_error(head_sha: str, error: str) -> str:
    """A terminal, non-retryable error (e.g. merge conflict)."""
    return f"""{BOT_MARKER}
## PR Review Agent — Error

Commit: `{head_sha[:7]}`

```
{error}
```

Push a fix and comment `/request_bot_review` again to retrigger.
"""


def _target_list(targets: list[BuildTarget], driver_paths: list[str]) -> str:
    names = []
    for t in targets:
        if t == BuildTarget.DRIVER:
            names.extend(driver_paths)
        else:
            names.append(t.value)
    return ", ".join(f"`{n}`" for n in names) if names else "None"

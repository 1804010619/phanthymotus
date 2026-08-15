"""PR comment formatting.

Lives in its own module so both `trigger` (acknowledgment) and `worker`
(progress / results) can format comments without importing each other.
"""

from .builder import split_image_ref
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
            _, tag = split_image_ref(r.image_tag)
            version = f"`{tag}`" if tag else "—"
            rows.append(f"| {name} | :white_check_mark: Success | {version} |")
        else:
            rows.append(f"| {name} | :x: Failed | — |")

    table = (
        "| Target | Status | Version |\n"
        "|--------|--------|---------|\n" + "\n".join(rows)
    )

    all_ok = all(r.success for r in results)
    headline = (
        "All builds succeeded."
        if all_ok
        else "Build failed. See the collapsed logs below."
    )

    body = f"""{BOT_MARKER}
## PR Review Agent — Build Result

Commit: `{head_sha[:7]}`

{headline}

{table}
"""

    # Full refs, so the image can be pulled or deployed without hunting through
    # the log for it.
    built = [r for r in results if r.success and r.image_tag]
    if built:
        body += "\n### Images\n\n"
        for r in built:
            name = r.driver_path or r.target.value
            body += f"**{name}**\n```\n{r.image_tag}\n```\n"
        body += _deploy_help(built)

    missing_ref = [r for r in results if r.success and not r.image_tag]
    if missing_ref:
        names = ", ".join(r.driver_path or r.target.value for r in missing_ref)
        body += (
            f"\n> Built successfully, but the image reference could not be read "
            f"from the build log for: {names}. Check the full log on the "
            f"dashboard.\n"
        )

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


def _deploy_help(built: list[BuildResult]) -> str:
    """How to run a freshly built image.

    Leads with a single `docker run`, because the common case is a reviewer
    wanting to try the build on a spare machine and throw it away — the full
    compose-merge procedure is too much ceremony for that. The command is
    translated from the driver's own `deploy/service.yml`, not invented, so the
    privileged/host-network/device flags the driver actually needs are present.
    """
    out = "\n### Try it\n"

    for r in built:
        name = r.driver_path or r.target.value
        if not r.run_command:
            continue
        cn = r.container_name or "the-container"
        out += f"""
<details open><summary><b>{name}</b> — run it directly</summary>

```bash
{r.run_command}
```

Follow the logs, get a shell, then clean up:

```bash
docker logs -f {cn}
docker exec -it {cn} bash
docker rm -f {cn}
```

</details>
"""

    # Anything without a translated command (core, perception, or a driver with
    # no service.yml) still gets the pull, since that is all we can honestly say.
    plain = [r for r in built if not r.run_command]
    if plain:
        refs = "\n".join(r.image_tag for r in plain)
        out += f"""
```bash
docker pull {plain[0].image_tag}
```
"""
        if len(plain) > 1:
            out += f"\n<details><summary>All images</summary>\n\n```\n{refs}\n```\n\n</details>\n"

    out += """
For a real deployment, use the Agent Core dashboard instead — set the driver's
image to the reference above and deploy. It merges the same `service.yml` into
the host's compose file so the container survives reboots.

Once this PR is approved, the version becomes installable from the web console.
"""
    return out


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


def format_interrupted(head_sha: str, was_running: bool) -> str:
    """The agent shut down before this job finished.

    Restarts are operational events the author cannot infer from a comment
    frozen at "Building...", so say plainly what happened and what to do.
    """
    what = "was interrupted mid-run" if was_running else "never started"
    return f"""{BOT_MARKER}
## PR Review Agent — Interrupted

Review of `{head_sha[:7]}` {what} because the agent was stopped or restarted.

Comment `/request_bot_review` again to retrigger.
"""


def format_superseded(old_sha: str, new_sha: str) -> str:
    """A queued job dropped because a newer request arrived for the same PR."""
    return f"""{BOT_MARKER}
## PR Review Agent — Superseded

Queued review of `{old_sha[:7]}` was dropped because a newer request arrived
for `{new_sha[:7]}`. See the newer comment for status.
"""


def format_skipped_in_flight(head_sha: str) -> str:
    """A repeat trigger arrived for a commit that is already being reviewed."""
    return f"""{BOT_MARKER}
## PR Review Agent — Already in progress

A review of `{head_sha[:7]}` is already running. This request was skipped rather
than starting a second build of the same commit.

Push a new commit to review the change, or use
`/request_bot_review force` to re-run this one.
"""


def format_skipped_already_reviewed(
    head_sha: str, status: str, finished_at: float | None
) -> str:
    """A repeat trigger arrived for a commit that was already reviewed."""
    when = ""
    if finished_at:
        from datetime import datetime, timezone
        ts = datetime.fromtimestamp(finished_at, tz=timezone.utc)
        when = f" on {ts.strftime('%Y-%m-%d %H:%M')} UTC"

    return f"""{BOT_MARKER}
## PR Review Agent — Already reviewed

`{head_sha[:7]}` was already reviewed{when} (result: `{status}`). This request
was skipped rather than rebuilding an unchanged commit.

- Pushed a fix? The new commit will be reviewed when you trigger again.
- Want this commit re-reviewed anyway? Use `/request_bot_review force`.
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

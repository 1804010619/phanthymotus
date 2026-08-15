"""Data models and errors for PR review jobs."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


# ── Errors ────────────────────────────────────────────────────────────────────


class ReviewError(Exception):
    """Base for pipeline errors.

    `retryable` decides whether the worker tries again. Conditions caused by
    the PR itself (merge conflict) are terminal — retrying just spends another
    hour reporting the same thing. Infrastructure problems (network, git,
    registry, timeouts) are worth another attempt.
    """

    retryable = True


class MergeConflictError(ReviewError):
    """PR cannot be merged onto main — the author must resolve it."""

    retryable = False


class JobTimeoutError(ReviewError):
    """Job exceeded its whole-job wall-clock budget and is presumed lost."""

    retryable = True


# ── Enums ─────────────────────────────────────────────────────────────────────


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    RETRYING = "retrying"
    BUILD_SUCCESS = "build_success"
    BUILD_FAILED = "build_failed"
    REVIEW_DONE = "review_done"
    TIMEOUT = "timeout"
    ERROR = "error"
    CANCELLED = "cancelled"


class BuildTarget(str, Enum):
    CORE = "core"
    PERCEPTION = "perception"
    DRIVER = "driver"


# Statuses from which a job will never advance. Plain strings, because the
# store hands back rows where status is a string, not an enum member.
TERMINAL_STATUSES = frozenset({
    JobStatus.BUILD_FAILED.value,
    JobStatus.REVIEW_DONE.value,
    JobStatus.TIMEOUT.value,
    JobStatus.ERROR.value,
    JobStatus.CANCELLED.value,
})

# Terminal statuses that actually produced an answer for the requester. Only
# these should block a repeat trigger on the same commit.
#
# The rest — cancelled, timeout, error — are terminal but delivered nothing, so
# re-triggering is the correct response to them, not something to refuse. A job
# killed by a restart or by an infrastructure failure must not leave a commit
# permanently un-reviewable.
CONCLUSIVE_STATUSES = frozenset({
    JobStatus.REVIEW_DONE.value,
    JobStatus.BUILD_FAILED.value,
})


class Stage(str, Enum):
    """Where in the pipeline a running job is.

    `status` alone is too coarse: a job sits at RUNNING through a fetch, a
    worktree merge, several builds, and the review. Without this the dashboard
    shows "running" with nothing else for minutes at a time.
    """

    QUEUED = "queued"
    FETCHING = "fetching refs"
    WORKTREE = "preparing worktree"
    DETECTING = "detecting changes"
    BUILDING = "building"
    RULE_CHECKS = "running rule checks"
    LLM_REVIEW = "generating review"
    POSTING = "posting results"
    DONE = "done"


# ── Data ──────────────────────────────────────────────────────────────────────


@dataclass
class BuildResult:
    target: BuildTarget
    driver_path: str | None  # e.g. "unitree/g1", only for DRIVER
    success: bool
    image_tag: str  # full image ref when successful
    log_tail: str  # last N lines, for the PR comment
    log_path: str = ""  # full log on disk, for the dashboard
    # Container the target's deploy/service.yml declares. Set only when the
    # target ships a parseable fragment, so it doubles as the signal that
    # deploy/run-pr-image.sh will work for this image.
    container_name: str = ""


@dataclass
class ReviewJob:
    repo_full_name: str  # "4paradigm/phanthymotus"
    pr_number: int
    pr_head_sha: str
    pr_head_ref: str  # branch name
    pr_base_ref: str  # e.g. "main"
    comment_id: int  # triggering comment
    requester: str  # GitHub username
    source: str = "webhook"  # "webhook" | "poll"

    # Options parsed from the command
    skip_build: bool = False
    build_only: bool = False
    force_targets: list[str] = field(default_factory=list)  # e.g. ["core"]

    # Runtime state
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: JobStatus = JobStatus.QUEUED
    # Finer-grained progress within RUNNING, plus a free-text detail such as
    # "1/2 unitree/g1" so the dashboard can say what is being built.
    stage: str = Stage.QUEUED.value
    stage_detail: str = ""
    stage_started_at: datetime | None = None
    attempt: int = 0
    attempt_errors: list[str] = field(default_factory=list)
    build_results: list[BuildResult] = field(default_factory=list)
    review_text: str = ""
    # Rule-check findings as plain dicts, so they survive persistence and can
    # be rendered by the dashboard rather than only formatted into a comment.
    findings: list[dict] = field(default_factory=list)
    error: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    worktree_path: str = ""
    # Acknowledgment comment, edited in place through the job's lifetime.
    progress_comment_id: int | None = None

    def elapsed_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds()

    def stage_elapsed_seconds(self) -> float | None:
        """How long the current stage has been running.

        Surfaced so a stage that is taking unusually long (a slow fetch, a long
        build) is visible as such rather than looking like a hang.
        """
        if self.stage_started_at is None:
            return None
        end = self.finished_at or datetime.now(timezone.utc)
        return (end - self.stage_started_at).total_seconds()

    def set_stage(self, stage: "Stage", detail: str = ""):
        self.stage = stage.value
        self.stage_detail = detail
        self.stage_started_at = datetime.now(timezone.utc)


# ── Command parsing ───────────────────────────────────────────────────────────

TRIGGER = "/request_bot_review"


def parse_trigger_command(comment_body: str) -> dict | None:
    """Parse a `/request_bot_review` command out of a comment body.

    Returns {"skip_build", "build_only", "force_targets"}, or None when the
    comment does not contain the trigger.
    """
    if not comment_body:
        return None

    for raw_line in comment_body.splitlines():
        # Tolerate markdown quote/list prefixes, but require the trigger to
        # start the line so it is not picked up from surrounding prose.
        line = raw_line.strip().lstrip(">*- \t")
        if not line.lower().startswith(TRIGGER):
            continue

        args = line[len(TRIGGER):].split()
        result = {
            "skip_build": False,
            "build_only": False,
            "force": False,
            "force_targets": [],
        }
        for arg in args:
            token = arg.strip().strip("`,")
            lowered = token.lower()
            if lowered == "skip-build":
                result["skip_build"] = True
            elif lowered == "build-only":
                result["build_only"] = True
            elif lowered in ("force", "--force", "-f"):
                # Re-review a commit that was already reviewed.
                result["force"] = True
            elif lowered in ("core", "perception"):
                result["force_targets"].append(lowered)
            elif "/" in token:
                # A driver path such as unitree/g1
                result["force_targets"].append(token)
        return result

    return None

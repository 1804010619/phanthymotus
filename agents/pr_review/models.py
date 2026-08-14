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


# ── Data ──────────────────────────────────────────────────────────────────────


@dataclass
class BuildResult:
    target: BuildTarget
    driver_path: str | None  # e.g. "unitree/g1", only for DRIVER
    success: bool
    image_tag: str  # full image ref when successful
    log_tail: str  # last N lines of build output


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
    attempt: int = 0
    attempt_errors: list[str] = field(default_factory=list)
    build_results: list[BuildResult] = field(default_factory=list)
    review_text: str = ""
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
        result = {"skip_build": False, "build_only": False, "force_targets": []}
        for arg in args:
            token = arg.strip().strip("`,")
            lowered = token.lower()
            if lowered == "skip-build":
                result["skip_build"] = True
            elif lowered == "build-only":
                result["build_only"] = True
            elif lowered in ("core", "perception"):
                result["force_targets"].append(lowered)
            elif "/" in token:
                # A driver path such as unitree/g1
                result["force_targets"].append(token)
        return result

    return None

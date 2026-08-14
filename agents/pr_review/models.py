"""Data models for PR review jobs."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    BUILD_SUCCESS = "build_success"
    BUILD_FAILED = "build_failed"
    REVIEW_DONE = "review_done"
    ERROR = "error"
    CANCELLED = "cancelled"


class BuildTarget(str, Enum):
    CORE = "core"
    PERCEPTION = "perception"
    DRIVER = "driver"


@dataclass
class BuildResult:
    target: BuildTarget
    driver_path: str | None  # e.g. "unitree/g1", only for DRIVER
    success: bool
    image_tag: str  # full image ref if success
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
    # Options parsed from command
    skip_build: bool = False
    build_only: bool = False
    force_targets: list[str] = field(default_factory=list)  # e.g. ["core"]

    # Runtime state
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: JobStatus = JobStatus.QUEUED
    build_results: list[BuildResult] = field(default_factory=list)
    review_text: str = ""
    error: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    worktree_path: str = ""
    progress_comment_id: int | None = None

    def elapsed_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds()


def parse_trigger_command(comment_body: str) -> dict | None:
    """Parse /request_bot_review command from comment body.

    Returns dict with keys: skip_build, build_only, force_targets
    or None if no trigger found.
    """
    lines = comment_body.strip().splitlines()
    for line in lines:
        line = line.strip()
        if not line.lower().startswith("/request_bot_review"):
            continue
        parts = line.split()[1:]  # args after command
        result = {"skip_build": False, "build_only": False, "force_targets": []}
        for part in parts:
            p = part.lower().strip()
            if p == "skip-build":
                result["skip_build"] = True
            elif p == "build-only":
                result["build_only"] = True
            elif p in ("core", "perception"):
                result["force_targets"].append(p)
            else:
                # Could be a driver path like "unitree/g1"
                if "/" in part:
                    result["force_targets"].append(part)
        return result
    return None

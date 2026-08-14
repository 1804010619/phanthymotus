"""Job queue — asyncio queue with a configurable worker pool.

Concurrency is bounded by `max_concurrent_jobs`. Isolation between concurrent
jobs comes from git worktrees: each job checks out into its own directory, so
two builds never share a working tree. The only shared state is the bare clone
(fetch-only, guarded by a per-repo lock) and the Docker daemon (which
serializes internally).
"""

import asyncio
import logging
from collections import deque

from . import comments
from .config import Config
from .git_workspace import GitWorkspaceManager
from .github_client import GitHubClient
from .models import JobStatus, ReviewJob
from .worker import run_job

logger = logging.getLogger(__name__)

# Retained job history for the /jobs endpoints.
JOB_HISTORY_MAX = 100


class JobQueue:
    """Async job queue with N concurrent workers."""

    def __init__(
        self,
        max_workers: int,
        config: Config,
        github_client: GitHubClient,
        workspace_mgr: GitWorkspaceManager,
    ):
        self._max_workers = max_workers
        self._config = config
        self._github_client = github_client
        self._workspace_mgr = workspace_mgr
        self._queue: asyncio.Queue[ReviewJob] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._jobs: deque[ReviewJob] = deque(maxlen=JOB_HISTORY_MAX)
        self._active: dict[str, ReviewJob] = {}
        self.total_processed = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self):
        for i in range(self._max_workers):
            self._workers.append(
                asyncio.create_task(self._worker_loop(i), name=f"worker-{i}")
            )
        logger.info(f"Started {self._max_workers} workers")

    async def stop(self):
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    # ── Enqueue / introspection ───────────────────────────────────────────────

    async def enqueue(self, job: ReviewJob):
        """Add a job, superseding queued jobs for the same PR."""
        await self._supersede_queued_for_pr(job)
        self._jobs.append(job)
        await self._queue.put(job)

    def has_pending_job(self, repo: str, pr_number: int, sha: str) -> bool:
        """True if this exact commit is already queued or running."""
        return any(
            j.repo_full_name == repo
            and j.pr_number == pr_number
            and j.pr_head_sha == sha
            and j.status in (JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.RETRYING)
            for j in self._jobs
        )

    def queue_size(self) -> int:
        return self._queue.qsize()

    def active_count(self) -> int:
        return len(self._active)

    def recent_jobs(self, limit: int = 20) -> list[ReviewJob]:
        return list(self._jobs)[-limit:]

    def get_job(self, job_id: str) -> ReviewJob | None:
        for job in self._jobs:
            if job.id == job_id:
                return job
        return None

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _supersede_queued_for_pr(self, new_job: ReviewJob):
        """Cancel still-queued jobs for the same PR.

        Only queued jobs are cancelled. A job already running is left alone —
        its build is expensive and its result is still valid for the commit it
        started on.
        """
        for job in self._jobs:
            if (
                job.repo_full_name == new_job.repo_full_name
                and job.pr_number == new_job.pr_number
                and job.status == JobStatus.QUEUED
            ):
                job.status = JobStatus.CANCELLED
                job.error = f"Superseded by newer request ({new_job.pr_head_sha[:7]})"
                logger.info(
                    f"Superseded queued job {job.id} for PR #{job.pr_number}"
                )
                # Leave the PR in a truthful state rather than a stale
                # "Queued" comment that will never advance.
                if job.progress_comment_id is not None:
                    try:
                        await self._github_client.edit_comment(
                            job.repo_full_name,
                            job.progress_comment_id,
                            comments.format_superseded(
                                job.pr_head_sha, new_job.pr_head_sha
                            ),
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update superseded comment: {e}")

    async def _worker_loop(self, worker_id: int):
        while True:
            try:
                job = await self._queue.get()
            except asyncio.CancelledError:
                break

            try:
                if job.status == JobStatus.CANCELLED:
                    continue

                self._active[job.id] = job
                try:
                    # run_job owns its own timeout and retry policy.
                    await run_job(
                        job,
                        config=self._config,
                        github_client=self._github_client,
                        workspace_mgr=self._workspace_mgr,
                    )
                    self.total_processed += 1
                except asyncio.CancelledError:
                    # Shutdown mid-job.
                    raise
                except Exception as e:
                    # run_job is supposed to absorb its own failures; anything
                    # reaching here is a bug, so record it and keep serving.
                    logger.exception(f"Worker-{worker_id}: job {job.id} escaped")
                    job.status = JobStatus.ERROR
                    job.error = f"{type(e).__name__}: {e}"
                finally:
                    self._active.pop(job.id, None)
            except asyncio.CancelledError:
                break
            finally:
                self._queue.task_done()

"""Job queue — asyncio-based queue with configurable worker pool."""

import asyncio
import logging
from collections import deque

from .config import Config
from .github_client import GitHubClient
from .git_workspace import GitWorkspaceManager
from .models import JobStatus, ReviewJob
from .worker import run_job

logger = logging.getLogger(__name__)


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
        self._jobs: deque[ReviewJob] = deque(maxlen=100)  # recent history
        self._active: dict[str, ReviewJob] = {}
        self.total_processed = 0

    async def start(self):
        """Start worker tasks."""
        for i in range(self._max_workers):
            task = asyncio.create_task(self._worker_loop(i), name=f"worker-{i}")
            self._workers.append(task)
        logger.info(f"Started {self._max_workers} workers")

    async def stop(self):
        """Stop all workers gracefully."""
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def enqueue(self, job: ReviewJob):
        """Add a job to the queue. Cancels pending jobs for same PR."""
        # Cancel pending (not running) jobs for same PR
        self._cancel_pending_for_pr(job.repo_full_name, job.pr_number)
        self._jobs.append(job)
        await self._queue.put(job)

    def has_pending_job(self, repo: str, pr_number: int, sha: str) -> bool:
        """Check if a job for this repo+PR+SHA is already queued or running."""
        for job in self._jobs:
            if (
                job.repo_full_name == repo
                and job.pr_number == pr_number
                and job.pr_head_sha == sha
                and job.status in (JobStatus.QUEUED, JobStatus.RUNNING)
            ):
                return True
        return False

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

    def _cancel_pending_for_pr(self, repo: str, pr_number: int):
        """Cancel queued (not yet running) jobs for the same PR."""
        for job in self._jobs:
            if (
                job.repo_full_name == repo
                and job.pr_number == pr_number
                and job.status == JobStatus.QUEUED
            ):
                job.status = JobStatus.CANCELLED
                logger.info(f"Cancelled superseded job {job.id} for PR #{pr_number}")

    async def _worker_loop(self, worker_id: int):
        """Worker loop — picks jobs from queue and processes them."""
        while True:
            try:
                job = await self._queue.get()

                # Skip cancelled jobs
                if job.status == JobStatus.CANCELLED:
                    self._queue.task_done()
                    continue

                self._active[job.id] = job
                try:
                    await run_job(
                        job,
                        config=self._config,
                        github_client=self._github_client,
                        workspace_mgr=self._workspace_mgr,
                    )
                    self.total_processed += 1
                except Exception as e:
                    logger.exception(f"Worker-{worker_id}: unhandled error in job {job.id}")
                    job.status = JobStatus.ERROR
                    job.error = str(e)
                finally:
                    self._active.pop(job.id, None)
                    self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception(f"Worker-{worker_id}: loop error")
                await asyncio.sleep(1)

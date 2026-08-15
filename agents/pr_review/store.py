"""SQLite persistence for review jobs.

Mirrors the pattern in agent-core's `src/perf_log.py`: fresh connection per
call, DDL re-applied idempotently, `row_factory = sqlite3.Row`, explicit
`close()`, and age-based `prune()`. Deliberately not agent-core's
`task_store.py`, which stores one opaque JSON blob per row and deletes terminal
rows — that makes filtering, sorting, and history impossible.

Two departures from the house pattern, both because this process has real
concurrency (N build workers + the poller + dashboard reads) where agent-core's
config reads are mostly serial:

- WAL journal mode and a busy timeout. agent-core omits both, and several of its
  call sites swallow "database is locked" in a bare except. With workers writing
  while the dashboard reads, that would surface here.
- Writes and queries go through `asyncio.to_thread()`. sqlite3 is synchronous;
  blocking the event loop would stall the poller and every in-flight log pump.
"""

import asyncio
import json
import logging
import shutil
import sqlite3
import time
from pathlib import Path

from .models import BuildResult, ReviewJob

logger = logging.getLogger(__name__)

DB_FILENAME = "jobs.db"
LOGS_DIRNAME = "logs"


class JobStore:
    """Persistent job history backed by SQLite, with build logs on disk."""

    def __init__(self, data_dir: str):
        self._data_dir = Path(data_dir)
        self._db_path = self._data_dir / DB_FILENAME
        self._logs_dir = self._data_dir / LOGS_DIRNAME

    # ── Setup ─────────────────────────────────────────────────────────────────

    @property
    def logs_dir(self) -> Path:
        return self._logs_dir

    def log_dir_for(self, job_id: str) -> Path:
        return self._logs_dir / job_id

    def init(self):
        """Create directories and schema. Safe to call repeatedly."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            self._migrate(conn)
            conn.commit()
        finally:
            conn.close()
        logger.info(f"Job store ready at {self._db_path}")

    @staticmethod
    def _migrate(conn: sqlite3.Connection):
        """Add columns introduced after a deployment already created its DB.

        `CREATE TABLE IF NOT EXISTS` is a no-op on an existing table, so new
        columns have to be added explicitly or every INSERT fails against an
        older database. Following the house style (agent-core has no migration
        ledger), this is an idempotent fixup rather than a versioned migration.
        """
        existing = {
            row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
        }
        for column, ddl in (
            ("stage", "ALTER TABLE jobs ADD COLUMN stage TEXT"),
            ("stage_detail", "ALTER TABLE jobs ADD COLUMN stage_detail TEXT"),
            ("stage_started_at",
             "ALTER TABLE jobs ADD COLUMN stage_started_at REAL"),
        ):
            if column not in existing:
                conn.execute(ddl)
                logger.info(f"Migrated jobs table: added {column}")

        br_existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(build_results)").fetchall()
        }
        for column, ddl in (
            ("run_command",
             "ALTER TABLE build_results ADD COLUMN run_command TEXT"),
            ("container_name",
             "ALTER TABLE build_results ADD COLUMN container_name TEXT"),
        ):
            if column not in br_existing:
                conn.execute(ddl)
                logger.info(f"Migrated build_results table: added {column}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        # WAL lets the dashboard read while a worker writes; without it
        # concurrent access produces "database is locked".
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ── Writes ────────────────────────────────────────────────────────────────

    async def save_job(self, job: ReviewJob):
        """Upsert a job row. Called at each status transition."""
        try:
            await asyncio.to_thread(self._save_job_sync, job)
        except Exception as e:
            # Persistence failing must never break the review pipeline.
            logger.warning(f"Failed to persist job {job.id}: {e}")

    def _save_job_sync(self, job: ReviewJob):
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO jobs (
                  id, repo, pr_number, head_sha, head_ref, base_ref,
                  requester, source, status, stage, stage_detail,
                  stage_started_at, attempt,
                  skip_build, build_only, force_targets,
                  review_text, findings, error, attempt_errors,
                  created_at, started_at, finished_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    job.id,
                    job.repo_full_name,
                    job.pr_number,
                    job.pr_head_sha,
                    job.pr_head_ref,
                    job.pr_base_ref,
                    job.requester,
                    job.source,
                    job.status.value,
                    job.stage,
                    job.stage_detail,
                    job.stage_started_at.timestamp() if job.stage_started_at else None,
                    job.attempt,
                    int(job.skip_build),
                    int(job.build_only),
                    json.dumps(job.force_targets),
                    job.review_text,
                    json.dumps(job.findings),
                    job.error,
                    json.dumps(job.attempt_errors),
                    job.created_at.timestamp(),
                    job.started_at.timestamp() if job.started_at else None,
                    job.finished_at.timestamp() if job.finished_at else None,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    async def save_build_result(self, job_id: str, idx: int, result: BuildResult):
        try:
            await asyncio.to_thread(self._save_build_result_sync, job_id, idx, result)
        except Exception as e:
            logger.warning(f"Failed to persist build result {job_id}[{idx}]: {e}")

    def _save_build_result_sync(self, job_id: str, idx: int, result: BuildResult):
        conn = self._connect()
        try:
            # INSERT OR REPLACE against UNIQUE(job_id, idx): a retried job
            # re-runs every build, so the latest attempt overwrites rather than
            # accumulating a duplicate row per attempt.
            conn.execute(
                """
                INSERT OR REPLACE INTO build_results (
                  job_id, idx, target, driver_path,
                  success, image_tag, log_path,
                  run_command, container_name, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    job_id,
                    idx,
                    result.target.value,
                    result.driver_path,
                    int(result.success),
                    result.image_tag,
                    result.log_path,
                    result.run_command,
                    result.container_name,
                    time.time(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    # ── Reads ─────────────────────────────────────────────────────────────────

    async def list_jobs(
        self,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
        repo: str | None = None,
    ) -> tuple[list[dict], int]:
        """Return (rows, total_matching) for the history view."""
        return await asyncio.to_thread(
            self._list_jobs_sync, limit, offset, status, repo
        )

    def _list_jobs_sync(
        self, limit: int, offset: int, status: str | None, repo: str | None
    ) -> tuple[list[dict], int]:
        where, params = [], []
        if status:
            where.append("status = ?")
            params.append(status)
        if repo:
            where.append("repo = ?")
            params.append(repo)
        clause = f"WHERE {' AND '.join(where)}" if where else ""

        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            total = conn.execute(
                f"SELECT COUNT(*) FROM jobs {clause}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"""SELECT * FROM jobs {clause}
                    ORDER BY created_at DESC LIMIT ? OFFSET ?""",
                (*params, limit, offset),
            ).fetchall()
            jobs = [self._row_to_summary(r) for r in rows]

            # Attach build targets so the table can show them without N+1
            # per-row queries.
            if jobs:
                ids = [j["id"] for j in jobs]
                marks = ",".join("?" * len(ids))
                br = conn.execute(
                    f"""SELECT job_id, idx, target, driver_path, success, image_tag
                        FROM build_results WHERE job_id IN ({marks})
                        ORDER BY job_id, idx""",
                    ids,
                ).fetchall()
                by_job: dict[str, list[dict]] = {}
                for row in br:
                    by_job.setdefault(row["job_id"], []).append({
                        "idx": row["idx"],
                        "target": row["target"],
                        "driver_path": row["driver_path"],
                        "success": bool(row["success"]),
                        "image_tag": row["image_tag"],
                    })
                for j in jobs:
                    j["build_results"] = by_job.get(j["id"], [])
            return jobs, total
        finally:
            conn.close()

    async def get_job(self, job_id: str) -> dict | None:
        return await asyncio.to_thread(self._get_job_sync, job_id)

    async def find_jobs_for_commit(
        self, repo: str, pr_number: int, head_sha: str
    ) -> list[dict]:
        """Jobs already recorded for this exact commit, newest first.

        Used to decide what a repeated `/request_bot_review` should do. Reads
        SQLite rather than the in-memory queue, because that queue is empty
        after a restart and would let a completed review be redone silently.
        """
        return await asyncio.to_thread(
            self._find_jobs_for_commit_sync, repo, pr_number, head_sha
        )

    def _find_jobs_for_commit_sync(
        self, repo: str, pr_number: int, head_sha: str
    ) -> list[dict]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """SELECT * FROM jobs
                   WHERE repo = ? AND pr_number = ? AND head_sha = ?
                   ORDER BY created_at DESC""",
                (repo, pr_number, head_sha),
            ).fetchall()
            return [self._row_to_summary(r) for r in rows]
        finally:
            conn.close()

    def _get_job_sync(self, job_id: str) -> dict | None:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                return None
            job = self._row_to_detail(row)
            br = conn.execute(
                """SELECT * FROM build_results
                   WHERE job_id = ? ORDER BY idx""",
                (job_id,),
            ).fetchall()
            job["build_results"] = [
                {
                    "idx": r["idx"],
                    "target": r["target"],
                    "driver_path": r["driver_path"],
                    "success": bool(r["success"]),
                    "image_tag": r["image_tag"],
                    "has_log": bool(r["log_path"]) and Path(r["log_path"]).exists(),
                    "run_command": _col(r, "run_command"),
                    "container_name": _col(r, "container_name"),
                }
                for r in br
            ]
            return job
        finally:
            conn.close()

    async def stats(self) -> dict:
        return await asyncio.to_thread(self._stats_sync)

    def _stats_sync(self) -> dict:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            by_status = {
                r["status"]: r["n"]
                for r in conn.execute(
                    "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
                ).fetchall()
            }
            repos = [
                r["repo"]
                for r in conn.execute(
                    "SELECT DISTINCT repo FROM jobs ORDER BY repo"
                ).fetchall()
            ]
            return {"total": total, "by_status": by_status, "repos": repos}
        finally:
            conn.close()

    async def read_log(
        self, job_id: str, idx: int, offset: int = 0, max_bytes: int = 512 * 1024
    ) -> dict | None:
        """Read a build log from `offset`, for incremental tailing.

        Returns {content, offset, size, truncated} or None if there is no such
        log. `offset` beyond EOF yields empty content, which is what a poller
        sees while a build is between writes.
        """
        return await asyncio.to_thread(
            self._read_log_sync, job_id, idx, offset, max_bytes
        )

    def _read_log_sync(
        self, job_id: str, idx: int, offset: int, max_bytes: int
    ) -> dict | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT log_path FROM build_results WHERE job_id = ? AND idx = ?",
                (job_id, idx),
            ).fetchone()
        finally:
            conn.close()

        # Fall back to the conventional path: the row is written only after the
        # build finishes, but the file exists (and is being appended) during it.
        path = Path(row[0]) if row and row[0] else None
        if path is None or not path.exists():
            found = sorted(self.log_dir_for(job_id).glob(f"{idx}-*.log"))
            if not found:
                return None
            path = found[0]

        size = path.stat().st_size
        if offset < 0:
            offset = 0
        if offset >= size:
            return {"content": "", "offset": size, "size": size, "truncated": False}

        with open(path, "rb") as f:
            f.seek(offset)
            chunk = f.read(max_bytes)

        return {
            # errors="replace" because a chunk boundary can split a multi-byte
            # character; the next poll picks up the remainder cleanly.
            "content": chunk.decode("utf-8", errors="replace"),
            "offset": offset + len(chunk),
            "size": size,
            "truncated": offset + len(chunk) < size,
        }

    # ── Retention ─────────────────────────────────────────────────────────────

    async def reconcile_orphans(self) -> int:
        """Close out jobs left non-terminal by an unclean shutdown.

        A graceful stop marks in-flight jobs cancelled itself, but SIGKILL
        (`docker kill`, OOM) bypasses that and leaves rows stuck at `running`
        forever — the dashboard would show work that no longer exists. Nothing
        resumes across a restart, so on boot any non-terminal row is an orphan.
        """
        return await asyncio.to_thread(self._reconcile_orphans_sync)

    def _reconcile_orphans_sync(self) -> int:
        conn = self._connect()
        try:
            cursor = conn.execute(
                """UPDATE jobs
                   SET status = ?, error = ?, finished_at = ?
                   WHERE status IN (?, ?, ?)""",
                (
                    "cancelled",
                    "Interrupted by an unclean agent shutdown",
                    time.time(),
                    "queued", "running", "retrying",
                ),
            )
            conn.commit()
            n = cursor.rowcount or 0
        finally:
            conn.close()
        if n:
            logger.warning(f"Reconciled {n} orphaned job(s) from a previous run")
        return n

    async def prune(self, days: int) -> int:
        return await asyncio.to_thread(self._prune_sync, days)

    def _prune_sync(self, days: int) -> int:
        if days <= 0:
            return 0
        cutoff = time.time() - days * 86400
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            stale = [
                r["id"]
                for r in conn.execute(
                    "SELECT id FROM jobs WHERE created_at < ?", (cutoff,)
                ).fetchall()
            ]
            if not stale:
                return 0
            marks = ",".join("?" * len(stale))
            conn.execute(f"DELETE FROM build_results WHERE job_id IN ({marks})", stale)
            conn.execute(f"DELETE FROM jobs WHERE id IN ({marks})", stale)
            conn.commit()
        finally:
            conn.close()

        # Log directories are pruned with their job row, so the two never drift.
        for job_id in stale:
            shutil.rmtree(self.log_dir_for(job_id), ignore_errors=True)

        logger.info(f"Pruned {len(stale)} job(s) older than {days} days")
        return len(stale)

    # ── Row mapping ───────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_summary(row: sqlite3.Row) -> dict:
        keys = row.keys()
        stage_started = row["stage_started_at"] if "stage_started_at" in keys else None
        return {
            "id": row["id"],
            "repo": row["repo"],
            "pr_number": row["pr_number"],
            "head_sha": row["head_sha"],
            "head_ref": row["head_ref"],
            "requester": row["requester"],
            "source": row["source"],
            "status": row["status"],
            "stage": (row["stage"] if "stage" in keys else None) or "",
            "stage_detail": (row["stage_detail"] if "stage_detail" in keys else None) or "",
            "stage_elapsed": _elapsed(stage_started, row["finished_at"]),
            "attempt": row["attempt"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "elapsed": _elapsed(row["started_at"], row["finished_at"]),
            "build_results": [],
        }

    @classmethod
    def _row_to_detail(cls, row: sqlite3.Row) -> dict:
        detail = cls._row_to_summary(row)
        detail.update({
            "base_ref": row["base_ref"],
            "options": {
                "skip_build": bool(row["skip_build"]),
                "build_only": bool(row["build_only"]),
                "force_targets": _load_json(row["force_targets"], []),
            },
            "review_text": row["review_text"] or "",
            "findings": _load_json(row["findings"], []),
            "error": row["error"] or "",
            "attempt_errors": _load_json(row["attempt_errors"], []),
        })
        return detail


# ── Helpers ───────────────────────────────────────────────────────────────────


def _col(row: sqlite3.Row, name: str, default=""):
    """Read a column that may not exist on an un-migrated row."""
    return (row[name] if name in row.keys() else default) or default


def _load_json(raw, default):
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return default


def _elapsed(started_at, finished_at) -> float | None:
    """Seconds a job ran. Still-running jobs measure against now, so the
    dashboard shows a live-advancing duration without needing a separate field.
    """
    if started_at is None:
        return None
    return (finished_at or time.time()) - started_at


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  repo TEXT NOT NULL,
  pr_number INTEGER NOT NULL,
  head_sha TEXT,
  head_ref TEXT,
  base_ref TEXT,
  requester TEXT,
  source TEXT,
  status TEXT NOT NULL,
  stage TEXT,
  stage_detail TEXT,
  stage_started_at REAL,
  attempt INTEGER DEFAULT 0,
  skip_build INTEGER DEFAULT 0,
  build_only INTEGER DEFAULT 0,
  force_targets TEXT,
  review_text TEXT,
  findings TEXT,
  error TEXT,
  attempt_errors TEXT,
  created_at REAL NOT NULL,
  started_at REAL,
  finished_at REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_pr      ON jobs(repo, pr_number);

CREATE TABLE IF NOT EXISTS build_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL,
  idx INTEGER NOT NULL,
  target TEXT,
  driver_path TEXT,
  success INTEGER,
  image_tag TEXT,
  log_path TEXT,
  run_command TEXT,
  container_name TEXT,
  created_at REAL NOT NULL,
  UNIQUE(job_id, idx)
);
CREATE INDEX IF NOT EXISTS idx_br_job ON build_results(job_id);
"""

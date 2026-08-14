"""Git workspace management — bare clones + worktrees for parallel PR processing."""

import asyncio
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class GitWorkspaceManager:
    """Manages bare git clones and worktrees for parallel PR builds."""

    def __init__(self, data_dir: str, repos: dict[str, str]):
        """
        Args:
            data_dir: Base directory for all git data (e.g. /data/repos)
            repos: Mapping of repo full_name -> git clone URL
        """
        self._data_dir = Path(data_dir)
        self._repos = repos
        self._worktrees_dir = self._data_dir / "worktrees"
        self._fetch_locks: dict[str, asyncio.Lock] = {}

    def _bare_path(self, repo_full_name: str) -> Path:
        """Path to bare clone for a repo."""
        # "4paradigm/phanthymotus" -> "phanthymotus.git"
        name = repo_full_name.split("/")[-1]
        return self._data_dir / f"{name}.git"

    def _get_fetch_lock(self, repo_full_name: str) -> asyncio.Lock:
        if repo_full_name not in self._fetch_locks:
            self._fetch_locks[repo_full_name] = asyncio.Lock()
        return self._fetch_locks[repo_full_name]

    async def ensure_clones(self):
        """Ensure bare clones exist for all configured repos."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._worktrees_dir.mkdir(parents=True, exist_ok=True)

        for full_name, url in self._repos.items():
            bare_path = self._bare_path(full_name)
            if not bare_path.exists():
                logger.info(f"Cloning bare repo: {full_name} -> {bare_path}")
                await self._run_git(
                    ["git", "clone", "--bare", url, str(bare_path)],
                    cwd=str(self._data_dir),
                )
                # Configure fetch refspec to include PR refs
                await self._run_git(
                    ["git", "config", "--add", "remote.origin.fetch",
                     "+refs/pull/*/head:refs/pull/*/head"],
                    cwd=str(bare_path),
                )
            else:
                logger.info(f"Bare repo exists: {bare_path}")

    async def fetch(self, repo_full_name: str):
        """Fetch latest refs from origin (incremental, fast)."""
        bare_path = self._bare_path(repo_full_name)
        lock = self._get_fetch_lock(repo_full_name)
        async with lock:
            logger.info(f"Fetching {repo_full_name}...")
            await self._run_git(
                ["git", "fetch", "origin", "--prune"],
                cwd=str(bare_path),
            )

    async def create_worktree(
        self, repo_full_name: str, pr_number: int, head_sha: str
    ) -> Path:
        """Create an isolated worktree for a PR, merged onto main.

        Returns the worktree path.
        Raises RuntimeError on merge conflict.
        """
        bare_path = self._bare_path(repo_full_name)
        repo_short = repo_full_name.split("/")[-1]
        wt_name = f"{repo_short}-pr-{pr_number}-{head_sha[:7]}"
        wt_path = self._worktrees_dir / wt_name

        # Clean up if exists from a previous failed run
        if wt_path.exists():
            await self._remove_worktree_force(bare_path, wt_path)

        # Create worktree at origin/main
        await self._run_git(
            ["git", "worktree", "add", str(wt_path), "origin/main", "--detach"],
            cwd=str(bare_path),
        )

        # Merge PR head into it
        pr_ref = f"refs/pull/{pr_number}/head"
        try:
            await self._run_git(
                ["git", "merge", pr_ref, "--no-edit"],
                cwd=str(wt_path),
            )
        except RuntimeError as e:
            # Merge conflict — abort and clean up
            await self._run_git(
                ["git", "merge", "--abort"],
                cwd=str(wt_path),
                check=False,
            )
            await self._remove_worktree_force(bare_path, wt_path)
            raise RuntimeError(
                f"Merge conflict: PR #{pr_number} conflicts with main. "
                "Please resolve conflicts in the PR first."
            ) from e

        logger.info(f"Worktree created: {wt_path}")
        return wt_path

    async def remove_worktree(self, repo_full_name: str, worktree_path: Path):
        """Remove a worktree after use."""
        bare_path = self._bare_path(repo_full_name)
        await self._remove_worktree_force(bare_path, worktree_path)

    async def get_changed_files(self, worktree_path: Path) -> list[str]:
        """Get list of files changed relative to origin/main."""
        stdout = await self._run_git(
            ["git", "diff", "--name-only", "origin/main...HEAD"],
            cwd=str(worktree_path),
        )
        return [f for f in stdout.strip().splitlines() if f]

    async def get_diff(self, worktree_path: Path, max_lines: int = 3000) -> str:
        """Get full diff relative to origin/main, truncated."""
        stdout = await self._run_git(
            ["git", "diff", "origin/main...HEAD"],
            cwd=str(worktree_path),
        )
        lines = stdout.splitlines()
        if len(lines) > max_lines:
            return "\n".join(lines[:max_lines]) + f"\n\n... (truncated, {len(lines)} total lines)"
        return stdout

    async def get_diff_stat(self, worktree_path: Path) -> str:
        """Get diff stat (for large file detection)."""
        return await self._run_git(
            ["git", "diff", "--stat", "origin/main...HEAD"],
            cwd=str(worktree_path),
        )

    async def cleanup_stale_worktrees(self):
        """Remove any worktrees left over from crashed jobs."""
        if not self._worktrees_dir.exists():
            return
        for entry in self._worktrees_dir.iterdir():
            if entry.is_dir():
                logger.warning(f"Cleaning up stale worktree: {entry}")
                # Find the bare repo for this worktree
                for full_name in self._repos:
                    bare_path = self._bare_path(full_name)
                    if bare_path.exists():
                        await self._run_git(
                            ["git", "worktree", "remove", "--force", str(entry)],
                            cwd=str(bare_path),
                            check=False,
                        )
                        break
                # If still exists, force remove directory
                if entry.exists():
                    await self._run_cmd(["rm", "-rf", str(entry)])

    async def _remove_worktree_force(self, bare_path: Path, wt_path: Path):
        await self._run_git(
            ["git", "worktree", "remove", "--force", str(wt_path)],
            cwd=str(bare_path),
            check=False,
        )
        # Belt and suspenders
        if wt_path.exists():
            await self._run_cmd(["rm", "-rf", str(wt_path)])

    async def _run_git(
        self, cmd: list[str], cwd: str, check: bool = True
    ) -> str:
        return await self._run_cmd(cmd, cwd=cwd, check=check)

    async def _run_cmd(
        self, cmd: list[str], cwd: str | None = None, check: bool = True
    ) -> str:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        stdout_bytes, _ = await proc.communicate()
        stdout = stdout_bytes.decode(errors="replace")

        if check and proc.returncode != 0:
            raise RuntimeError(
                f"Command failed (rc={proc.returncode}): {' '.join(cmd)}\n{stdout}"
            )
        return stdout

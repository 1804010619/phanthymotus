"""Builder — shells out to the repos' existing build scripts.

Build output is streamed to a file on disk as it is produced, rather than
buffered in memory until the process exits. That is what makes the dashboard's
live log tailing possible, and it means a build that hangs or is killed still
leaves behind everything it printed up to that point — which is exactly what is
needed to diagnose it.
"""

import asyncio
import logging
import os
import re
import signal
from pathlib import Path

from .config import Config
from .models import BuildResult, BuildTarget

logger = logging.getLogger(__name__)

# Image tag markers printed by the build scripts. "Image :" appears near the
# start (before the build), the "Done." forms at the very end.
IMAGE_PATTERN = re.compile(r"Image\s*:\s*(\S+)")
DONE_PUSH_PATTERN = re.compile(r"Done\.\s*Image pushed:\s*(\S+)")
DONE_LOCAL_PATTERN = re.compile(r"Done\.\s*Image built locally:\s*(\S+)")

LOG_TAIL_LINES = 80
READ_CHUNK = 8192
# Bytes read from each end of a large log when scanning for the image tag.
# Covers "Image :" at the head and "Done. ..." at the tail without loading a
# multi-megabyte docker build log into memory.
SCAN_WINDOW = 256 * 1024
# Bytes read from the end when building the tail for the PR comment.
TAIL_WINDOW = 64 * 1024


def _build_env(config: Config) -> dict[str, str]:
    """Construct environment variables for the build scripts."""
    env = os.environ.copy()
    env["REGISTRY"] = config.registry
    env["REGISTRY_USER"] = config.registry_user
    env["REGISTRY_PASSWORD"] = config.registry_password
    env["IMAGE_NAMESPACE"] = config.image_namespace
    env["MIRROR"] = config.mirror
    env["IMAGE_NAMESPACE_DRIVERS"] = config.image_namespace_drivers
    env["DEBIAN_FRONTEND"] = "noninteractive"
    if config.resource_center_url:
        env["RESOURCE_CENTER_URL"] = config.resource_center_url
    if config.resource_center_api_key:
        env["RESOURCE_CENTER_API_KEY"] = config.resource_center_api_key
    return env


# ── Public build entry points ─────────────────────────────────────────────────


async def build_core(worktree: Path, config: Config, log_path: Path) -> BuildResult:
    """Build the agent-core image via deploy/build_core.sh."""
    return await _build_with_script(
        target=BuildTarget.CORE,
        driver_path=None,
        script=worktree / "deploy" / "build_core.sh",
        args=["--mirror", config.mirror],
        cwd=worktree,
        config=config,
        log_path=log_path,
    )


async def build_perception(
    worktree: Path, config: Config, log_path: Path
) -> BuildResult:
    """Build the perception image via deploy/build_perception.sh."""
    return await _build_with_script(
        target=BuildTarget.PERCEPTION,
        driver_path=None,
        script=worktree / "deploy" / "build_perception.sh",
        args=["--mirror", config.mirror],
        cwd=worktree,
        config=config,
        log_path=log_path,
    )


async def build_driver(
    worktree: Path, driver_path: str, config: Config, log_path: Path
) -> BuildResult:
    """Build one driver image via build.sh in CI mode."""
    env_overrides = {"IMAGE_NAMESPACE": config.image_namespace_drivers}
    return await _build_with_script(
        target=BuildTarget.DRIVER,
        driver_path=driver_path,
        script=worktree / "build.sh",
        args=["--mirror", config.mirror, driver_path],
        cwd=worktree,
        config=config,
        log_path=log_path,
        env_overrides=env_overrides,
    )


async def _build_with_script(
    target: BuildTarget,
    driver_path: str | None,
    script: Path,
    args: list[str],
    cwd: Path,
    config: Config,
    log_path: Path,
    env_overrides: dict[str, str] | None = None,
) -> BuildResult:
    label = driver_path or target.value

    if not script.exists():
        message = f"{script.name} not found in worktree"
        _write_log(log_path, message)
        return BuildResult(
            target=target,
            driver_path=driver_path,
            success=False,
            image_tag="",
            log_tail=message,
            log_path=str(log_path),
        )

    env = _build_env(config)
    if env_overrides:
        env.update(env_overrides)

    success = await _run_build(
        ["bash", str(script), *args],
        cwd=str(cwd),
        env=env,
        timeout=config.build_timeout_seconds,
        log_path=log_path,
        label=label,
    )

    return BuildResult(
        target=target,
        driver_path=driver_path,
        success=success,
        image_tag=_extract_image_tag(log_path),
        log_tail=_read_tail(log_path),
        log_path=str(log_path),
    )


# ── Process execution ─────────────────────────────────────────────────────────


async def _run_build(
    cmd: list[str],
    cwd: str,
    env: dict[str, str],
    timeout: int,
    log_path: Path,
    label: str,
) -> bool:
    """Run a build, streaming its output to `log_path`. Returns success.

    The subprocess is killed on both timeout and cancellation. Cancellation
    matters because the whole-job timeout cancels this coroutine from the
    outside — without the explicit kill, `docker build` would keep running
    orphaned, holding the build cache and CPU for the retry to contend with.
    """
    logger.info(f"Building {label}: {' '.join(cmd[:3])}... (cwd={cwd})")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # start_new_session puts the child in its own process group so the whole
    # build tree (bash -> docker -> buildx) dies with it, not just bash.
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )

    # buffering=0 so a reader tailing this file sees output as it is produced
    # rather than one block per flush.
    log_file = open(log_path, "wb", buffering=0)

    async def pump_and_wait() -> int:
        """Copy the child's output to disk until EOF, then reap it.

        Fixed-size chunks rather than readline(): StreamReader.readline()
        raises ValueError once a line exceeds the stream limit (64 KiB by
        default), and docker build output can contain very long single lines.

        The reap is inside the timed region deliberately — a process that
        closes stdout without exiting would otherwise hang past the timeout.
        """
        while True:
            chunk = await proc.stdout.read(READ_CHUNK)
            if not chunk:
                break
            log_file.write(chunk)
        return await proc.wait()

    task = asyncio.create_task(pump_and_wait())

    try:
        # Shielded so a timeout or outer cancellation does not kill the task
        # mid-write; it is drained explicitly below so the partial log survives.
        returncode = await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    except asyncio.TimeoutError:
        await _terminate(proc)
        await _drain(task)
        log_file.write(f"\n[agent] Build timed out after {timeout}s\n".encode())
        logger.error(f"Build timed out after {timeout}s: {label}")
        return False
    except asyncio.CancelledError:
        await _terminate(proc)
        await _drain(task)
        log_file.write(b"\n[agent] Build cancelled (agent stopping)\n")
        logger.warning(f"Build cancelled, subprocess killed: {label}")
        raise
    finally:
        # Closed after the task is drained, so the partial log is flushed even
        # when the build was killed.
        log_file.close()

    success = returncode == 0
    if success:
        logger.info(f"Build succeeded: {label}")
    else:
        logger.error(f"Build failed (rc={returncode}): {label}")
    return success


async def _drain(task: asyncio.Task):
    """Let the pump finish writing whatever the child already emitted.

    Shielded and bounded: we are often already being cancelled here, and an
    unshielded await would abort immediately and lose the tail of the log.
    """
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=5)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        task.cancel()
    except Exception:
        # The pump itself failed; the log is whatever made it to disk.
        pass


async def _terminate(proc: asyncio.subprocess.Process):
    """Kill a build subprocess and its process group, then reap it."""
    if proc.returncode is not None:
        return

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except ProcessLookupError:
            return

    try:
        await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=10)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass


# ── Log file helpers ──────────────────────────────────────────────────────────


def _write_log(log_path: Path, text: str):
    """Write a one-off message as the whole log (used for pre-flight errors)."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(text + "\n")
    except OSError as e:
        logger.warning(f"Failed to write log {log_path}: {e}")


def _read_tail(log_path: Path, lines: int = LOG_TAIL_LINES) -> str:
    """Last N lines of a log, for the PR comment."""
    try:
        size = log_path.stat().st_size
        with open(log_path, "rb") as f:
            if size > TAIL_WINDOW:
                f.seek(size - TAIL_WINDOW)
                # Drop the first (probably partial) line after seeking.
                f.readline()
            data = f.read()
    except OSError:
        return ""
    text = data.decode("utf-8", errors="replace")
    parts = text.splitlines()
    return "\n".join(parts[-lines:])


def _extract_image_tag(log_path: Path) -> str:
    """Pull the built image reference out of a build log.

    Scans the head and tail rather than the whole file: the markers only ever
    appear at the start ("Image : ...") or the end ("Done. Image pushed: ...").
    """
    try:
        size = log_path.stat().st_size
        with open(log_path, "rb") as f:
            if size <= 2 * SCAN_WINDOW:
                head = f.read()
                tail = b""
            else:
                head = f.read(SCAN_WINDOW)
                f.seek(size - SCAN_WINDOW)
                tail = f.read(SCAN_WINDOW)
    except OSError:
        return ""

    head_text = head.decode("utf-8", errors="replace")
    tail_text = tail.decode("utf-8", errors="replace")

    # Most specific first: the "Done." lines confirm the build finished.
    for pattern in (DONE_PUSH_PATTERN, DONE_LOCAL_PATTERN):
        for text in (tail_text, head_text):
            m = pattern.search(text)
            if m:
                return m.group(1)

    m = IMAGE_PATTERN.search(head_text) or IMAGE_PATTERN.search(tail_text)
    return m.group(1) if m else ""


def log_filename(idx: int, target: BuildTarget, driver_path: str | None) -> str:
    """Log filename for one build within a job: `{idx}-{safe-label}.log`."""
    label = driver_path or target.value
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", label)
    return f"{idx}-{safe}.log"

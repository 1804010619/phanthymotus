"""Builder — shells out to existing build scripts."""

import asyncio
import logging
import os
import re
import signal
from pathlib import Path

from .config import Config
from .models import BuildResult, BuildTarget

logger = logging.getLogger(__name__)

# Regex to extract image tag from build script output
IMAGE_PATTERN = re.compile(r"Image\s*:\s*(\S+)")
DONE_PUSH_PATTERN = re.compile(r"Done\.\s*Image pushed:\s*(\S+)")
DONE_LOCAL_PATTERN = re.compile(r"Done\.\s*Image built locally:\s*(\S+)")

LOG_TAIL_LINES = 80


def _build_env(config: Config) -> dict[str, str]:
    """Construct environment variables for build scripts."""
    env = os.environ.copy()
    env["REGISTRY"] = config.registry
    env["REGISTRY_USER"] = config.registry_user
    env["REGISTRY_PASSWORD"] = config.registry_password
    env["IMAGE_NAMESPACE"] = config.image_namespace
    env["MIRROR"] = config.mirror
    # For driver builds
    env["IMAGE_NAMESPACE_DRIVERS"] = config.image_namespace_drivers
    # Disable interactive prompts in build scripts
    env["DEBIAN_FRONTEND"] = "noninteractive"
    # Resource Center (optional)
    if config.resource_center_url:
        env["RESOURCE_CENTER_URL"] = config.resource_center_url
    if config.resource_center_api_key:
        env["RESOURCE_CENTER_API_KEY"] = config.resource_center_api_key
    return env


def _extract_image_tag(output: str) -> str:
    """Extract full image reference from build output."""
    # Try "Done. Image pushed: xxx" first (most specific)
    for m in DONE_PUSH_PATTERN.finditer(output):
        return m.group(1)
    # Try "Done. Image built locally: xxx"
    for m in DONE_LOCAL_PATTERN.finditer(output):
        return m.group(1)
    # Fall back to "Image : xxx"
    for m in IMAGE_PATTERN.finditer(output):
        return m.group(1)
    return ""


def _tail(text: str, n: int = LOG_TAIL_LINES) -> str:
    """Get last N lines of text."""
    lines = text.splitlines()
    if len(lines) <= n:
        return text
    return "\n".join(lines[-n:])


async def build_core(worktree: Path, config: Config) -> BuildResult:
    """Build agent-core image using deploy/build_core.sh."""
    script = worktree / "deploy" / "build_core.sh"
    if not script.exists():
        return BuildResult(
            target=BuildTarget.CORE,
            driver_path=None,
            success=False,
            image_tag="",
            log_tail="build_core.sh not found in worktree",
        )

    env = _build_env(config)
    output = await _run_build(
        ["bash", str(script), "--mirror", config.mirror],
        cwd=str(worktree),
        env=env,
        timeout=config.build_timeout_seconds,
    )

    image_tag = _extract_image_tag(output.stdout)
    return BuildResult(
        target=BuildTarget.CORE,
        driver_path=None,
        success=output.success,
        image_tag=image_tag,
        log_tail=_tail(output.stdout),
    )


async def build_perception(worktree: Path, config: Config) -> BuildResult:
    """Build perception image using deploy/build_perception.sh."""
    script = worktree / "deploy" / "build_perception.sh"
    if not script.exists():
        return BuildResult(
            target=BuildTarget.PERCEPTION,
            driver_path=None,
            success=False,
            image_tag="",
            log_tail="build_perception.sh not found in worktree",
        )

    env = _build_env(config)
    output = await _run_build(
        ["bash", str(script), "--mirror", config.mirror],
        cwd=str(worktree),
        env=env,
        timeout=config.build_timeout_seconds,
    )

    image_tag = _extract_image_tag(output.stdout)
    return BuildResult(
        target=BuildTarget.PERCEPTION,
        driver_path=None,
        success=output.success,
        image_tag=image_tag,
        log_tail=_tail(output.stdout),
    )


async def build_driver(
    worktree: Path, driver_path: str, config: Config
) -> BuildResult:
    """Build a driver image using build.sh in CI mode.

    Args:
        worktree: Path to the phanthymotus-driver worktree
        driver_path: e.g. "unitree/g1"
        config: App config
    """
    script = worktree / "build.sh"
    if not script.exists():
        return BuildResult(
            target=BuildTarget.DRIVER,
            driver_path=driver_path,
            success=False,
            image_tag="",
            log_tail="build.sh not found in worktree",
        )

    env = _build_env(config)
    # Override IMAGE_NAMESPACE for drivers
    env["IMAGE_NAMESPACE"] = config.image_namespace_drivers

    output = await _run_build(
        ["bash", str(script), "--mirror", config.mirror, driver_path],
        cwd=str(worktree),
        env=env,
        timeout=config.build_timeout_seconds,
    )

    image_tag = _extract_image_tag(output.stdout)
    return BuildResult(
        target=BuildTarget.DRIVER,
        driver_path=driver_path,
        success=output.success,
        image_tag=image_tag,
        log_tail=_tail(output.stdout),
    )


class _BuildOutput:
    def __init__(self, success: bool, stdout: str):
        self.success = success
        self.stdout = stdout


async def _run_build(
    cmd: list[str],
    cwd: str,
    env: dict[str, str],
    timeout: int,
) -> _BuildOutput:
    """Run a build command with timeout, capturing output.

    The subprocess is killed on both timeout and cancellation. Cancellation
    matters because the whole-job timeout cancels this coroutine from the
    outside — without the explicit kill, `docker build` would keep running
    orphaned, holding the build cache and CPU for the retry to contend with.
    """
    logger.info(f"Running build: {' '.join(cmd[:3])}... (cwd={cwd})")

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

    try:
        stdout_bytes, _ = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        await _terminate(proc)
        logger.error(f"Build timed out after {timeout}s: {' '.join(cmd[:3])}")
        return _BuildOutput(
            success=False,
            stdout=f"Build timed out after {timeout}s",
        )
    except asyncio.CancelledError:
        await _terminate(proc)
        logger.warning(f"Build cancelled, subprocess killed: {' '.join(cmd[:3])}")
        raise

    stdout = stdout_bytes.decode(errors="replace")
    success = proc.returncode == 0

    if success:
        logger.info(f"Build succeeded: {' '.join(cmd[:3])}")
    else:
        logger.error(f"Build failed (rc={proc.returncode}): {' '.join(cmd[:3])}")

    return _BuildOutput(success=success, stdout=stdout)


async def _terminate(proc: asyncio.subprocess.Process):
    """Kill a build subprocess and its process group, then reap it."""
    if proc.returncode is not None:
        return

    # Kill the whole process group so docker/buildx children go too.
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        # Group already gone, or we cannot signal it — fall back to the child.
        try:
            proc.kill()
        except ProcessLookupError:
            return

    # Reap so we do not leave a zombie. Shielded because we are often already
    # being cancelled, and an unshielded await would abort immediately.
    try:
        await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=10)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass

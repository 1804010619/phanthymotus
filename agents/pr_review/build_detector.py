"""Build target detection — analyze changed files to determine what to build."""

import logging
from pathlib import Path

from .models import BuildTarget

logger = logging.getLogger(__name__)

# Known driver providers in phanthymotus-driver
KNOWN_PROVIDERS = {"unitree", "dji", "noetix", "x-humanoid", "engineai", "pnpbotics"}

# Files/dirs that never trigger a build
IGNORED_PATTERNS = {
    "README.md", "README_zh.md", "README_dev.md",
    "CONTRIBUTING.md", "LICENSE", "CODEOWNERS",
    ".env.example", ".gitignore",
}


def detect_targets(
    repo_full_name: str, changed_files: list[str]
) -> tuple[list[BuildTarget], list[str]]:
    """Analyze changed files to determine build targets.

    Args:
        repo_full_name: e.g. "4paradigm/phanthymotus"
        changed_files: list of relative file paths from git diff

    Returns:
        (targets, driver_paths) where driver_paths is populated for DRIVER targets.
    """
    repo_name = repo_full_name.split("/")[-1]

    if repo_name == "phanthymotus":
        return _detect_motus_targets(changed_files)
    elif repo_name == "phanthymotus-driver":
        return _detect_driver_targets(changed_files)
    else:
        logger.warning(f"Unknown repo: {repo_full_name}")
        return [], []


def _detect_motus_targets(changed_files: list[str]) -> tuple[list[BuildTarget], list[str]]:
    """Detect build targets for phanthymotus repo."""
    targets = set()

    for f in changed_files:
        if _is_ignored(f):
            continue
        parts = Path(f).parts
        if not parts:
            continue

        if parts[0] == "agent-core":
            targets.add(BuildTarget.CORE)
        elif parts[0] == "perception":
            targets.add(BuildTarget.PERCEPTION)
        # deploy/ changes don't trigger build unless it's the Dockerfile
        # that's already covered by agent-core/ or perception/ above

    return list(targets), []


def _detect_driver_targets(changed_files: list[str]) -> tuple[list[BuildTarget], list[str]]:
    """Detect build targets for phanthymotus-driver repo."""
    driver_paths = set()

    for f in changed_files:
        if _is_ignored(f):
            continue
        parts = Path(f).parts
        if len(parts) < 2:
            continue

        provider = parts[0]
        model = parts[1]

        # Skip if not a known provider directory
        if provider not in KNOWN_PROVIDERS:
            continue

        # Skip "base" directories (like dji/base — not a standalone driver)
        if model == "base":
            continue

        driver_path = f"{provider}/{model}"
        driver_paths.add(driver_path)

    if driver_paths:
        return [BuildTarget.DRIVER], sorted(driver_paths)
    return [], []


def _is_ignored(filepath: str) -> bool:
    """Check if a file should be ignored for build detection."""
    name = Path(filepath).name
    return name in IGNORED_PATTERNS

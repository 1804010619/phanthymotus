"""Configuration loaded from environment variables."""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    # GitHub
    github_token: str = ""
    github_webhook_secret: str = ""

    # Repos: mapping of full_name -> git clone URL.
    # HTTPS is used because both repos are public — no SSH key needed on the host.
    repos: dict[str, str] = field(default_factory=lambda: {
        "4paradigm/phanthymotus": "https://github.com/4paradigm/phanthymotus.git",
        "4paradigm/phanthymotus-driver": "https://github.com/4paradigm/phanthymotus-driver.git",
    })

    # Trigger mode — polling needs only outbound network, webhook needs inbound.
    poll_enabled: bool = True
    poll_interval_seconds: int = 30
    poll_initial_lookback_minutes: int = 10
    webhook_enabled: bool = True

    # Registry
    registry: str = ""
    registry_user: str = ""
    registry_password: str = ""
    image_namespace: str = "phanthy-motus"
    image_namespace_drivers: str = "phanthy-motus/drivers"
    mirror: str = "tencent"
    push_enabled: bool = True

    # LLM Review (OpenAI-compatible)
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"
    max_diff_lines: int = 3000

    # Worker
    max_concurrent_jobs: int = 2
    # Timeout for a single docker build invocation.
    build_timeout_seconds: int = 1800
    # Whole-job timeout. A job exceeding this is treated as lost and retried.
    job_timeout_seconds: int = 3600
    # Total attempts per job including the first (3 = two retries).
    max_attempts: int = 3
    retry_backoff_seconds: int = 60

    # How long job history and build logs are retained. Pruned at startup.
    job_history_days: int = 30

    # Paths
    data_dir: str = "/data/repos"

    # Server
    host: str = "0.0.0.0"
    port: int = 25000

    # Resource Center (optional)
    resource_center_url: str = ""
    resource_center_api_key: str = ""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _load_repos() -> dict[str, str] | None:
    """Optionally override the repo list via GITHUB_REPOS.

    Format: comma-separated full names, e.g.
        GITHUB_REPOS=4paradigm/phanthymotus,4paradigm/phanthymotus-driver
    """
    raw = os.environ.get("GITHUB_REPOS", "").strip()
    if not raw:
        return None
    repos = {}
    for item in raw.split(","):
        full_name = item.strip()
        if not full_name or "/" not in full_name:
            continue
        repos[full_name] = f"https://github.com/{full_name}.git"
    return repos or None


def load_config() -> Config:
    overrides = {}
    repos = _load_repos()
    if repos is not None:
        overrides["repos"] = repos

    return Config(
        github_token=os.environ.get("GITHUB_TOKEN", ""),
        github_webhook_secret=os.environ.get("GITHUB_WEBHOOK_SECRET", ""),
        poll_enabled=_env_bool("POLL_ENABLED", True),
        poll_interval_seconds=_env_int("POLL_INTERVAL_SECONDS", 30),
        poll_initial_lookback_minutes=_env_int("POLL_INITIAL_LOOKBACK_MINUTES", 10),
        webhook_enabled=_env_bool("WEBHOOK_ENABLED", True),
        registry=os.environ.get("REGISTRY", ""),
        registry_user=os.environ.get("REGISTRY_USER", ""),
        registry_password=os.environ.get("REGISTRY_PASSWORD", ""),
        image_namespace=os.environ.get("IMAGE_NAMESPACE", "phanthy-motus"),
        image_namespace_drivers=os.environ.get(
            "IMAGE_NAMESPACE_DRIVERS", "phanthy-motus/drivers"
        ),
        mirror=os.environ.get("MIRROR", "tencent"),
        push_enabled=_env_bool("PUSH_ENABLED", True),
        llm_base_url=os.environ.get("LLM_BASE_URL", ""),
        llm_api_key=os.environ.get("LLM_API_KEY", ""),
        llm_model=os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514"),
        max_diff_lines=_env_int("MAX_DIFF_LINES", 3000),
        max_concurrent_jobs=_env_int("MAX_CONCURRENT_JOBS", 2),
        build_timeout_seconds=_env_int("BUILD_TIMEOUT_SECONDS", 1800),
        job_timeout_seconds=_env_int("JOB_TIMEOUT_SECONDS", 3600),
        max_attempts=_env_int("MAX_ATTEMPTS", 3),
        retry_backoff_seconds=_env_int("RETRY_BACKOFF_SECONDS", 60),
        job_history_days=_env_int("JOB_HISTORY_DAYS", 30),
        data_dir=os.environ.get("DATA_DIR", "/data/repos"),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=_env_int("PORT", 25000),
        resource_center_url=os.environ.get("RESOURCE_CENTER_URL", ""),
        resource_center_api_key=os.environ.get("RESOURCE_CENTER_API_KEY", ""),
        **overrides,
    )

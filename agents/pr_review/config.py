"""Configuration loaded from environment variables."""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    # GitHub
    github_webhook_secret: str = ""
    github_token: str = ""

    # Repos: mapping of full_name -> git SSH URL
    repos: dict[str, str] = field(default_factory=lambda: {
        "4paradigm/phanthymotus": "git@github.com:4paradigm/phanthymotus.git",
        "4paradigm/phanthymotus-driver": "git@github.com:4paradigm/phanthymotus-driver.git",
    })

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
    build_timeout_seconds: int = 1800

    # Paths
    data_dir: str = "/data/repos"

    # Server
    host: str = "0.0.0.0"
    port: int = 15690

    # Resource Center (optional)
    resource_center_url: str = ""
    resource_center_api_key: str = ""


def load_config() -> Config:
    return Config(
        github_webhook_secret=os.environ.get("GITHUB_WEBHOOK_SECRET", ""),
        github_token=os.environ.get("GITHUB_TOKEN", ""),
        registry=os.environ.get("REGISTRY", ""),
        registry_user=os.environ.get("REGISTRY_USER", ""),
        registry_password=os.environ.get("REGISTRY_PASSWORD", ""),
        image_namespace=os.environ.get("IMAGE_NAMESPACE", "phanthy-motus"),
        image_namespace_drivers=os.environ.get("IMAGE_NAMESPACE_DRIVERS", "phanthy-motus/drivers"),
        mirror=os.environ.get("MIRROR", "tencent"),
        push_enabled=os.environ.get("PUSH_ENABLED", "true").lower() == "true",
        llm_base_url=os.environ.get("LLM_BASE_URL", ""),
        llm_api_key=os.environ.get("LLM_API_KEY", ""),
        llm_model=os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514"),
        max_diff_lines=int(os.environ.get("MAX_DIFF_LINES", "3000")),
        max_concurrent_jobs=int(os.environ.get("MAX_CONCURRENT_JOBS", "2")),
        build_timeout_seconds=int(os.environ.get("BUILD_TIMEOUT_SECONDS", "1800")),
        data_dir=os.environ.get("DATA_DIR", "/data/repos"),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "15690")),
        resource_center_url=os.environ.get("RESOURCE_CENTER_URL", ""),
        resource_center_api_key=os.environ.get("RESOURCE_CENTER_API_KEY", ""),
    )

"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from .config import load_config
from .git_workspace import GitWorkspaceManager
from .github_client import GitHubClient
from .job_queue import JobQueue
from .poller import Poller
from .router_status import router as status_router
from .router_webhook import router as webhook_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    app.state.config = config

    if not config.github_token:
        logger.warning("GITHUB_TOKEN is not set — GitHub API calls will fail")

    app.state.github_client = GitHubClient(config.github_token)

    # Persistent bare clones — cloned once, then fetched incrementally.
    workspace_mgr = GitWorkspaceManager(config.data_dir, config.repos)
    await workspace_mgr.ensure_clones()
    await workspace_mgr.cleanup_stale_worktrees()
    app.state.workspace_mgr = workspace_mgr

    job_queue = JobQueue(
        max_workers=config.max_concurrent_jobs,
        config=config,
        github_client=app.state.github_client,
        workspace_mgr=workspace_mgr,
    )
    await job_queue.start()
    app.state.job_queue = job_queue

    # Polling is the default trigger: outbound-only, so the agent needs no
    # public IP, no open port, and no webhook registration.
    app.state.poller = None
    if config.poll_enabled:
        poller = Poller(config, app.state.github_client, job_queue)
        await poller.start()
        app.state.poller = poller

    logger.info(
        f"PR Review Agent listening on {config.host}:{config.port} "
        f"(workers={config.max_concurrent_jobs}, "
        f"poll={'on' if config.poll_enabled else 'off'}, "
        f"webhook={'on' if config.webhook_enabled else 'off'}, "
        f"job_timeout={config.job_timeout_seconds}s, "
        f"max_attempts={config.max_attempts})"
    )

    yield

    if app.state.poller is not None:
        await app.state.poller.stop()
    await job_queue.stop()
    await app.state.github_client.close()


app = FastAPI(title="PR Review Agent", lifespan=lifespan)
app.include_router(webhook_router)
app.include_router(status_router)


def main():
    config = load_config()
    uvicorn.run(
        "agents.pr_review.server:app",
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

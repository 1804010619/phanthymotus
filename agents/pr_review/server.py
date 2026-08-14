"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from .config import load_config
from .github_client import GitHubClient
from .git_workspace import GitWorkspaceManager
from .job_queue import JobQueue
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

    # Initialize GitHub client
    app.state.github_client = GitHubClient(config.github_token)

    # Initialize git workspace manager
    workspace_mgr = GitWorkspaceManager(config.data_dir, config.repos)
    await workspace_mgr.ensure_clones()
    app.state.workspace_mgr = workspace_mgr

    # Initialize and start job queue
    job_queue = JobQueue(
        max_workers=config.max_concurrent_jobs,
        config=config,
        github_client=app.state.github_client,
        workspace_mgr=workspace_mgr,
    )
    await job_queue.start()
    app.state.job_queue = job_queue

    logger.info(
        f"PR Review Agent started on :{config.port} "
        f"(workers={config.max_concurrent_jobs})"
    )
    yield

    # Shutdown
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

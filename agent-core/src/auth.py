"""
auth.py — Access token authentication for Dashboard and API.

Token source: /opt/phanthy-motus/.env file (volume-mounted from host).
Read on every request so changes take effect immediately without restart.
"""

import os
import pathlib

from fastapi import Request, WebSocket
from fastapi.responses import JSONResponse


_ENV_PATH = pathlib.Path('/opt/phanthy-motus/.env')
# Fallback for local dev
_ENV_PATH_DEV = pathlib.Path('.env')


def _read_token() -> str:
    """Read ACCESS_TOKEN from .env file (host volume mount)."""
    env_file = _ENV_PATH if _ENV_PATH.exists() else _ENV_PATH_DEV
    if not env_file.exists():
        return ''
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith('ACCESS_TOKEN=') and not line.startswith('#'):
            return line.split('=', 1)[1].strip()
    return ''


def init():
    """Print auth status on startup."""
    token = _read_token()
    if token:
        print(f'[auth] Token authentication enabled (from {_ENV_PATH})')
    else:
        print(f'[auth] No ACCESS_TOKEN in .env — authentication disabled')


def get_token() -> str:
    return _read_token()


def is_enabled() -> bool:
    return bool(_read_token())


def verify(token: str) -> bool:
    current = _read_token()
    if not current:
        return True  # Auth disabled
    return bool(token) and token == current


async def auth_middleware(request: Request, call_next):
    """FastAPI middleware: enforce token auth on /api/* and /ws/* paths."""
    if not is_enabled():
        return await call_next(request)

    path = request.url.path

    # Static files and HTML — no auth needed
    if not path.startswith('/api/') and not path.startswith('/ws/'):
        return await call_next(request)

    # Exempt paths
    if path == '/api/auth/verify':
        return await call_next(request)
    if path.startswith('/api/channel/webhook/'):
        return await call_next(request)
    # MCP registration from driver containers
    if path == '/api/mcp' and request.method == 'POST':
        return await call_next(request)
    # /ws/mic stays open (internal browser mic)
    if path == '/ws/mic':
        return await call_next(request)

    # Check token
    token = _extract_token(request)
    if not verify(token):
        return JSONResponse(status_code=401, content={'detail': 'Unauthorized'})

    return await call_next(request)


def check_ws_token(websocket: WebSocket) -> bool:
    """Check token in WebSocket query params."""
    if not is_enabled():
        return True
    token = websocket.query_params.get('token', '')
    return verify(token)


def _extract_token(request: Request) -> str:
    """Extract token from Authorization header or query param."""
    auth = request.headers.get('authorization', '')
    if auth.startswith('Bearer '):
        return auth[7:]
    return request.query_params.get('token', '')


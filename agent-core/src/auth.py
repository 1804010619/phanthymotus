"""
auth.py — Access token authentication for Dashboard and API.

Token source (priority):
1. Environment variable ACCESS_TOKEN
2. SQLite config table 'access_token' key (auto-generated on first startup)

Protected: all /api/* (except /api/auth/verify, /api/channel/webhook/*), /ws/motus, /ws/bus/*
Open: static files, /ws/mic
"""

import os

from fastapi import Request, WebSocket
from fastapi.responses import JSONResponse


_token: str = ''
_auth_enabled: bool = False


def init():
    """Initialize access token (call during startup)."""
    global _token, _auth_enabled
    # Only source: ACCESS_TOKEN environment variable
    token = os.environ.get('ACCESS_TOKEN', '').strip()
    if token:
        _token = token
        _auth_enabled = True
        print(f'[auth] Token authentication enabled (from environment)')
        return

    # No token configured — auth disabled
    _auth_enabled = False
    print(f'[auth] No ACCESS_TOKEN configured — authentication disabled')
    print(f'[auth] Set ACCESS_TOKEN env var to enable authentication')


def get_token() -> str:
    return _token


def is_enabled() -> bool:
    return _auth_enabled


def verify(token: str) -> bool:
    if not _auth_enabled:
        return True
    return bool(token) and token == _token


async def auth_middleware(request: Request, call_next):
    """FastAPI middleware: enforce token auth on /api/* and /ws/* paths."""
    if not _auth_enabled:
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
    # MCP registration from driver containers (localhost)
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
    if not _auth_enabled:
        return True
    token = websocket.query_params.get('token', '')
    return verify(token)


def _extract_token(request: Request) -> str:
    """Extract token from Authorization header or query param."""
    # Header: Authorization: Bearer xxx
    auth = request.headers.get('authorization', '')
    if auth.startswith('Bearer '):
        return auth[7:]
    # Query param: ?token=xxx
    return request.query_params.get('token', '')

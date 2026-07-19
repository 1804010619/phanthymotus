"""
auth.py — Access token authentication for Dashboard and API.

Token source (priority):
1. Environment variable ACCESS_TOKEN
2. SQLite config table 'access_token' key (auto-generated on first startup)

Protected: all /api/* (except /api/auth/verify, /api/channel/webhook/*), /ws/motus, /ws/bus/*
Open: static files, /ws/mic
"""

import os
import secrets

import config
from fastapi import Request, WebSocket
from fastapi.responses import JSONResponse


_token: str = ''


def init():
    """Initialize access token (call during startup)."""
    global _token
    # Priority 1: env var
    token = os.environ.get('ACCESS_TOKEN', '').strip()
    if token:
        _token = token
        print(f'[auth] Using ACCESS_TOKEN from environment')
        return

    # Priority 2: DB
    token = config.main.get('access_token', '')
    if token:
        _token = token
        print(f'[auth] Access token: {_token}')
        return

    # Auto-generate
    _token = secrets.token_urlsafe(24)
    config.main['access_token'] = _token
    print(f'[auth] Generated access token: {_token}')
    print(f'[auth] Use this token to access the Dashboard and API')


def get_token() -> str:
    return _token


def verify(token: str) -> bool:
    return bool(token) and token == _token


async def auth_middleware(request: Request, call_next):
    """FastAPI middleware: enforce token auth on /api/* and /ws/* paths."""
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

"""
canvas.py — Canvas layout persistence + per-tool config storage.

Stores the orchestration canvas layout (card positions) and per-tool
configuration in the SQLite config table.

Includes editor lock: only one session can edit at a time.
"""

import asyncio
import json
import time
import fastapi
from pydantic import BaseModel
from typing import Any, Optional

import config

router = fastapi.APIRouter(prefix='/canvas', tags=['canvas'])

_TOOL_CONFIG_PREFIX = 'tool_config:'

# ── Editor Lock State (in-memory, resets on restart) ─────────────────────────

_editor_session: Optional[str] = None   # session_id of current editor
_editor_last_seen: float = 0.0          # monotonic timestamp of last activity
_EDITOR_TIMEOUT = 60.0                  # seconds before auto-release


def _check_editor_expired():
    """Release editor if inactive for too long."""
    global _editor_session, _editor_last_seen
    if _editor_session and (time.monotonic() - _editor_last_seen) > _EDITOR_TIMEOUT:
        _editor_session = None
        _editor_last_seen = 0.0


def current_editor() -> Optional[str]:
    """Session id currently holding the edit lock, or None.

    Used by api/solutions.py: applying a solution rewrites canvas_layout behind
    the lock's back, so it has to refuse while someone is editing — their next
    autosave would otherwise clobber the freshly loaded layout.
    """
    _check_editor_expired()
    return _editor_session


def apply_tool_config(mcp_id: str, tool_name: str, body: Any,
                      instance_id: str = '') -> None:
    """Push a saved config down to the MCP plugin (fire-and-forget).

    Shared by the tool-config endpoints below and by api/solutions.py when a
    solution is applied, so both paths send the exact same `action: config`
    call shape.
    """
    from api.mcp_manage import mcp_call_tool, MCPCallRequest

    args = dict(body) if isinstance(body, dict) else {}
    if instance_id:
        args['instance_id'] = instance_id

    async def _apply():
        try:
            req = MCPCallRequest(tool=tool_name, arguments={'action': 'config', **args})
            await mcp_call_tool(mcp_id, req)
        except Exception:
            pass

    try:
        asyncio.create_task(_apply())
    except RuntimeError:
        # No running loop (called from a sync context outside a request). The
        # config row is already persisted, and mcp_manage._restore_saved_configs
        # re-sends it when the device next comes online — so skipping the live
        # push here degrades timing, not correctness.
        pass


def tool_config_key(mcp_id: str, tool_name: str, instance_id: str = '') -> str:
    """ConfigDB key for a tool config row."""
    key = f'{_TOOL_CONFIG_PREFIX}{mcp_id}:{tool_name}'
    return f'{key}:{instance_id}' if instance_id else key



class CanvasLayout(BaseModel):
    cards:           list  = []
    connections:     list  = []
    execConnections: list  = []
    transform:       dict  = {}
    session_id:      Optional[str] = None


# ── Editor Lock Endpoints ────────────────────────────────────────────────────

@router.post('/claim-edit')
async def claim_edit(body: dict = fastapi.Body(...)):
    """Request edit permission. Returns 423 if someone else is editing."""
    global _editor_session, _editor_last_seen
    _check_editor_expired()

    session_id = body.get('session_id', '')
    if not session_id:
        return fastapi.responses.JSONResponse(
            status_code=400, content={'code': 400, 'message': 'session_id required'})

    if _editor_session and _editor_session != session_id:
        return fastapi.responses.JSONResponse(
            status_code=423, content={'code': 423, 'message': 'Canvas is locked by another editor',
                                      'editor': _editor_session})

    _editor_session = session_id
    _editor_last_seen = time.monotonic()
    return {'code': 200, 'editor': session_id}


@router.post('/release-edit')
async def release_edit(body: dict = fastapi.Body(...)):
    """Release edit permission."""
    global _editor_session, _editor_last_seen
    session_id = body.get('session_id', '')
    if _editor_session == session_id:
        _editor_session = None
        _editor_last_seen = 0.0
    return {'code': 200}


@router.get('/edit-status')
async def edit_status():
    """Check who is currently editing."""
    _check_editor_expired()
    return {'code': 200, 'editor': _editor_session}


# ── Layout Endpoints ─────────────────────────────────────────────────────────

@router.get('/layout')
async def get_layout():
    """Return the saved canvas layout + current editor info."""
    _check_editor_expired()
    data = config.main.get('canvas_layout', {'cards': []})
    return {'code': 200, 'data': data, 'editor': _editor_session}


@router.post('/layout')
async def save_layout(layout: CanvasLayout):
    """Persist the canvas layout. Only the current editor can save."""
    global _editor_session, _editor_last_seen
    _check_editor_expired()

    session_id = layout.session_id
    if session_id != _editor_session:
        return fastapi.responses.JSONResponse(
            status_code=403, content={'code': 403, 'message': 'Not the current editor',
                                      'editor': _editor_session})

    _editor_last_seen = time.monotonic()

    save_data = layout.dict()
    save_data.pop('session_id', None)
    config.main['canvas_layout'] = save_data
    return {'code': 200}


# ── Per-tool config CRUD ─────────────────────────────────────────────────────

@router.get('/tool-config/{mcp_id}/{tool_name}')
async def get_tool_config(mcp_id: str, tool_name: str):
    """Get saved config for a tool."""
    data = config.main.get(tool_config_key(mcp_id, tool_name), None)
    return {'code': 200, 'data': data}


def all_tool_configs() -> dict:
    """Every saved tool config, keyed by "mcp_id:tool_name[:instance_id]"."""
    result = {}
    try:
        conn = config._get_conn()
        rows = conn.execute(
            "SELECT key, value FROM config WHERE key LIKE ?",
            (f'{_TOOL_CONFIG_PREFIX}%',)
        ).fetchall()
        for key, value in rows:
            tool_key = key[len(_TOOL_CONFIG_PREFIX):]  # "mcp_id:tool_name"
            result[tool_key] = json.loads(value)
    except Exception:
        pass
    return result


def delete_all_tool_configs() -> int:
    """Drop every saved tool config. Returns the number of rows removed.

    Only used when a solution replaces the whole canvas: the incoming cards
    bring their own configs, and leftovers would keep pushing stale settings
    (old topics, old device paths) to plugins that the new canvas reuses.
    """
    try:
        conn = config._get_conn()
        cur = conn.execute("DELETE FROM config WHERE key LIKE ?",
                           (f'{_TOOL_CONFIG_PREFIX}%',))
        conn.commit()
        return cur.rowcount or 0
    except Exception:
        return 0


@router.get('/tool-configs')
async def get_all_tool_configs():
    """Batch-get all tool configs."""
    return {'code': 200, 'data': all_tool_configs()}



@router.put('/tool-config/{mcp_id}/{tool_name}')
async def save_tool_config(mcp_id: str, tool_name: str, body: Any = fastapi.Body(...)):
    """Save config for a tool and apply it to the MCP plugin."""
    config.main[tool_config_key(mcp_id, tool_name)] = body
    apply_tool_config(mcp_id, tool_name, body)
    return {'code': 200}


@router.delete('/tool-config/{mcp_id}/{tool_name}')
async def delete_tool_config(mcp_id: str, tool_name: str):
    """Delete config for a tool."""
    try:
        conn = config._get_conn()
        conn.execute("DELETE FROM config WHERE key = ?",
                     (tool_config_key(mcp_id, tool_name),))
        conn.commit()
    except Exception:
        pass
    return {'code': 200}


# ── Per-instance config CRUD ────────────────────────────────────────────────

@router.get('/tool-config/{mcp_id}/{tool_name}/{instance_id}')
async def get_instance_config(mcp_id: str, tool_name: str, instance_id: str):
    """Get saved config for a specific tool instance."""
    data = config.main.get(tool_config_key(mcp_id, tool_name, instance_id), None)
    return {'code': 200, 'data': data}


@router.put('/tool-config/{mcp_id}/{tool_name}/{instance_id}')
async def save_instance_config(mcp_id: str, tool_name: str, instance_id: str, body: Any = fastapi.Body(...)):
    """Save config for a specific tool instance and apply it."""
    config.main[tool_config_key(mcp_id, tool_name, instance_id)] = body
    apply_tool_config(mcp_id, tool_name, body, instance_id)
    return {'code': 200}


@router.delete('/tool-config/{mcp_id}/{tool_name}/{instance_id}')
async def delete_instance_config(mcp_id: str, tool_name: str, instance_id: str):
    """Delete config for a specific tool instance."""
    try:
        conn = config._get_conn()
        conn.execute("DELETE FROM config WHERE key = ?",
                     (tool_config_key(mcp_id, tool_name, instance_id),))
        conn.commit()
    except Exception:
        pass
    return {'code': 200}

"""
channel/relay_client.py — WebSocket client for connecting to relay server.

Maintains a persistent outbound WebSocket connection to the relay server
(motus-relay.phanthy.com). Receives forwarded webhooks and dispatches them
to the appropriate adapter. Sends reply requests through the relay.

Used in "relay" mode when the device is behind NAT without public IP.
"""

import asyncio
import json
import time
import socket

import config

RELAY_URL = 'wss://motus-relay.phanthy.com/ws'
RECONNECT_DELAY = 5  # seconds between reconnection attempts
MAX_RECONNECT_DELAY = 60


class RelayClient:
    """Maintains WebSocket connection to relay server."""

    def __init__(self, on_webhook):
        """
        Args:
            on_webhook: async callback(platform, channel_id, headers, body) -> None
        """
        self._on_webhook = on_webhook
        self._ws = None
        self._task: asyncio.Task | None = None
        self._connected = False
        self._device_id = ''
        self._channels: set[str] = set()

    @property
    def connected(self) -> bool:
        return self._connected

    def set_channels(self, channels: set[str]):
        """Update the set of channels this device handles via relay."""
        self._channels = channels
        # If connected, send update
        if self._connected and self._ws:
            asyncio.create_task(self._send({
                'type': 'update_channels',
                'channels': list(channels),
            }))

    async def start(self):
        """Start the relay client (connect in background)."""
        self._device_id = self._get_device_id()
        if not self._channels:
            return  # No relay channels configured
        self._task = asyncio.create_task(self._connect_loop())
        print(f'[relay_client] started, device_id={self._device_id}')

    async def stop(self):
        """Stop the relay client."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._ws:
            await self._ws.close()
        self._connected = False
        print('[relay_client] stopped')

    async def send_reply(self, platform: str, channel_id: str, payload: dict) -> bool:
        """Send a reply request through the relay server."""
        if not self._connected or not self._ws:
            return False

        request_id = f'{time.time():.6f}'
        msg = {
            'type': 'reply',
            'device_id': self._device_id,
            'platform': platform,
            'channel_id': channel_id,
            'request_id': request_id,
            'payload': payload,
        }
        return await self._send(msg)

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _connect_loop(self):
        """Reconnect loop with exponential backoff."""
        delay = RECONNECT_DELAY
        while True:
            try:
                await self._connect()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f'[relay_client] connection error: {e}')

            self._connected = False
            print(f'[relay_client] reconnecting in {delay}s...')
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, MAX_RECONNECT_DELAY)

    async def _connect(self):
        """Establish WebSocket connection and handle messages."""
        try:
            import websockets
        except ImportError:
            print('[relay_client] websockets package not installed, relay mode unavailable')
            return

        async with websockets.connect(RELAY_URL, ping_interval=30, ping_timeout=10) as ws:
            self._ws = ws

            # Register
            await ws.send(json.dumps({
                'type': 'register',
                'device_id': self._device_id,
                'channels': list(self._channels),
            }))

            # Wait for ack
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            ack = json.loads(raw)
            if ack.get('type') != 'registered':
                print(f'[relay_client] unexpected ack: {ack}')
                return

            self._connected = True
            print(f'[relay_client] connected to relay, channels={list(self._channels)}')

            # Listen for messages
            async for raw in ws:
                msg = json.loads(raw)
                await self._handle_relay_message(msg)

    async def _handle_relay_message(self, msg: dict):
        """Handle message from relay server."""
        msg_type = msg.get('type', '')

        if msg_type == 'webhook':
            platform = msg.get('platform', '')
            channel_id = msg.get('channel_id', '')
            headers = msg.get('headers', {})
            body = msg.get('body', '')
            await self._on_webhook(platform, channel_id, headers, body)

        elif msg_type == 'reply_result':
            # Reply acknowledgment (fire-and-forget for now)
            request_id = msg.get('request_id', '')
            success = msg.get('success', False)
            if not success:
                print(f'[relay_client] reply failed: {msg.get("error", "unknown")}')

    async def _send(self, msg: dict) -> bool:
        """Send JSON message to relay server."""
        if not self._ws:
            return False
        try:
            await self._ws.send(json.dumps(msg))
            return True
        except Exception as e:
            print(f'[relay_client] send error: {e}')
            return False

    def _get_device_id(self) -> str:
        """Derive a stable device ID."""
        # Use hostname as device identifier
        return socket.gethostname()

"""
channel/adapters/feishu.py — Feishu (Lark) adapter.

Supports webhook mode (platform POSTs events to our endpoint).
Uses Feishu Open API for sending messages.

Config: {app_id, app_secret, verification_token, encrypt_key, mode}
"""

import asyncio
import hashlib
import hmac
import json
import time

import httpx

from channel.adapter import ChannelAdapter, InboundMessage, OutboundMessage, OnMessageCallback


FEISHU_API = 'https://open.feishu.cn/open-apis'


class FeishuAdapter(ChannelAdapter):
    """Feishu/Lark webhook adapter."""

    def __init__(self, channel_id: str, platform: str, config: dict,
                 on_message: OnMessageCallback):
        super().__init__(channel_id, platform, config, on_message)
        self._tenant_token: str = ''
        self._token_expires: float = 0
        self._refresh_task: asyncio.Task | None = None

    async def start(self) -> None:
        app_id = self.config.get('app_id', '')
        app_secret = self.config.get('app_secret', '')
        if not app_id or not app_secret:
            raise ValueError('Feishu app_id and app_secret are required')

        # Get initial tenant access token
        await self._refresh_token()
        # Periodically refresh (token valid for ~2 hours)
        self._refresh_task = asyncio.create_task(self._token_refresh_loop())
        self._running = True
        print(f'[feishu] adapter started: {self.channel_id}')

    async def stop(self) -> None:
        self._running = False
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        print(f'[feishu] adapter stopped: {self.channel_id}')

    async def send_message(self, msg: OutboundMessage) -> None:
        """Send message via Feishu Open API."""
        token = await self._get_token()
        if not token:
            print(f'[feishu] no token, cannot send message')
            return

        headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

        if msg.image_bytes:
            # Upload image first, then send
            image_key = await self._upload_image(token, msg.image_bytes)
            if image_key:
                body = {
                    'receive_id': msg.chat_id,
                    'msg_type': 'image',
                    'content': json.dumps({'image_key': image_key}),
                }
            else:
                body = {
                    'receive_id': msg.chat_id,
                    'msg_type': 'text',
                    'content': json.dumps({'text': msg.image_caption or msg.text or '[image]'}),
                }
        else:
            body = {
                'receive_id': msg.chat_id,
                'msg_type': 'text',
                'content': json.dumps({'text': msg.text}),
            }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f'{FEISHU_API}/im/v1/messages?receive_id_type=chat_id',
                headers=headers,
                json=body,
                timeout=10,
            )
            if resp.status_code != 200:
                print(f'[feishu] send_message error: {resp.status_code} {resp.text[:200]}')

    # ── Webhook handling ─────────────────────────────────────────────────────

    async def handle_webhook(self, request) -> dict:
        """Process inbound webhook from Feishu platform."""
        body = await request.body()
        data = json.loads(body)

        # Challenge verification (initial webhook URL setup)
        if 'challenge' in data:
            return {'challenge': data['challenge']}

        # Verify token if configured
        verification_token = self.config.get('verification_token', '')
        if verification_token:
            header_token = data.get('token', '')
            if header_token and header_token != verification_token:
                return {'code': 403, 'msg': 'invalid token'}

        # Parse event
        event = data.get('event', {})
        header = data.get('header', {})
        event_type = header.get('event_type', '') or data.get('event', {}).get('type', '')

        # Handle message events
        if event_type in ('im.message.receive_v1', 'message'):
            await self._handle_message_event(event)

        return {'code': 0}

    async def _handle_message_event(self, event: dict):
        """Parse Feishu message event into InboundMessage."""
        # v2 event structure
        message = event.get('message', {})
        sender = event.get('sender', {})

        if not message:
            return

        msg_type = message.get('message_type', '')
        chat_id = message.get('chat_id', '')
        sender_id = sender.get('sender_id', {}).get('open_id', '') or sender.get('sender_id', {}).get('user_id', '')
        sender_type = sender.get('sender_type', '')

        # Skip bot messages
        if sender_type == 'app':
            return

        # Extract text content
        text = ''
        if msg_type == 'text':
            content = json.loads(message.get('content', '{}'))
            text = content.get('text', '')
        else:
            text = f'[{msg_type}]'

        if not text or not sender_id:
            return

        msg = InboundMessage(
            platform='feishu',
            channel_id=self.channel_id,
            user_id=sender_id,
            chat_id=chat_id,
            display_name=sender_id,  # Could resolve via user API
            text=text,
        )
        await self._on_message(msg)

    # ── Token management ─────────────────────────────────────────────────────

    async def _refresh_token(self):
        """Get tenant_access_token from Feishu."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f'{FEISHU_API}/auth/v3/tenant_access_token/internal',
                json={
                    'app_id': self.config.get('app_id', ''),
                    'app_secret': self.config.get('app_secret', ''),
                },
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                self._tenant_token = data.get('tenant_access_token', '')
                expire = data.get('expire', 7200)
                self._token_expires = time.time() + expire - 300  # Refresh 5 min early
                print(f'[feishu] token refreshed, expires in {expire}s')
            else:
                print(f'[feishu] token refresh failed: {resp.status_code}')

    async def _token_refresh_loop(self):
        """Periodically refresh tenant access token."""
        try:
            while True:
                sleep_time = max(self._token_expires - time.time(), 60)
                await asyncio.sleep(sleep_time)
                await self._refresh_token()
        except asyncio.CancelledError:
            pass

    async def _get_token(self) -> str:
        if time.time() >= self._token_expires:
            await self._refresh_token()
        return self._tenant_token

    async def _upload_image(self, token: str, image_bytes: bytes) -> str | None:
        """Upload image to Feishu and return image_key."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f'{FEISHU_API}/im/v1/images',
                    headers={'Authorization': f'Bearer {token}'},
                    data={'image_type': 'message'},
                    files={'image': ('image.jpg', image_bytes, 'image/jpeg')},
                    timeout=30,
                )
                if resp.status_code == 200:
                    return resp.json().get('data', {}).get('image_key', '')
        except Exception as e:
            print(f'[feishu] upload image failed: {e}')
        return None

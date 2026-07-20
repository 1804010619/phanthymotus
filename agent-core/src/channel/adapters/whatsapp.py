"""
channel/adapters/whatsapp.py — WhatsApp Cloud API adapter.

Supports webhook mode (Meta POSTs events to our endpoint).
Uses WhatsApp Cloud API for sending messages.

Config: {phone_number_id, access_token, verify_token, app_secret, mode}
"""

import hashlib
import hmac
import json

import httpx
from fastapi.responses import PlainTextResponse

from channel.adapter import ChannelAdapter, InboundMessage, OutboundMessage, OnMessageCallback


WHATSAPP_API = 'https://graph.facebook.com/v18.0'


class WhatsAppAdapter(ChannelAdapter):
    """WhatsApp Cloud API webhook adapter."""

    def __init__(self, channel_id: str, platform: str, config: dict,
                 on_message: OnMessageCallback):
        super().__init__(channel_id, platform, config, on_message)

    async def start(self) -> None:
        phone_id = self.config.get('phone_number_id', '')
        access_token = self.config.get('access_token', '')
        if not phone_id or not access_token:
            raise ValueError('WhatsApp phone_number_id and access_token are required')
        self._running = True
        print(f'[whatsapp] adapter started: {self.channel_id}')

    async def stop(self) -> None:
        self._running = False
        print(f'[whatsapp] adapter stopped: {self.channel_id}')

    async def send_message(self, msg: OutboundMessage) -> None:
        """Send message via WhatsApp Cloud API."""
        phone_id = self.config.get('phone_number_id', '')
        access_token = self.config.get('access_token', '')

        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json',
        }

        if msg.image_bytes:
            # WhatsApp requires media upload first, fallback to text
            body = {
                'messaging_product': 'whatsapp',
                'to': msg.chat_id,
                'type': 'text',
                'text': {'body': msg.image_caption or msg.text or '[image]'},
            }
        else:
            body = {
                'messaging_product': 'whatsapp',
                'to': msg.chat_id,
                'type': 'text',
                'text': {'body': msg.text},
            }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f'{WHATSAPP_API}/{phone_id}/messages',
                headers=headers,
                json=body,
                timeout=10,
            )
            if resp.status_code not in (200, 201):
                print(f'[whatsapp] send_message error: {resp.status_code} {resp.text[:200]}')

    # ── Webhook handling ─────────────────────────────────────────────────────

    async def handle_webhook(self, request) -> dict:
        """Process inbound webhook from WhatsApp/Meta."""
        # Verify signature
        app_secret = self.config.get('app_secret', '')
        if app_secret:
            signature = request.headers.get('x-hub-signature-256', '')
            body = await request.body()
            if not self._verify_signature(app_secret, body, signature):
                return {'error': 'invalid signature'}
            data = json.loads(body)
        else:
            body = await request.body()
            data = json.loads(body)

        # Parse webhook payload
        entries = data.get('entry', [])
        for entry in entries:
            changes = entry.get('changes', [])
            for change in changes:
                value = change.get('value', {})
                messages = value.get('messages', [])
                contacts = value.get('contacts', [])

                contact_map = {c['wa_id']: c.get('profile', {}).get('name', c['wa_id']) for c in contacts}

                for message in messages:
                    await self._handle_message(message, contact_map)

        return {'status': 'ok'}

    async def handle_verification(self, request) -> PlainTextResponse:
        """Handle WhatsApp webhook verification (GET challenge)."""
        params = request.query_params
        mode = params.get('hub.mode', '')
        token = params.get('hub.verify_token', '')
        challenge = params.get('hub.challenge', '')

        verify_token = self.config.get('verify_token', '')
        if mode == 'subscribe' and token == verify_token:
            return PlainTextResponse(challenge)

        return PlainTextResponse('Forbidden', status_code=403)

    async def _handle_message(self, message: dict, contact_map: dict):
        """Parse WhatsApp message into InboundMessage."""
        msg_type = message.get('type', '')
        from_id = message.get('from', '')

        # Extract text
        text = ''
        if msg_type == 'text':
            text = message.get('text', {}).get('body', '')
        elif msg_type == 'image':
            text = message.get('image', {}).get('caption', '') or '[image]'
        elif msg_type == 'audio':
            text = '[audio]'
        elif msg_type == 'video':
            text = '[video]'
        elif msg_type == 'document':
            text = '[document]'
        else:
            text = f'[{msg_type}]'

        if not text or not from_id:
            return

        display_name = contact_map.get(from_id, from_id)

        msg = InboundMessage(
            platform='whatsapp',
            channel_id=self.channel_id,
            user_id=from_id,
            chat_id=from_id,  # WhatsApp uses phone number as chat_id
            display_name=display_name,
            text=text,
        )
        await self._on_message(msg)

    def _verify_signature(self, app_secret: str, body: bytes, signature: str) -> bool:
        """Verify X-Hub-Signature-256 header."""
        if not signature.startswith('sha256='):
            return False
        expected = 'sha256=' + hmac.HMAC(
            app_secret.encode(), body, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

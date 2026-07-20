"""
channel/adapters/feishu.py — Feishu (Lark) adapter using WebSocket long connection.

Uses lark-oapi SDK's built-in WebSocket mode — outbound connection to Feishu servers,
no public IP or webhook needed. Same pattern as Telegram long-polling and Slack Socket Mode.

Requires: pip install lark-oapi
Config: {app_id, app_secret}
"""

import asyncio
import json

from channel.adapter import ChannelAdapter, InboundMessage, OutboundMessage, OnMessageCallback


class FeishuAdapter(ChannelAdapter):
    """Feishu/Lark adapter using SDK WebSocket long connection."""

    def __init__(self, channel_id: str, platform: str, config: dict,
                 on_message: OnMessageCallback):
        super().__init__(channel_id, platform, config, on_message)
        self._client = None
        self._task: asyncio.Task | None = None
        self._api_client = None

    async def start(self) -> None:
        app_id = self.config.get('app_id', '')
        app_secret = self.config.get('app_secret', '')
        if not app_id or not app_secret:
            raise ValueError('Feishu app_id and app_secret are required')

        import lark_oapi as lark

        # Store the main event loop for cross-thread scheduling
        self._loop = asyncio.get_event_loop()

        # Create API client for sending messages
        self._api_client = lark.Client.builder() \
            .app_id(app_id) \
            .app_secret(app_secret) \
            .build()

        # Create event handler
        event_handler = lark.EventDispatcherHandler.builder("", "") \
            .register_p2_im_message_receive_v1(self._handle_message_event) \
            .build()

        # Create WebSocket long connection client
        self._client = lark.ws.Client(
            app_id=app_id,
            app_secret=app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.WARNING,
        )

        # Start in background thread (SDK blocks)
        self._task = asyncio.create_task(self._run())
        self._running = True
        print(f'[feishu] adapter started (WebSocket mode): {self.channel_id}')

    async def _run(self):
        """Run the lark WebSocket client in a dedicated thread with its own event loop."""
        import threading

        def _thread_target():
            # lark SDK uses a module-level event loop (loop.run_until_complete)
            # Must create a fresh loop for this thread
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                # Monkey-patch the SDK's module-level loop reference
                import lark_oapi.ws.client as ws_mod
                ws_mod.loop = new_loop
                self._client.start()
            except Exception as e:
                print(f'[feishu] WebSocket thread error: {e}')
                self._running = False
            finally:
                new_loop.close()

        self._thread = threading.Thread(target=_thread_target, daemon=True)
        self._thread.start()

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._client = None
        print(f'[feishu] adapter stopped: {self.channel_id}')

    async def send_message(self, msg: OutboundMessage) -> None:
        """Send message via Feishu Open API."""
        if not self._api_client:
            return

        import lark_oapi as lark
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        body = CreateMessageRequestBody.builder() \
            .receive_id(msg.chat_id) \
            .msg_type('text') \
            .content(json.dumps({'text': msg.text})) \
            .build()

        request = CreateMessageRequest.builder() \
            .receive_id_type('chat_id') \
            .request_body(body) \
            .build()

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None, self._api_client.im.v1.message.create, request
            )
            if not response.success():
                print(f'[feishu] send_message error: {response.code} {response.msg}')
        except Exception as e:
            print(f'[feishu] send_message exception: {e}')

    def _handle_message_event(self, data):
        """Handle im.message.receive_v1 event from SDK callback (runs in thread)."""
        try:
            event = data.event
            message = event.message
            sender = event.sender

            # Skip bot messages
            if sender.sender_type == 'app':
                return

            # Extract text
            msg_type = message.message_type
            text = ''
            if msg_type == 'text':
                content = json.loads(message.content)
                text = content.get('text', '')
            else:
                text = f'[{msg_type}]'

            if not text:
                return

            sender_id = sender.sender_id.open_id or sender.sender_id.user_id or ''
            chat_id = message.chat_id

            msg = InboundMessage(
                platform='feishu',
                channel_id=self.channel_id,
                user_id=sender_id,
                chat_id=chat_id,
                display_name=sender_id,
                text=text,
            )

            # Schedule coroutine from SDK thread to main event loop
            asyncio.run_coroutine_threadsafe(self._on_message(msg), self._loop)

        except Exception as e:
            print(f'[feishu] handle message error: {e}')

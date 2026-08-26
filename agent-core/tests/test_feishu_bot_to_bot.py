"""Focused checks for Feishu group bot-to-bot mentions."""

import json
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ['DB_PATH'] = os.path.join(tempfile.mkdtemp(), 'test.db')

import config  # noqa: E402
import mcp_client  # noqa: E402
from channel.adapter import ChannelAdapter, InboundMessage, OutboundMessage  # noqa: E402
from channel.adapters.feishu import (  # noqa: E402
    FeishuAdapter,
    _BOT_FINAL_LABEL,
    _BOT_REQUEST_LABEL,
    _TEXT_CHUNK,
)
from channel.manager import ChannelManager  # noqa: E402


class DummyAdapter(ChannelAdapter):
    def __init__(self):
        super().__init__('feishu_test', 'feishu', {}, mock.AsyncMock())
        self.sent = []
        self._running = True

    async def start(self):
        self._running = True

    async def stop(self):
        self._running = False

    async def send_message(self, msg):
        self.sent.append(msg)


def _raw_bot_event(message_id='om_1', *, chat_type='group', mentions=None,
                   sender_id='ou_peer', text='请检查结果'):
    return {
        'message_id': message_id,
        'message_type': 'text',
        'content': json.dumps({'text': text}),
        'chat_id': 'oc_group',
        'chat_type': chat_type,
        'sender_id': sender_id,
        'sender_type': 'bot',
        'mentions': mentions if mentions is not None else [
            {
                'key': '@_user_1',
                'open_id': 'ou_self',
                'user_id': '',
                'name': 'Self bot',
                'mentioned_type': 'bot',
            }
        ],
    }


class FeishuAdapterBotEventTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.received = []

        async def receive(msg):
            self.received.append(msg)

        self.adapter = FeishuAdapter(
            'feishu_test',
            'feishu',
            {'bot_to_bot_enabled': True},
            receive,
        )
        self.adapter._bot_open_id = 'ou_self'

    async def test_only_group_bot_message_that_mentions_self_is_accepted(self):
        await self.adapter._process_event(_raw_bot_event(
            text=f'{_BOT_REQUEST_LABEL}\n@_user_1 请给出检查结果',
        ))

        self.assertEqual(len(self.received), 1)
        msg = self.received[0]
        self.assertEqual(msg.sender_type, 'bot')
        self.assertEqual(msg.chat_type, 'group')
        self.assertTrue(msg.expect_reply)
        self.assertEqual(msg.text, '@_user_1 请给出检查结果')
        self.assertTrue(msg.mentions[0]['is_self'])

        await self.adapter._process_event(_raw_bot_event('om_p2p', chat_type='p2p'))
        await self.adapter._process_event(_raw_bot_event('om_no_at', mentions=[]))
        await self.adapter._process_event(_raw_bot_event('om_self', sender_id='ou_self'))
        self.assertEqual(len(self.received), 1)

    async def test_final_label_and_unlabelled_peer_message_have_explicit_semantics(self):
        await self.adapter._process_event(_raw_bot_event(
            text=f'{_BOT_FINAL_LABEL}\n@_user_1 最终结果',
        ))
        await self.adapter._process_event(_raw_bot_event(
            'om_external', text='@_user_1 外部机器人请求',
        ))

        self.assertFalse(self.received[0].expect_reply)
        self.assertTrue(self.received[1].expect_reply)

    async def test_human_messages_remain_compatible_and_bot_feature_defaults_closed(self):
        human = _raw_bot_event('om_human', chat_type='p2p', mentions=[])
        human['sender_type'] = 'user'
        await self.adapter._process_event(human)

        self.adapter.config['bot_to_bot_enabled'] = False
        await self.adapter._process_event(_raw_bot_event('om_disabled'))

        self.assertEqual(len(self.received), 1)
        self.assertEqual(self.received[0].sender_type, 'user')
        self.assertIsNone(self.received[0].expect_reply)

    async def test_unknown_sender_type_is_dropped(self):
        unknown = _raw_bot_event('om_unknown_sender')
        unknown['sender_type'] = ''

        await self.adapter._process_event(unknown)

        self.assertEqual(self.received, [])

    async def test_missing_bot_id_log_is_throttled_with_drop_count(self):
        self.adapter._bot_open_id = ''

        with mock.patch('channel.adapters.feishu.time.time', side_effect=[100, 110, 161]), \
                mock.patch('builtins.print') as printed:
            await self.adapter._process_event(_raw_bot_event('om_missing_1'))
            await self.adapter._process_event(_raw_bot_event('om_missing_2'))
            await self.adapter._process_event(_raw_bot_event('om_missing_3'))

        logs = [str(call.args[0]) for call in printed.call_args_list
                if 'bot events dropped' in str(call.args[0])]
        self.assertEqual(len(logs), 2)
        self.assertIn('count=1', logs[0])
        self.assertIn('count=2', logs[1])

    async def test_probe_reads_root_bot_open_id(self):
        self.adapter._request = mock.AsyncMock(return_value={
            'code': 0,
            'msg': 'ok',
            'bot': {'open_id': 'ou_from_probe'},
        })

        ok, reason = await self.adapter._probe(force=True)

        self.assertTrue(ok, reason)
        self.assertEqual(self.adapter._bot_open_id, 'ou_from_probe')


class FeishuAdapterBotSendTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.adapter = FeishuAdapter(
            'feishu_test',
            'feishu',
            {'bot_to_bot_enabled': True},
            mock.AsyncMock(),
        )
        self.adapter._running = True
        self.adapter._bot_open_id = 'ou_self'
        self.sent = []

        async def capture(chat_id, msg_type, content):
            self.sent.append((chat_id, msg_type, content))

        self.adapter._send_raw = capture

    async def test_structured_mentions_carry_request_or_final_label(self):
        await self.adapter.send_message(OutboundMessage(
            chat_id='oc_group',
            text='请检查',
            mention_open_id='ou_peer',
            expect_reply=True,
        ))
        await self.adapter.send_message(OutboundMessage(
            chat_id='oc_group',
            text='这是最终结果',
            mention_open_id='ou_peer',
        ))

        request_text = self.sent[0][2]['text']
        final_text = self.sent[1][2]['text']
        self.assertTrue(request_text.startswith(_BOT_REQUEST_LABEL))
        self.assertIn('<at user_id="ou_peer"></at>', request_text)
        self.assertTrue(final_text.startswith(_BOT_FINAL_LABEL))

    async def test_invalid_self_or_long_mentions_send_nothing(self):
        cases = [
            OutboundMessage(chat_id='oc_group', text='x', mention_open_id='bad'),
            OutboundMessage(chat_id='oc_group', text='x', mention_open_id='ou_self'),
            OutboundMessage(
                chat_id='oc_group',
                text='x' * _TEXT_CHUNK,
                mention_open_id='ou_peer',
            ),
        ]
        for msg in cases:
            with self.assertRaises(ValueError):
                await self.adapter.send_message(msg)
        self.assertEqual(self.sent, [])


class InternalChannelReplyDispatchTest(unittest.IsolatedAsyncioTestCase):
    async def test_forwards_bot_mention_arguments(self):
        send_reply = mock.AsyncMock(return_value='sent')

        with mock.patch('channel.manager.manager.send_reply', send_reply):
            result = await mcp_client._dispatch_internal(
                'channel',
                'channel_reply',
                {
                    'action': 'send',
                    'instance_id': 'reply_1',
                    'text': '请检查',
                    'mention_open_id': 'ou_peer',
                    'source_message_id': 'om_request',
                    'expect_reply': True,
                },
            )

        self.assertEqual(result, 'sent')
        send_reply.assert_awaited_once_with(
            instance_id='reply_1',
            text='请检查',
            files=[],
            mention_open_id='ou_peer',
            source_message_id='om_request',
            expect_reply=True,
        )


class ChannelManagerBotReplyTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        config.main['channel_configs'] = [{
            'id': 'feishu_test',
            'platform': 'feishu',
            'enabled': True,
            'bot_to_bot_enabled': True,
            'config': {},
        }]
        config.main['channel_last_context'] = {}
        config.main['channel_message_contexts'] = {}
        self.manager = ChannelManager()
        self.adapter = DummyAdapter()
        self.manager._adapters['feishu_test'] = self.adapter

    async def test_final_bot_message_cannot_continue_mentioning(self):
        self.manager._set_last_context(
            'feishu_test', 'oc_group', 'ou_peer',
            message_id='om_final', sender_type='bot', chat_type='group',
            expect_reply=False,
        )

        result = await self.manager.send_to_channel(
            'feishu_test',
            text='收到',
            mention_open_id='ou_peer',
            source_message_id='om_final',
        )

        self.assertIn('final answer', result)
        self.assertEqual(self.adapter.sent, [])

    async def test_bot_request_can_only_reply_to_sender_and_unknown_context_is_rejected(self):
        self.manager._set_last_context(
            'feishu_test', 'oc_group', 'ou_peer',
            message_id='om_request', sender_type='bot', chat_type='group',
            expect_reply=True,
        )

        wrong_target = await self.manager.send_to_channel(
            'feishu_test', text='结果', mention_open_id='ou_other',
            source_message_id='om_request',
        )
        stale = await self.manager.send_to_channel(
            'feishu_test', text='结果', mention_open_id='ou_peer',
            source_message_id='om_old',
        )
        sent = await self.manager.send_to_channel(
            'feishu_test', text='结果', mention_open_id='ou_peer',
            source_message_id='om_request', expect_reply=False,
        )

        self.assertIn('only @ the bot that sent', wrong_target)
        self.assertIn('unknown or expired', stale)
        self.assertIn('final answer', sent)
        self.assertEqual(len(self.adapter.sent), 1)

    async def test_human_reply_routes_by_trigger_message_not_latest_chat(self):
        self.manager._set_last_context(
            'feishu_test', 'oc_private', 'ou_human',
            message_id='om_private', sender_type='user', chat_type='p2p',
        )
        self.manager._set_last_context(
            'feishu_test', 'oc_group', 'ou_other',
            message_id='om_newer', sender_type='user', chat_type='group',
        )

        missing = await self.manager.send_to_channel(
            'feishu_test', text='私聊回复',
        )
        unknown = await self.manager.send_to_channel(
            'feishu_test', text='私聊回复', source_message_id='om_unknown',
        )
        sent = await self.manager.send_to_channel(
            'feishu_test', text='私聊回复', source_message_id='om_private',
        )

        self.assertIn('missing source_message_id', missing)
        self.assertIn('unknown or expired', unknown)
        self.assertIn('Reply sent', sent)
        self.assertEqual(len(self.adapter.sent), 1)
        self.assertEqual(self.adapter.sent[0].chat_id, 'oc_private')

    async def test_malformed_persisted_context_is_replaced(self):
        config.main['channel_last_context'] = {
            'broken': {
                'chat_id': 'oc_bad', 'user_id': 'ou_bad',
                'message_id': 'om_bad', 'ts': 'not-a-number',
            },
        }
        config.main['channel_message_contexts'] = {
            'feishu_test': {
                'om_broken': 'not-a-context',
                'om_bad_ts': {
                    'chat_id': 'oc_bad', 'user_id': 'ou_bad',
                    'message_id': 'om_bad_ts', 'ts': 'not-a-number',
                },
                'om_no_chat': {
                    'user_id': 'ou_bad', 'message_id': 'om_no_chat', 'ts': 1,
                },
            },
        }

        self.assertEqual(self.manager.resolve_target_channel()[0], '')

        self.manager._set_last_context(
            'feishu_test', 'oc_private', 'ou_human',
            message_id='om_valid', sender_type='user', chat_type='p2p',
        )
        result = await self.manager.send_to_channel(
            'feishu_test', text='回复', source_message_id='om_valid',
        )

        self.assertIn('Reply sent', result)
        saved = config.main.get('channel_message_contexts', {})['feishu_test']
        self.assertEqual(set(saved), {'om_valid'})

    async def test_bot_inbound_bypasses_people_acl_as_viewer(self):
        config.main['canvas_layout'] = {'cards': [{
            'mcpId': 'channel',
            'toolName': 'channel_request',
            'id': 'input_1',
        }]}
        config.main['tool_config:channel:channel_request:input_1'] = {
            'channel_id': 'feishu_test',
        }
        published = mock.AsyncMock()
        pushed = mock.AsyncMock()
        inspection = types.ModuleType('api.inspection')
        inspection.publish_to_topic = published
        motus_stream = types.ModuleType('api.motus_stream')
        motus_stream.push_event = pushed
        msg = InboundMessage(
            platform='feishu',
            channel_id='feishu_test',
            user_id='ou_peer',
            chat_id='oc_group',
            display_name='Peer bot',
            text='任务',
            message_id='om_request',
            sender_type='bot',
            chat_type='group',
            mentions=[{'open_id': 'ou_self', 'is_self': True}],
            expect_reply=True,
        )

        with mock.patch.dict(sys.modules, {
            'api.inspection': inspection,
            'api.motus_stream': motus_stream,
        }), mock.patch('channel.manager.acl.get_user') as get_user, \
                mock.patch('channel.manager.acl.upsert_user') as upsert_user:
            await self.manager._on_inbound_message(msg)

        get_user.assert_not_called()
        upsert_user.assert_not_called()
        payload = json.loads(published.await_args.args[1])
        self.assertEqual(payload['user_role'], 'viewer')
        self.assertEqual(payload['sender_type'], 'bot')
        self.assertTrue(payload['expect_reply'])


if __name__ == '__main__':
    unittest.main()

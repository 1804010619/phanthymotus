"""Channel turns get one retry when the model writes text without a tool call."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

from event.llm import _channel_tool_retry_message  # noqa: E402


class ChannelReplyRetryTest(unittest.TestCase):
    def setUp(self):
        self.trigger = {
            'payload': {'sources': ['dds:/channel/request/wlcb_23']},
        }

    def test_retries_only_first_channel_round_with_text(self):
        retry = _channel_tool_retry_message(self.trigger, 0, '在的')

        self.assertIn('channel_reply', retry)
        self.assertIn('wlcb_23', retry)
        self.assertEqual(_channel_tool_retry_message(self.trigger, 1, '在的'), '')
        self.assertEqual(_channel_tool_retry_message(self.trigger, 0, '  '), '')
        self.assertEqual(_channel_tool_retry_message({'payload': {'sources': []}}, 0, '在的'), '')


if __name__ == '__main__':
    unittest.main()

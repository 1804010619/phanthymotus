"""停止推送时那条提示，说的话和放的位置必须对得上底下是什么。

没有 JS 测试框架、没有构建步骤（见 test_control_renderer_registered.py 的同一段
理由），所以这里做文本检查 —— 而文本检查恰好抓得住这个 bug：它不是逻辑错，是
一句为画布写的话被套在了文本面板上。

现场截图里的样子：一条红字「画面是最后一帧」压在一行 JSON 日志上，红字和那行
数据都读不成。文本面板底下是滚动日志，不是一张停住的画；"画面"没有指涉，居中
无背景的覆盖层则正好糊掉一行。

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_detail_panel_stale_notice.py
"""
import pathlib
import re
import unittest

SOURCE = (pathlib.Path(__file__).resolve().parents[1]
          / 'web' / 'js' / 'detail-panel.js').read_text()


def _fn(name: str) -> str:
    match = re.search(rf'function {name}\(.*?\n\}}', SOURCE, re.S)
    assert match, f'{name} 不见了'
    return match.group(0)


class StaleNoticeTest(unittest.TestCase):

    def test_the_wording_depends_on_what_is_underneath(self):
        """画布会把最后一帧留在屏幕上；日志不会，它是一段历史。"""
        body = _fn('_staleNotice')

        self.assertIn('画面是最后一帧', body)
        self.assertIn('以下是最后收到的内容', body)
        # 文本那支绝不能说"画面"
        text_branch = body.split('if (_textLike)', 1)[1].split('}', 1)[0]
        self.assertNotIn('画面', text_branch)

    def test_the_text_banner_has_a_background(self):
        """没有底色的覆盖层会和底下那行数据糊在一起，两边都读不成 —— 这正是
        截图里发生的事。"""
        text_branch = _fn('_staleNotice').split('if (_textLike)', 1)[1]
        self.assertIn('background:', text_branch.split('return', 1)[0])

    def test_the_text_banner_does_not_sit_on_the_newest_lines(self):
        """日志自动滚到底，最新的内容在下面。横幅钉在顶部，不遮它们。"""
        style = _fn('_staleNotice').split('if (_textLike)', 1)[1].split('return', 1)[0]
        self.assertIn('top:0', style)
        self.assertNotIn('top:50%', style)

    def test_a_canvas_still_gets_the_centred_notice(self):
        """图上没有会被遮住的内容，居中是对的。"""
        body = _fn('_staleNotice')
        canvas_branch = body.rsplit('}', 2)[0].rsplit('_CENTERED', 1)
        self.assertEqual(len(canvas_branch), 2, '画布那支应当用 _CENTERED')

    def test_the_style_is_reset_for_each_topic(self):
        """_staleNotice 会把样式改成顶部横幅。不复位的话，下一个视频面板的
        「正在连接…」会贴在顶边而不是居中。"""
        self.assertIn('_status.style.cssText = _CENTERED;', SOURCE)
        self.assertRegex(SOURCE, r'_textLike = false;')

    def test_only_text_like_renderers_take_the_banner(self):
        """判断依据是选中的渲染器本身，不是格式字符串 —— 格式的写法会变，
        渲染器的身份不会。"""
        self.assertIn('Renderer === TextRenderer || Renderer === ActivityRenderer',
                      SOURCE)


if __name__ == '__main__':
    unittest.main()

from .manager import SubagentManager
from .protocol import SubagentSpec, SubagentResult, SubagentStatus

__all__ = ['SubagentManager', 'SubagentSpec', 'SubagentResult', 'SubagentStatus']

# Module-level reference to the active manager (set by event/llm.py on init)
_manager_instance: SubagentManager | None = None


def _set_manager(mgr: SubagentManager) -> None:
    global _manager_instance
    _manager_instance = mgr


def _get_active_subagents() -> list[SubagentStatus]:
    """Get active subagent statuses (used by prompt.py for L2 dynamic)."""
    if _manager_instance is None:
        return []
    return _manager_instance.list_active()


def _get_active_digests(max_turns: int = 3, since: dict | None = None) -> list[dict]:
    """running 子代理的**近况**：目标 + 最近几轮的真实消息。

    进度播报专用。只给 SubagentStatus（目标 + 轮数）的话，汇报器手里除了目标本身什么
    都没有，只能把目标换个说法念一遍 —— Orin5 实测播出来的就是"投研报告还在调研中，
    完成后告诉你结果"，等于没说。

    `since` 是 {agent_id: 上次已经喂过的 turn 数}，只返回那之后的新 turn。不给这个的话
    每次都是一个前后重叠的滑动窗口，模型只能把累积状态重新总结一遍 —— Tianyi 实测连着
    三条播报越说越像，第三条几乎是第二条加一个词，末尾都靠"马上整理成报告"凑数。

    水位用 **turn 数**而不是 rounds_completed：两者不保证一一对应（一轮可能不产生 turn），
    而这里要回答的恰恰是"哪几条我还没喂过"。子代理自己的上下文压缩会裁掉旧 turn，那时
    水位会大于列表长度，切片自然为空 —— 退化成"这次没有新东西"，比重复播一遍安全。

    返回的 `turn_count` 是当前总 turn 数，调用方拿它更新水位。
    """
    if _manager_instance is None:
        return []
    out = []
    for agent in getattr(_manager_instance, '_agents', {}).values():
        if agent.status != 'running':
            continue
        try:
            turns = list(agent.context.turns)
        except Exception:
            turns = []
        if since is None:
            picked = turns[-max_turns:] if max_turns > 0 else []
        else:
            fresh = turns[since.get(agent.id, 0):]
            if not fresh:
                continue                 # 上次喂过之后没有新 turn，没什么可说的
            picked = fresh[-max_turns:] if max_turns > 0 else []
        out.append({
            'id': agent.id,
            'goal': agent.spec.goal,
            'rounds': agent.rounds_completed,
            'turn_count': len(turns),
            'turns': picked,
        })
    return out

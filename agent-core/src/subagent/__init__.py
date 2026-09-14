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


def _get_active_digests(max_turns: int = 3) -> list[dict]:
    """running 子代理的**近况**：目标 + 最近几轮的真实消息。

    进度播报专用。只给 SubagentStatus（目标 + 轮数）的话，汇报器手里除了目标本身什么
    都没有，只能把目标换个说法念一遍 —— Orin5 实测播出来的就是"投研报告还在调研中，
    完成后告诉你结果"，等于没说。它需要看到子代理**具体搜了什么、拿到了什么**，才谈得上
    "做了什么 / 发现了什么 / 接下来做什么"。

    返回原始 turns，由调用方决定怎么渲染和截断（agent-core 那边用 _turns_to_text，
    工具结果会被截到 200 字符）。
    """
    if _manager_instance is None:
        return []
    out = []
    for agent in getattr(_manager_instance, '_agents', {}).values():
        if agent.status != 'running':
            continue
        try:
            turns = list(agent.context.turns)[-max_turns:]
        except Exception:
            turns = []
        out.append({
            'id': agent.id,
            'goal': agent.spec.goal,
            'rounds': agent.rounds_completed,
            'turns': turns,
        })
    return out

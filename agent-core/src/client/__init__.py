"""
client/__init__.py — LLM client singleton with unified usage tracking.

All LLM calls should go through `client.call()` to ensure usage is recorded.
"""

import client.llm
import perf_log

# Raw client instance (for direct access if needed)
llm = client.llm.Client()


async def call(
    message_list: list[dict],
    tool_list: list[dict],
    cancel_event=None,
    model_override: str | None = None,
    trace_id: str = '',
) -> dict:
    """Unified LLM call with automatic usage recording.

    All components (main agent, subagent, etc.) should use this function
    instead of calling client.llm directly, to ensure token usage is tracked.

    Args:
        message_list: Messages for the LLM
        tool_list: Available tools
        cancel_event: Optional asyncio.Event to cancel the call
        model_override: Optional model name override
        trace_id: Optional trace ID for usage attribution

    Returns:
        LLM response dict (with _usage field attached)
    """
    response = await llm(
        message_list=message_list,
        tool_list=tool_list,
        cancel_event=cancel_event,
        model_override=model_override,
    )

    # Record usage
    usage = response.get('_usage')
    if usage:
        perf_log.record_usage(trace_id or 'unknown', usage)

    return response

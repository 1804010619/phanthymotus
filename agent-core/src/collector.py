"""
collector.py — 双队列事件收集器。

架构：
  - P>0 事件（ASR/message/channel）→ 立即送 main agent，或 busy 时暂存等 turn 结束
  - P=0 事件（sensor/scheduler 等）→ 独立节奏送 bg subagent，main agent 永远不看到
  - Ring buffer 保留所有事件（供 raw_input_info 按需查询）
  - 支持 cancel_event 信号（仅 P>P_current 时 cancel）
"""

import asyncio
import datetime
import json as _json
import time
from collections import deque

import config
import event_bus


# ── P>0 管道 ─────────────────────────────────────────────────────────────────
_priority_pending: deque = deque()   # P>0 事件（busy 时暂存）
_output: asyncio.Queue = asyncio.Queue(maxsize=64)  # main agent 消费端

# ── P=0 管道 ─────────────────────────────────────────────────────────────────
_bg_buffer: deque = deque()          # P=0 事件（按节奏送 bg subagent）
_bg_last_accepted: dict[str, float] = {}  # per-source throttle for bg
_BG_THROTTLE_INTERVAL = 1.0

# ── 共享状态 ──────────────────────────────────────────────────────────────────
_busy: bool = False
_cancel_event: asyncio.Event | None = None
_current_turn_priority: int = 0
_source_ring: dict[str, deque] = {}  # per-source ring buffer（所有事件）

# 优先级判定规则
_PRIORITY_SOURCES = {'asr', 'message', 'channel'}


def _extract_priority(ev: dict) -> int:
    """从事件中解析 priority。JSON text 中的 priority 字段优先，否则按 source 匹配。"""
    text = ev.get('text', '')
    if text and text.startswith('{'):
        try:
            data = _json.loads(text)
            p = data.get('priority')
            if p is not None:
                return int(p)
        except (ValueError, TypeError):
            pass
    source = ev.get('source', '').lower()
    for key in _PRIORITY_SOURCES:
        if key in source:
            return 1
    return 0


def _extract_perf_timestamps(ev: dict):
    """从 ASR 事件 JSON 中提取性能 span 数据。"""
    text = ev.get('text', '')
    if not text or not text.startswith('{'):
        return
    try:
        data = _json.loads(text)
    except (ValueError, TypeError):
        return
    if 'spans' in data:
        ev['_perf_spans'] = data['spans']
        return
    spans = []
    audio_start = data.get('audio_start_ts')
    audio_end = data.get('audio_end_ts')
    asr_complete = data.get('asr_complete_ts')
    if audio_start and audio_start > 1e9 and audio_end and audio_end > 1e9:
        spans.append({'span': 'vad_collect', 'start_ts': audio_start, 'end_ts': audio_end,
                      'meta': {'audio_ms': data.get('audio_duration_ms')}})
    if audio_end and audio_end > 1e9 and asr_complete and asr_complete > 1e9:
        spans.append({'span': 'asr_inference', 'start_ts': audio_end, 'end_ts': asr_complete,
                      'meta': {'text_length': data.get('text_length')}})
    if spans:
        ev['_perf_spans'] = spans


# ── 公开接口 ──────────────────────────────────────────────────────────────────

def set_busy(busy: bool):
    """由 agent loop 调用：标记当前是否正在执行 turn。"""
    global _busy
    _busy = busy
    if not busy and _priority_pending:
        asyncio.ensure_future(_emit_priority())


def set_cancel_event(ev: asyncio.Event | None):
    """由 agent loop 调用：注册/清除当前 turn 的取消信号。"""
    global _cancel_event
    _cancel_event = ev


def set_turn_priority(priority: int):
    """由 agent loop 调用：设置当前 turn 的 priority。"""
    global _current_turn_priority
    _current_turn_priority = priority


async def next_trigger() -> dict:
    """阻塞等待下一批 P>0 事件（main agent 消费端）。"""
    return await _output.get()


def get_source_detail(source: str, limit: int = 20) -> list[dict]:
    """获取指定 source 的原始事件详情（从 ring buffer）。"""
    ring = _source_ring.get(source)
    if not ring:
        return []
    return list(ring)[-limit:]


def get_available_sources() -> list[str]:
    """返回当前有数据的所有 source 名称列表。"""
    return list(_source_ring.keys())


# ── 内部：P>0 管道 ────────────────────────────────────────────────────────────

async def _emit_priority():
    """busy 结束后，立即 emit 暂存的 P>0 事件。"""
    if not _priority_pending:
        return
    batch = list(_priority_pending)
    _priority_pending.clear()
    await _emit_batch(batch, urgent=True)


async def _emit_batch(batch: list[dict], urgent: bool = False):
    """将 P>0 事件格式化并放入 output。"""
    formatted = _format_priority_batch(batch)
    trigger = {
        'source': 'collector',
        'text': formatted,
        'payload': {'event_count': len(batch), 'sources': [e['source'] for e in batch]},
        'ts': batch[-1]['ts'],
        '_perf_trigger_emit_ts': time.time(),
        '_urgent': urgent,
    }
    for ev in reversed(batch):
        if '_perf_spans' in ev:
            trigger['_perf_spans'] = ev['_perf_spans']
            break
    await _output.put(trigger)


def _format_priority_batch(events: list[dict]) -> str:
    """格式化 P>0 事件为 XML（保留原文）。"""
    parts = []
    for ev in events:
        ts = datetime.datetime.fromtimestamp(ev['ts']).strftime('%Y-%m-%dT%H:%M:%S')
        channel = _infer_channel(ev)
        source = ev.get('source', '')
        text = ev.get('text', '')
        parts.append(f'<event source="{source}" channel="{channel}" ts="{ts}">\n{text}\n</event>')
    return '\n'.join(parts)


# ── 内部：P=0 管道 ────────────────────────────────────────────────────────────

def _bg_buffer_add(ev: dict):
    """将 P=0 事件加入 bg buffer（per-source throttle）。"""
    source = ev.get('source', 'unknown')
    now = ev.get('ts', time.time())
    last_ts = _bg_last_accepted.get(source, 0)

    if now - last_ts < _BG_THROTTLE_INTERVAL:
        # 替换同 source 最后一条
        for i in range(len(_bg_buffer) - 1, -1, -1):
            if _bg_buffer[i].get('source') == source:
                _bg_buffer[i] = ev
                return
    _bg_last_accepted[source] = now
    _bg_buffer.append(ev)

    # FIFO 限制
    max_window = config.main.get('event', {}).get('llm', {}).get('collector_max_window', 20)
    while len(_bg_buffer) > max_window:
        _bg_buffer.popleft()


def _format_bg_batch(events: list[dict]) -> str:
    """格式化 P=0 事件为摘要（按 source 分组）。"""
    groups: dict[str, list[dict]] = {}
    for ev in events:
        source = ev.get('source', 'unknown')
        groups.setdefault(source, []).append(ev)

    parts = []
    for source, evs in groups.items():
        ts = datetime.datetime.fromtimestamp(evs[-1]['ts']).strftime('%Y-%m-%dT%H:%M:%S')
        last_text = evs[-1].get('text', '')
        if len(evs) == 1:
            parts.append(f'<source name="{source}" ts="{ts}">\n{last_text}\n</source>')
        else:
            parts.append(f'<source name="{source}" count="{len(evs)}" ts="{ts}">\n{last_text}\n(共 {len(evs)} 条，显示最新)\n</source>')
    return '\n'.join(parts)


async def _route_to_bg_subagent(batch: list[dict]) -> bool:
    """将 P=0 事件批次路由到 bg subagent。"""
    try:
        from subagent import _manager_instance
    except ImportError:
        return False

    if not _manager_instance:
        return False

    bg_config = config.main.get('subagent', {})
    if not bg_config.get('bg_route_enabled', True):
        return False

    summary = _format_bg_batch(batch)

    # 同步 main agent 最近对话上下文
    try:
        from event.llm import get_recent_context
        recent_context = get_recent_context(max_turns=5)
    except (ImportError, AttributeError):
        recent_context = ''

    if recent_context:
        message = f'[主代理最近对话]\n{recent_context}\n\n[新数据]\n{summary}'
    else:
        message = summary

    # 检查是否有活跃的 bg subagent
    active = _manager_instance.list_active()
    bg_agents = [a for a in active if a.goal.startswith('[bg]')]

    if bg_agents:
        _manager_instance.send_message(bg_agents[0].id, message)
    else:
        from subagent.protocol import SubagentSpec, P_LOW
        spec = SubagentSpec(
            goal='[bg] 后台监控：分析传入的信息。如果发现值得注意的变化或异常，调用 subagent_report 上报。无重要变化时直接调用 subagent_finish。不要主动调用任何工具去查询数据，只分析传入的内容。',
            priority=P_LOW,
            model=bg_config.get('bg_model'),
            tool_deny=['mcp__*'],
            max_rounds=50,
            timeout_s=3600,
            context_seed=message,
        )
        await _manager_instance.spawn(spec)

    return True


# ── Channel 推断 ──────────────────────────────────────────────────────────────

def _infer_channel(ev: dict) -> str:
    """从事件 source 推断渠道标签。"""
    source = ev.get('source', '')
    if '/channel/' in source or source.startswith('channel:'):
        text = ev.get('text', '')
        if text and text.startswith('{'):
            try:
                data = _json.loads(text)
                platform = data.get('platform', '')
                if platform:
                    return f'channel:{platform}'
            except (ValueError, TypeError):
                pass
        return 'channel'
    if '/remote_control/message' in source:
        return 'remote_web'
    if 'asr' in source.lower() or '/mic' in source:
        if 'remote' in source:
            return 'remote_mic'
        return 'local_mic'
    return 'sensor'


# ── 主循环 ────────────────────────────────────────────────────────────────────

async def _drain_loop():
    """持续从 event_bus 消费事件，按 priority 分流到两个管道。"""
    ring_size = config.main.get('event', {}).get('llm', {}).get('source_ring_size', 50)
    while True:
        ev = await event_bus.dequeue()
        source = ev.get('source', 'unknown')

        _extract_perf_timestamps(ev)
        priority = _extract_priority(ev)

        # Ring buffer 始终存储（所有事件，供 raw_input_info 查询）
        if source not in _source_ring:
            _source_ring[source] = deque(maxlen=ring_size)
        _source_ring[source].append(ev)

        if priority > 0:
            # ── P>0: 送 main agent ──
            if not _busy:
                await _emit_batch([ev], urgent=True)
            else:
                _priority_pending.append(ev)
                if priority > _current_turn_priority and _cancel_event:
                    _cancel_event.set()
        else:
            # ── P=0: 送 bg buffer ──
            _bg_buffer_add(ev)


async def _bg_trigger_loop():
    """独立节奏：每 interval 把 bg_buffer 送给 bg subagent。"""
    while True:
        interval = config.main.get('event', {}).get('llm', {}).get('trigger_interval_ms', 1000) / 1000.0
        await asyncio.sleep(interval)
        if not _bg_buffer:
            continue
        batch = list(_bg_buffer)
        _bg_buffer.clear()
        await _route_to_bg_subagent(batch)


def start():
    """启动 collector 后台任务。"""
    asyncio.ensure_future(_drain_loop())
    asyncio.ensure_future(_bg_trigger_loop())
    interval = config.main.get('event', {}).get('llm', {}).get('trigger_interval_ms', 1000)
    print(f'[collector] started: dual-queue mode, bg_interval={interval}ms')

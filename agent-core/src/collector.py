"""
collector.py — 信息整理器。

职责：
  - 从 event_bus 持续消费事件，在触发间隔内积累
  - 超过 max_window 的事件 FIFO 丢弃
  - 间隔到达时，按 source 分组摘要并输出（而非原始全量 XML）
  - agent loop 通过 next_trigger() 获取下一批格式化事件
  - priority > 0 的事件立即触发 emit（不等 interval）
  - per-source ring buffer 保留原始事件，供 detailed_info tool 按需查询
  - 支持 cancel_event 信号，允许用户消息中断正在进行的 turn
"""

import asyncio
import datetime
import json as _json
import time
from collections import deque

import config
import event_bus


_buffer: deque = deque()
_priority_buffer: deque = deque()  # 高优先级事件（busy 时暂存）
_output: asyncio.Queue = asyncio.Queue(maxsize=64)
_task: asyncio.Task | None = None
# Per-source throttle: source → timestamp of last accepted event
_last_accepted: dict[str, float] = {}
_THROTTLE_INTERVAL = 1.0  # 每个 source 最多 1 条/秒
# Agent loop busy flag — 忙时不发射 trigger，让事件继续积累
_busy: bool = False

# ── Per-source ring buffer（保留原始事件供 detailed_info tool 查询）────────────
_source_ring: dict[str, deque] = {}

# ── Turn 取消信号（Phase 2：用户消息抢占）────────────────────────────────────────
_cancel_event: asyncio.Event | None = None

# 根据 source 自动赋 priority 的规则（包含匹配）
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
    # 按 source 名匹配
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

    # 新格式：perception 直接上报 spans 数组
    if 'spans' in data:
        ev['_perf_spans'] = data['spans']
        return

    # 旧格式兼容：从 audio_start_ts 等字段构造 spans
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


def set_busy(busy: bool):
    """由 agent loop 调用：标记当前是否正在执行 turn。"""
    global _busy
    _busy = busy
    # turn 结束时，如果有积压的高优先级事件，立即 emit
    if not busy and _priority_buffer:
        asyncio.ensure_future(_emit_priority())


def set_cancel_event(ev: asyncio.Event | None):
    """由 agent loop 调用：注册/清除当前 turn 的取消信号。"""
    global _cancel_event
    _cancel_event = ev


async def _emit_priority():
    """立即 emit 优先级 buffer 中的事件。"""
    if not _priority_buffer:
        return
    batch = list(_priority_buffer)
    _priority_buffer.clear()
    # 也把普通 buffer 一起带上
    if _buffer:
        batch = list(_buffer) + batch
        _buffer.clear()
    await _emit_batch(batch)


async def _emit_batch(batch: list[dict], urgent: bool = False):
    """将一批事件格式化并放入 output。"""
    formatted = _format_batch(batch)
    trigger = {
        'source': 'collector',
        'text': formatted,
        'payload': {'event_count': len(batch), 'sources': [e['source'] for e in batch]},
        'ts': batch[-1]['ts'],
        '_perf_trigger_emit_ts': time.time(),
        '_urgent': urgent,
    }
    # 传递 perception spans
    for ev in reversed(batch):
        if '_perf_spans' in ev:
            trigger['_perf_spans'] = ev['_perf_spans']
            break
    await _output.put(trigger)


def _get_interval_ms() -> int:
    return config.main.get('event', {}).get('llm', {}).get('trigger_interval_ms', 1000)


def _get_max_window() -> int:
    return config.main.get('event', {}).get('llm', {}).get('collector_max_window', 20)


def _infer_channel(ev: dict) -> str:
    """从事件 source 推断渠道标签。"""
    source = ev.get('source', '')
    # 来自 channel 系统的消息（/channel/request/xxx 或 channel:platform:user）
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
    # 远程控制页面文字消息
    if '/remote_control/message' in source:
        return 'remote_web'
    # ASR 事件或麦克风相关 — 根据 source 中是否含 remote 判断
    if 'asr' in source.lower() or '/mic' in source:
        if 'remote' in source:
            return 'remote_mic'
        return 'local_mic'
    # 其他（传感器等）
    return 'sensor'


def _format_batch(events: list[dict]) -> str:
    """将事件列表按 source 分组摘要。

    每个 source 组调用对应 MCP 设备的 output_summary tool（如果存在），
    否则 fallback 到取该 source 最后一条事件的 text。
    高优先级事件（用户消息）始终保留原文。
    """
    # 按 source 分组，保持出现顺序
    groups: dict[str, list[dict]] = {}
    for ev in events:
        source = ev.get('source', 'unknown')
        groups.setdefault(source, []).append(ev)

    parts = []
    for source, evs in groups.items():
        # 高优先级事件保留原始 XML（用户消息不摘要）
        priority = max(_extract_priority(e) for e in evs)
        if priority > 0:
            for ev in evs:
                ts = datetime.datetime.fromtimestamp(ev['ts']).strftime('%Y-%m-%dT%H:%M:%S')
                channel = _infer_channel(ev)
                text = ev.get('text', '')
                parts.append(f'<event source="{source}" channel="{channel}" ts="{ts}">\n{text}\n</event>')
        else:
            # Sensor/低优先级：摘要模式
            summary = _get_source_summary_sync(source, evs)
            ts = datetime.datetime.fromtimestamp(evs[-1]['ts']).strftime('%Y-%m-%dT%H:%M:%S')
            parts.append(f'<source name="{source}" count="{len(evs)}" ts="{ts}">\n{summary}\n</source>')
    return '\n'.join(parts)


def _get_source_summary_sync(source: str, events: list[dict]) -> str:
    """同步获取 source 的摘要。Fallback：取最后一条事件的 text。

    注意：output_summary 的 MCP 调用是异步的，将在 _emit_batch_async 中处理。
    这里先用 fallback 逻辑（最后一条事件），异步版本在 _build_summary_batch 中。
    """
    # 取最后一条事件的 text 作为摘要
    last_text = events[-1].get('text', '')
    if len(events) == 1:
        return last_text
    # 多条时，补充统计信息
    return f'{last_text}\n(共 {len(events)} 条事件，显示最新)'


async def _drain_loop():
    """后台任务：持续从 event_bus 消费事件存入 buffer，per-source 限流。"""
    max_window = _get_max_window()
    ring_size = config.main.get('event', {}).get('llm', {}).get('source_ring_size', 50)
    while True:
        ev = await event_bus.dequeue()
        source = ev.get('source', 'unknown')
        now = ev.get('ts', time.time())

        # 提取性能数据
        _extract_perf_timestamps(ev)

        # 解析优先级
        priority = _extract_priority(ev)

        # ── 存入 per-source ring buffer（保留原始事件供 detailed_info 查询）
        if source not in _source_ring:
            _source_ring[source] = deque(maxlen=ring_size)
        _source_ring[source].append(ev)

        last_ts = _last_accepted.get(source, 0)
        if now - last_ts < _THROTTLE_INTERVAL:
            # 同 source 在 1s 内：替换 buffer 中该 source 的最后一条（保留最新）
            for i in range(len(_buffer) - 1, -1, -1):
                if _buffer[i].get('source') == source:
                    _buffer[i] = ev
                    break
            else:
                _buffer.append(ev)
        else:
            _last_accepted[source] = now
            _buffer.append(ev)

        # FIFO 丢弃超过窗口的旧事件
        while len(_buffer) > max_window:
            _buffer.popleft()

        # 高优先级：立即 emit
        if priority > 0:
            if not _busy:
                # 立即 emit 整个 buffer
                batch = list(_buffer)
                _buffer.clear()
                await _emit_batch(batch, urgent=True)
            else:
                # busy 时暂存到优先级 buffer，并触发取消信号
                _priority_buffer.append(ev)
                if _cancel_event:
                    _cancel_event.set()


async def _trigger_loop():
    """后台任务：每隔 trigger_interval 检查 buffer，有内容则格式化并放入 output。"""
    while True:
        interval = _get_interval_ms() / 1000.0
        await asyncio.sleep(interval)

        if not _buffer:
            continue

        # 防堆积：agent loop 忙时不发射
        if _busy:
            continue

        # 取出当前所有积累的事件
        batch = list(_buffer)
        _buffer.clear()
        await _emit_batch(batch)


def get_source_detail(source: str, limit: int = 20) -> list[dict]:
    """获取指定 source 的原始事件详情（从 ring buffer）。供 detailed_info tool 调用。"""
    ring = _source_ring.get(source)
    if not ring:
        return []
    events = list(ring)[-limit:]
    return events


def get_available_sources() -> list[str]:
    """返回当前有数据的所有 source 名称列表。"""
    return list(_source_ring.keys())


def start():
    """启动 collector 后台任务（在 lifespan 中调用）。"""
    global _task
    loop = asyncio.get_event_loop()
    asyncio.ensure_future(_drain_loop())
    asyncio.ensure_future(_trigger_loop())
    print(f'[collector] started: interval={_get_interval_ms()}ms, max_window={_get_max_window()}')


async def next_trigger() -> dict:
    """阻塞等待下一批格式化事件。返回合成的 trigger event dict。"""
    return await _output.get()


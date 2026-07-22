"""
topic_subscriber.py — 直接用 rclpy 订阅配置的 DDS topic，结果注入 event_bus。

配置（SQLite config table, key='event'）：
  {"subscribe_topics": ["/robot/mic/audio/asr_event", ...]}

每个 topic 收到 String 消息后，enqueue 到 event_bus（source='dds:<topic>'，text=msg.data）。

NOTE: 使用 ros2_bridge._node_main 来创建 subscription，确保和 inspection 用
同一个 DDS participant，避免跨容器 discovery/transport 问题。
"""

import asyncio
import logging
import threading

log = logging.getLogger(__name__)

_HAS_RCLPY = False
try:
    import rclpy
    _HAS_RCLPY = True
except ImportError:
    pass

# Module-level state for dynamic subscribe/unsubscribe
_loop: asyncio.AbstractEventLoop | None = None
_subscriptions: dict = {}  # topic -> subscription object
_lock = threading.Lock()


def start(topics: list[str], loop: asyncio.AbstractEventLoop):
    """订阅给定的 DDS topic 列表，使用 ros2_bridge 的 node。"""
    global _loop
    _loop = loop

    if not _HAS_RCLPY:
        log.warning('[topic_sub] rclpy not available, DDS subscription disabled')
        return

    import ros2_bridge
    if not ros2_bridge._node_main:
        log.warning('[topic_sub] ros2_bridge node not ready, cannot subscribe')
        return

    if not topics:
        log.info('[topic_sub] no topics to subscribe')
        return

    for topic in topics:
        _subscribe_one(topic, loop)

    log.info('[topic_sub] subscribing to %d topics: %s', len(topics), topics)


def _subscribe_one(topic: str, loop: asyncio.AbstractEventLoop):
    """在 ros2_bridge 的 node 上创建一个 subscription。"""
    import event_bus
    import ros2_bridge
    from std_msgs.msg import String
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

    node = ros2_bridge._node_main
    if node is None:
        log.warning('[topic_sub] ros2_bridge node is None')
        return

    def _on_msg(msg, t=topic):
        asyncio.run_coroutine_threadsafe(
            event_bus.enqueue(source=f'dds:{t}', text=msg.data),
            loop,
        )

    sub = node.create_subscription(String, topic, _on_msg, qos)
    # Wake the executor so it picks up the new subscription
    if ros2_bridge._executor:
        ros2_bridge._executor.wake()

    with _lock:
        _subscriptions[topic] = sub
    log.info('[topic_sub] subscribed to %s (via ros2_bridge node)', topic)


def subscribe(topic: str):
    """动态订阅单个 topic（运行时调用）。"""
    if not _HAS_RCLPY:
        log.warning('[topic_sub] cannot subscribe: rclpy not available')
        return False

    import ros2_bridge
    if not ros2_bridge._node_main:
        log.warning('[topic_sub] cannot subscribe: ros2_bridge node not ready')
        return False

    with _lock:
        if topic in _subscriptions:
            log.info('[topic_sub] already subscribed to %s', topic)
            return True

    _subscribe_one(topic, _loop)
    return True


def unsubscribe(topic: str):
    """动态退订单个 topic（运行时调用）。"""
    import ros2_bridge

    with _lock:
        sub = _subscriptions.pop(topic, None)

    if sub is None:
        log.info('[topic_sub] not subscribed to %s, nothing to remove', topic)
        return False

    node = ros2_bridge._node_main
    if node is not None:
        node.destroy_subscription(sub)
    log.info('[topic_sub] dynamically unsubscribed from %s', topic)
    return True

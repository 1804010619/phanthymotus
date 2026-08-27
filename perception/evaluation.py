#!/usr/bin/env python3
import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests


SAMPLE_RATE = 16000
CHUNK_BYTES = int(os.getenv("ASR_CHUNK_BYTES", "1024"))
MCP_TOPIC_MODE = os.getenv("MCP_TOPIC_MODE", "1") == "1"  # Enable MCP topic mode by default

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("asr-eval")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def decode_audio_bytes(case: Dict[str, Any], dataset_dir: Path) -> bytes:
    if case.get("audio_data"):
        return base64.b64decode(case["audio_data"])
    audio_file = case.get("audio_file") or case.get("file") or case.get("path")
    if not audio_file:
        raise ValueError("case missing audio_data/audio_file")
    path = Path(audio_file)
    if not path.is_absolute():
        path = dataset_dir / path
    return path.read_bytes()


def wav_to_pcm16(wav_bytes: bytes) -> bytes:
    import io
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
    except wave.Error:
        return wav_bytes

    if channels != 1 or sample_width != 2 or sample_rate != SAMPLE_RATE:
        try:
            import subprocess
            result = subprocess.run(
                ["ffmpeg", "-i", "pipe:0", "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "pipe:1"],
                input=wav_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
                check=True,
            )
            return result.stdout
        except Exception as exc:
            raise RuntimeError(f"audio format must be 16k mono pcm16 or ffmpeg available: {exc}")
    return frames


def mcp_endpoint(mcp_url: str) -> str:
    url = mcp_url.rstrip("/")
    return url if url.endswith("/mcp") else url + "/mcp"


def mcp_call(mcp_url: str, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "asr-eval", "version": "1.0.0"},
        },
    }
    response = requests.post(mcp_endpoint(mcp_url), json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(data["error"])

    payload = {
        "jsonrpc": "2.0",
        "id": int(time.time() * 1000),
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    response = requests.post(mcp_endpoint(mcp_url), json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    content = data.get("result", {}).get("content", [])
    if content:
        return json.loads(content[0].get("text", "{}"))
    return data.get("result", {})


class RosAsrTopicClient:
    def __init__(self, input_topic: str, output_topic: str, start_time: float):
        import rclpy
        from audio_msgs.msg import AudioChunk
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        if not rclpy.ok():
            rclpy.init(args=None)

        qos_in = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=int(os.getenv("ASR_TOPIC_QOS_DEPTH", "200")),
            durability=DurabilityPolicy.VOLATILE,
        )
        qos_out = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._AudioChunk = AudioChunk
        self._node: Node = rclpy.create_node(f"asr_eval_{int(time.time() * 1000)}")
        self._pub = self._node.create_publisher(AudioChunk, input_topic, qos_in)
        self._input_topic = input_topic
        self._output_topic = output_topic
        self._texts: List[str] = []
        self._timestamps: List[float] = []
        self._lock = threading.Lock()
        self._start_time = start_time

        def on_result(msg):
            try:
                payload = json.loads(msg.data)
                text = payload.get("text", "").strip()
            except Exception:
                text = str(msg.data).strip()
            if text:
                with self._lock:
                    self._texts.append(text)
                    self._timestamps.append(time.time() - self._start_time)

        self._sub = self._node.create_subscription(String, output_topic, on_result, qos_out)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def wait_for_connections_ready(self, timeout: float = 5.0) -> bool:
        """等待 publisher/subscriber 双向连接建立"""
        start = time.time()

        # 等待 input topic 的 subscriber（ASR 端订阅音频）
        while time.time() - start < timeout:
            count = self._pub.get_subscription_count()
            if count > 0:
                log.info(f"[ros] input subscriber connected (count={count})")
                break
            time.sleep(0.2)
        else:
            log.warning(f"[ros] no input subscriber after {timeout}s")

        # 等待 output topic 的 publisher（ASR 端发布结果）
        pub_start = time.time()
        remaining = timeout - (pub_start - start)
        while time.time() - pub_start < max(remaining, 2.0):
            count = self._node.count_publishers(self._output_topic)
            if count > 0:
                log.info(f"[ros] output publisher connected (count={count})")
                return True
            time.sleep(0.2)
        log.warning(f"[ros] no output publisher after {max(remaining, 2.0):.1f}s")
        return False

    def publish(self, chunk: bytes):
        msg = self._AudioChunk()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.format = "audio/pcm-16k"
        msg.data = list(chunk)
        self._pub.publish(msg)

    def results(self) -> tuple[List[str], List[float]]:
        with self._lock:
            return list(self._texts), list(self._timestamps)

    def close(self):
        self._executor.shutdown()
        self._thread.join(timeout=2)
        self._node.destroy_node()


def _pad_chunk(chunk: bytes, chunk_bytes: int) -> bytes:
    if len(chunk) < chunk_bytes:
        return chunk + b"\x00" * (chunk_bytes - len(chunk))
    return chunk


def _get_topics():
    """获取固定的 topic 名称，整个评估实例共享。"""
    input_topic = os.getenv("ASR_INPUT_TOPIC", "/benchmark/mic/audio/default")
    output_topic = f"{input_topic}/asr"
    return input_topic, output_topic


async def recognize_mcp_topic_mode(mcp_url: str, case: Dict[str, Any], dataset_dir: Path, asr_config: Dict[str, Any], chunk_bytes: int, asr_tool: str = "asr") -> Dict[str, Any]:
    """ASR recognition using phanthymotus MCP tools plus ROS2 audio/result topics."""
    case_id = case.get("id", 0)
    t0 = time.time()
    log.info(f"[case {case_id}] 解码音频 → PCM16...")
    pcm = wav_to_pcm16(decode_audio_bytes(case, dataset_dir))
    log.info(f"[case {case_id}] PCM16 大小 {len(pcm)} bytes, 将切分为 {(len(pcm) + chunk_bytes - 1) // chunk_bytes} 个 chunk")

    # 整个评估实例使用固定 topic
    input_topic, output_topic = _get_topics()
    total_chunks = (len(pcm) + chunk_bytes - 1) // chunk_bytes
    start = time.time()
    client: Optional[RosAsrTopicClient] = None

    try:
        log.info(f"[case {case_id}] MCP 配置 ASR instance (tool={asr_tool})...")
        mcp_call(mcp_url, asr_tool, {"action": "config", **asr_config})
        status = mcp_call(mcp_url, asr_tool, {"action": "start", "input_topic": input_topic})
        # ready_wait = float(os.getenv("ASR_READY_WAIT", "2.0"))
        # log.info(f"[case {case_id}] ASR started, waiting {ready_wait}s for engine ready...")
        # await asyncio.sleep(ready_wait)
        # log.info(f"[case {case_id}] ASR engine ready: topic_in={input_topic}, topic_out={output_topic}, state={status.get('state')}")

        client = RosAsrTopicClient(input_topic, output_topic, start)

        # 等待 ROS2 双向连接建立（publisher→subscriber 和 subscriber→publisher）
        ready = client.wait_for_connections_ready(timeout=float(os.getenv("ASR_SUBSCRIBER_WAIT_TIMEOUT", "5")))
        if not ready:
            log.warning(f"[case {case_id}] ROS2 连接未完全建立，仍尝试发送音频")

        for i, offset in enumerate(range(0, len(pcm), chunk_bytes)):
            chunk = _pad_chunk(pcm[offset:offset + chunk_bytes], chunk_bytes)
            client.publish(chunk)
            if (i + 1) % 50 == 0 or (i + 1) == total_chunks:
                log.info(f"[case {case_id}] 已发送 chunk {i + 1}/{total_chunks} ({(i + 1) * 100 // total_chunks}%)")
            # 限速：模拟实时流发送，避免 topic 队列积压
            await asyncio.sleep(chunk_bytes / (SAMPLE_RATE * 2))

        silence_ms = int(os.getenv("ASR_FLUSH_SILENCE_MS", "1500"))
        silence_chunks = max(1, int((silence_ms / 1000) * SAMPLE_RATE * 2 / chunk_bytes))
        log.info(f"[case {case_id}] 发送 {silence_ms}ms 静音触发 VAD 端点...")
        silence = b"\x00" * chunk_bytes
        for _ in range(silence_chunks):
            client.publish(silence)
            await asyncio.sleep(chunk_bytes / (SAMPLE_RATE * 2))
        start = time.time()
        log.info(f"[case {case_id}] 等待 ROS2 ASR 结果 (output_topic={output_topic})...")
        deadline = time.time() + float(os.getenv("ASR_CASE_TIMEOUT", "120"))
        quiet_after = float(os.getenv("ASR_RESULT_QUIET_SECONDS", "30"))
        last_count = 0
        last_change = time.time()
        while time.time() < deadline:
            texts, timestamps = client.results()
            if len(texts) != last_count:
                last_count = len(texts)
                last_change = time.time()
            if texts and time.time() - last_change >= quiet_after:
                break
            await asyncio.sleep(0.2)

        texts, timestamps = client.results()
    finally:
        log.info(f"[case {case_id}] 停止 MCP ASR instance (tool={asr_tool})")
        try:
            stop_result = mcp_call(mcp_url, asr_tool, {"action": "stop"})
            log.info(f"[case {case_id}] ASR stop 结果: {stop_result}")
            await asyncio.sleep(5)  # 等待 ASR 端 ROS2 节点完全销毁
        except Exception as e:
            log.warning(f"[case {case_id}] 停止 ASR instance 失败: {e}")
        if client:
            client.close()

    latency = time.time() - start - 5 - float(os.getenv("ASR_RESULT_QUIET_SECONDS", "30"))
    text = "".join(texts).strip()
    log.info(f"[case {case_id}] 完成: text={text!r}, latency={latency:.3f}s, 耗时={time.time() - t0:.3f}s")
    return {
        "text": text,
        "latency": latency,
        "timestamps": timestamps,
        "responses": [json.dumps({"recognition_results": {"text": text, "final_result": True}}, ensure_ascii=False)],
    }


async def recognize_ws(ws_url: str, case: Dict[str, Any], dataset_dir: Path, asr_config: Dict[str, Any], chunk_bytes: int) -> Dict[str, Any]:
    import websockets
    pcm = wav_to_pcm16(decode_audio_bytes(case, dataset_dir))
    start = time.time()
    texts = []
    timestamps = []
    async with websockets.connect(ws_url, max_size=None, ping_interval=None) as ws:
        await ws.send(json.dumps(asr_config, ensure_ascii=False))
        ready = await ws.recv()
        ready_obj = json.loads(ready)
        if ready_obj.get("type") == "asr_error":
            raise RuntimeError(ready_obj.get("payload", {}).get("error", "asr init failed"))
        for offset in range(0, len(pcm), chunk_bytes):
            chunk = pcm[offset:offset + chunk_bytes]
            if len(chunk) < 1024:
                chunk = chunk + b"\x00" * (1024 - len(chunk))
            await ws.send(chunk)
        await ws.send("flush")
        deadline = time.time() + float(os.getenv("ASR_CASE_TIMEOUT", "120"))
        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2)
            except asyncio.TimeoutError:
                break
            payload = json.loads(msg)
            if payload.get("type") == "asr_result":
                texts.append(payload.get("payload", {}).get("text", ""))
                timestamps.append(time.time() - start)
            elif payload.get("type") == "asr_error":
                raise RuntimeError(payload.get("payload", {}).get("error", "asr error"))
    latency = time.time() - start
    text = "".join(texts).strip()
    if not timestamps:
        timestamps = []
    return {
        "text": text,
        "latency": latency,
        "timestamps": timestamps,
        "responses": [json.dumps({"recognition_results": {"text": text, "final_result": True}}, ensure_ascii=False)],
    }


def case_reference(case: Dict[str, Any]) -> str:
    answer = case.get("answer") or case.get("voice") or case.get("text") or case.get("reference") or ""
    if isinstance(answer, list):
        return "".join(str(item.get("text", item)) for item in answer)
    return str(answer)


async def run_cases(args):
    dataset_path = Path(args.dataset)
    dataset_dir = dataset_path.parent
    log.info(f"[启动] 加载数据集: {dataset_path}")
    cases = read_jsonl(dataset_path)
    if args.limit:
        cases = cases[:args.limit]
    log.info(f"[启动] 共 {len(cases)} 条 case, chunk_bytes={args.chunk_bytes}, MCP_TOPIC_MODE={MCP_TOPIC_MODE}")

    mcp_url = args.mcp_url or os.getenv("MCP_URL", "http://127.0.0.1:15720")
    ws_url = args.ws_url or os.getenv("ASR_WS_URL", "ws://127.0.0.1:15721")
    if not ws_url.endswith("/ws/asr"):
        ws_url = ws_url.rstrip("/") + "/ws/asr"
    log.info(f"[启动] MCP_URL={mcp_url}, ASR_WS_URL={ws_url}")

    asr_config = {
        "trigger_mode": "vad",
        "language": os.getenv("ASR_LANGUAGE", "zh-CN"),
    }
    asr_tool = os.getenv("ASR_PLUGIN", "asr")
    log.info(f"[启动] ASR language={asr_config['language']}, tool={asr_tool}")

    results = []
    bad_cases = []
    total_start = time.time()
    for idx, case in enumerate(cases):
        item = {
            "idx": idx,
            "id": case.get("id", idx),
            "lang": case.get("lang", "zh"),
            "reference": case_reference(case),
            "success": False,
        }
        log.info(f"========== case {idx + 1}/{len(cases)} (id={item['id']}, lang={item['lang']}) ==========")
        try:
            if MCP_TOPIC_MODE:
                # Use MCP topic mode
                recog = await recognize_mcp_topic_mode(mcp_url, case, dataset_dir, asr_config, args.chunk_bytes, asr_tool)
            else:
                # Use WebSocket mode (original)
                recog = await recognize_ws(ws_url, case, dataset_dir, asr_config, args.chunk_bytes)
            item.update(recog)
            item["success"] = True
            log.info(f"[case {item['id']}] 成功: text={item.get('text', '')!r}")
        except Exception as exc:
            item["error_message"] = str(exc)
            item["text"] = ""
            item["latency"] = 0
            item["timestamps"] = [0]
            item["responses"] = [json.dumps({"recognition_results": {"text": "", "final_result": True}}, ensure_ascii=False)]
            bad_cases.append(item)
            log.error(f"[case {item['id']}] 失败: {exc}")
        results.append(item)
        write_json(Path(args.output) / "partial_results.json", {"cases": results})
        elapsed = time.time() - total_start
        avg = elapsed / (idx + 1)
        remaining = avg * (len(cases) - idx - 1)
        log.info(f"[进度] {idx + 1}/{len(cases)} 完成, 已用 {elapsed:.1f}s, 预计剩余 {remaining:.1f}s (avg {avg:.2f}s/case)")

    output_dir = Path(args.output)
    write_json(output_dir / "results.json", {"cases": results})
    write_json(output_dir / "detailed_cases.json", results)
    write_json(output_dir / "bad_cases.json", {"bad_cases": bad_cases, "total_bad_count": len(bad_cases)})
    log.info(f"[完成] 共 {len(results)} 条, 成功 {len(results) - len(bad_cases)}, 失败 {len(bad_cases)}, 总耗时 {time.time() - total_start:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.getenv("DATASET_PATH", "dataset/dataset.json"))
    parser.add_argument("--output", default=os.getenv("OUTPUT_DIR", "output"))
    parser.add_argument("--mcp-url", default=os.getenv("MCP_URL"))
    parser.add_argument("--ws-url", default=os.getenv("ASR_WS_URL"))
    parser.add_argument("--limit", type=int, default=int(os.getenv("DATASET_LIMIT", "0") or 0))
    parser.add_argument("--chunk-bytes", type=int, default=CHUNK_BYTES)
    parser.add_argument("--mcp-topic-mode", type=int, default=1, help="Use MCP topic mode (1) or WS mode (0)")
    args = parser.parse_args()

    # Override MCP_TOPIC_MODE based on CLI argument
    global MCP_TOPIC_MODE
    MCP_TOPIC_MODE = args.mcp_topic_mode == 1

    asyncio.run(run_cases(args))


if __name__ == "__main__":
    main()

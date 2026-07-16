"""Private streaming service for the JetPack 6 VITS2 TensorRT runtime."""

from __future__ import annotations

import logging
import os
import re
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from backends.trt_tts_engine import TensorRTTTSEngine


logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_BYTES = 3200
MAX_CHUNK_TOKENS = int(os.getenv("MIX_VITS_MAX_TEXT_TOKENS", "64"))
MODEL_CONFIG = os.getenv("MIX_VITS_CONFIG_PATH", "/models/vits2-mix/config.json")
ENGINE_DIR = os.getenv("MIX_VITS_TRT_ENGINE_DIR", "/models/vits2-mix/engines")

_engine = None
_lock = threading.Lock()
_ready = False


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    speed: float = Field(1.0, gt=0.0, le=4.0)


def _language_kind(char: str) -> str | None:
    if "\u4e00" <= char <= "\u9fff":
        return "ZH"
    if char.isascii() and char.isalnum():
        return "EN"
    return None


def _preferred_split(text: str) -> int:
    midpoint = len(text) // 2
    boundaries = []
    previous_kind = None
    for index, char in enumerate(text):
        kind = _language_kind(char)
        if kind is None:
            continue
        if previous_kind is not None and kind != previous_kind:
            boundaries.append(index)
        previous_kind = kind
    usable = [index for index in boundaries if 1 < index < len(text) - 1]
    if usable:
        return min(usable, key=lambda index: abs(index - midpoint))
    return max(1, midpoint)


def _iter_unit_chunks(text: str):
    text_ids = _engine._get_text_ids(text)
    if len(text_ids[0]) <= MAX_CHUNK_TOKENS:
        yield text, text_ids
        return
    if len(text) <= 1:
        raise ValueError("Unable to split text within TensorRT profile")
    split_at = _preferred_split(text)
    left, right = text[:split_at].strip(), text[split_at:].strip()
    if not left or not right:
        raise ValueError("Unable to split text within TensorRT profile")
    yield from _iter_unit_chunks(left)
    yield from _iter_unit_chunks(right)


def _iter_text_chunks(text: str):
    """Lazily split by punctuation, language boundary, then midpoint."""
    units = re.findall(r".*?[。！？!?；;，,：:\n]+|.+$", text, flags=re.DOTALL)
    for unit in units:
        if unit.strip():
            yield from _iter_unit_chunks(unit.strip())


def _stream_pcm(text: str, speed: float):
    silence = b"\x00\x00" * (SAMPLE_RATE // 10)
    with _lock:
        for chunk_index, (chunk, text_ids) in enumerate(_iter_text_chunks(text)):
            token_count = len(text_ids[0])
            log.info(
                "text redacted: chars=%d chunk=%d tokens=%d",
                len(text), chunk_index, token_count,
            )
            if chunk_index:
                yield silence
            pcm = _engine.synthesize(
                chunk, text_ids=text_ids, length_scale=1.0 / speed
            )
            for offset in range(0, len(pcm), CHUNK_BYTES):
                yield pcm[offset:offset + CHUNK_BYTES]


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _engine, _ready
    _ready = False
    _engine = TensorRTTTSEngine(MODEL_CONFIG, ENGINE_DIR)
    # Warm the same streaming and profile-splitting path used by requests.
    warmup_bytes = 0
    for text in (
        "你好。",
        "周末我拍了一张selfie发给朋友。",
        "Lucy今天去公园散步并喝coffee，David开会前仔细检查PPT。",
    ):
        case_bytes = 0
        for pcm in _stream_pcm(text, speed=1.0):
            case_bytes += len(pcm)
        if not case_bytes:
            raise RuntimeError("TensorRT warmup produced no audio")
        warmup_bytes += case_bytes
    _ready = True
    log.info("VITS2 TensorRT backend ready: warmup_bytes=%d", warmup_bytes)
    try:
        yield
    finally:
        _ready = False
        _engine = None


app = FastAPI(title="Private VITS2 TensorRT Backend", lifespan=lifespan)


@app.get("/ready")
def ready():
    if not _ready:
        return PlainTextResponse("False", status_code=503)
    return PlainTextResponse("True")


@app.get("/health/detail")
def health_detail():
    runtime = getattr(_engine, "runtime_info", {}) if _engine else {}
    return JSONResponse({"ready": _ready, "runtime": runtime})


@app.post("/synthesize/stream")
def synthesize_stream(request: SynthesizeRequest):
    if not _ready:
        return PlainTextResponse("Service not ready", status_code=503)
    text = re.sub(r"\s+", " ", request.text).strip()
    if not text:
        return PlainTextResponse("No text found", status_code=400)
    return StreamingResponse(
        _stream_pcm(text, request.speed),
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(SAMPLE_RATE), "X-Accel-Buffering": "no"},
    )

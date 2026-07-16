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


def _token_count(text: str) -> int:
    return len(_engine._get_text_ids(text)[0])


def _iter_text_chunks(text: str):
    units = re.findall(r".*?[。！？!?；;\n]+|.+$", text, flags=re.DOTALL)
    current = ""
    for unit in units:
        candidate = current + unit
        if _token_count(candidate) <= MAX_CHUNK_TOKENS:
            current = candidate
            continue
        if current.strip():
            yield current.strip()
            current = ""
        remainder = unit.strip()
        while remainder:
            if _token_count(remainder) <= MAX_CHUNK_TOKENS:
                current = remainder
                break
            low, high = 1, len(remainder)
            while low < high:
                middle = (low + high + 1) // 2
                if _token_count(remainder[:middle]) <= MAX_CHUNK_TOKENS:
                    low = middle
                else:
                    high = middle - 1
            if low < 1:
                raise ValueError("Unable to split text within TensorRT profile")
            yield remainder[:low].strip()
            remainder = remainder[low:].strip()
    if current.strip():
        yield current.strip()


def _stream_pcm(text: str, speed: float):
    silence = b"\x00\x00" * (SAMPLE_RATE // 10)
    with _lock:
        for chunk_index, chunk in enumerate(_iter_text_chunks(text)):
            token_count = _token_count(chunk)
            if token_count > MAX_CHUNK_TOKENS:
                raise ValueError("Text segmentation exceeded TensorRT profile")
            log.info(
                "text redacted: chars=%d chunk=%d tokens=%d",
                len(text), chunk_index, token_count,
            )
            if chunk_index:
                yield silence
            pcm = _engine.synthesize(chunk, length_scale=1.0 / speed)
            for offset in range(0, len(pcm), CHUNK_BYTES):
                yield pcm[offset:offset + CHUNK_BYTES]


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _engine, _ready
    _ready = False
    _engine = TensorRTTTSEngine(MODEL_CONFIG, ENGINE_DIR)
    # Warm Chinese and mixed-language paths before advertising ready.
    warmup_bytes = 0
    with _lock:
        for text in (
            "你好。",
            "Lucy今天去公园散步并喝coffee。",
            "David开会前仔细检查PPT。",
        ):
            pcm = _engine.synthesize(text)
            if not pcm:
                raise RuntimeError("TensorRT warmup produced no audio")
            warmup_bytes += len(pcm)
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

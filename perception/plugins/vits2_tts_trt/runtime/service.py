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


def _with_blanks(values):
    result = [0]
    for value in values:
        result.extend((value, 0))
    return tuple(result)


def _split_text_ids(text_ids):
    token_count = len(text_ids[0])
    if token_count <= MAX_CHUNK_TOKENS:
        yield text_ids
        return

    if _engine.add_blank:
        # _get_text_ids inserts zero between every phone. Split the original
        # phone streams and restore boundary blanks for every TensorRT call.
        # Prefer a nearby ZH/EN transition over cutting inside a language span.
        streams = tuple(values[1::2] for values in text_ids)
        phones_per_chunk = (MAX_CHUNK_TOKENS - 1) // 2
        language_ids = streams[2]
        offset = 0
        while offset < len(streams[0]):
            limit = min(offset + phones_per_chunk, len(streams[0]))
            if limit == len(streams[0]):
                end = limit
            else:
                minimum_boundary = offset + max(8, phones_per_chunk // 2)
                boundaries = [
                    index
                    for index in range(minimum_boundary, limit + 1)
                    if language_ids[index - 1] != language_ids[index]
                ]
                end = boundaries[-1] if boundaries else limit
            yield tuple(
                _with_blanks(values[offset:end])
                for values in streams
            )
            offset = end
        return

    for offset in range(0, token_count, MAX_CHUNK_TOKENS):
        yield tuple(
            tuple(values[offset:offset + MAX_CHUNK_TOKENS])
            for values in text_ids
        )


def _iter_text_id_chunks(text: str):
    """Normalize one sentence at a time and lazily yield safe ID chunks."""
    units = re.findall(r".*?[。！？!?；;\n]+|.+$", text, flags=re.DOTALL)
    for unit in units:
        if unit.strip():
            yield from _split_text_ids(_engine._get_text_ids(unit.strip()))


def _stream_pcm(text: str, speed: float):
    silence = b"\x00\x00" * (SAMPLE_RATE // 10)
    with _lock:
        for chunk_index, text_ids in enumerate(_iter_text_id_chunks(text)):
            token_count = len(text_ids[0])
            log.info(
                "text redacted: chars=%d chunk=%d tokens=%d",
                len(text), chunk_index, token_count,
            )
            if chunk_index:
                yield silence
            pcm = _engine.synthesize(
                "", text_ids=text_ids, length_scale=1.0 / speed
            )
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

"""HTTP adapter for the isolated JetPack 6 TensorRT TTS runtime."""

from __future__ import annotations

import json
import logging
import os
import struct
from abc import ABC, abstractmethod
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SAMPLE_RATE = 16000
CHUNK_BYTES = 3200
log = logging.getLogger(__name__)


class TTSAdapter(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        raise NotImplementedError

    def synthesize_stream(self, text: str):
        yield self.synthesize(text)

    def warmup(self) -> int:
        return 0


class Vits2TensorRTAdapter(TTSAdapter):
    def __init__(
        self,
        backend_url: str,
        speed: float = 1.0,
        timeout: float = 295.0,
    ):
        if speed <= 0:
            raise ValueError("TTS speed must be greater than zero")
        self._backend_url = backend_url.rstrip("/")
        self._speed = float(speed)
        self._timeout = float(timeout)

    def _request(self, path: str, payload: dict | None = None):
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(f"{self._backend_url}{path}", data=data, headers=headers)
        try:
            return urlopen(request, timeout=self._timeout)
        except HTTPError as exc:
            raise RuntimeError(
                f"TensorRT backend rejected request: HTTP {exc.code}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"TensorRT backend unavailable: {exc.reason}") from exc

    def warmup(self) -> int:
        with self._request("/ready") as response:
            if response.status != 200 or response.read().strip() != b"True":
                raise RuntimeError("TensorRT backend is not ready")
        return 0

    def synthesize(self, text: str) -> bytes:
        return b"".join(self.synthesize_stream(text))

    def synthesize_stream(self, text: str):
        if not text.strip():
            raise ValueError("TTS text must not be empty")
        payload = {"text": text, "speed": self._speed}
        with self._request("/synthesize/stream", payload) as response:
            sample_rate = int(response.headers.get("X-Sample-Rate", SAMPLE_RATE))
            if sample_rate != SAMPLE_RATE:
                raise RuntimeError(
                    f"TensorRT backend sample rate must be {SAMPLE_RATE}, got {sample_rate}"
                )
            pending = bytearray()
            while True:
                block = response.read(CHUNK_BYTES)
                if not block:
                    break
                pending.extend(block)
                while len(pending) >= CHUNK_BYTES:
                    yield bytes(pending[:CHUNK_BYTES])
                    del pending[:CHUNK_BYTES]
            if pending:
                if len(pending) % struct.calcsize("h"):
                    raise RuntimeError("TensorRT backend returned misaligned PCM")
                yield bytes(pending)


def build_adapter(cfg: dict) -> TTSAdapter:
    speaker_id = int(cfg.get("speaker_id", 0))
    if speaker_id != 0:
        raise ValueError("The VITS2 model supports only speaker_id=0")
    return Vits2TensorRTAdapter(
        backend_url=os.getenv(
            "VITS2_TRT_BACKEND_URL",
            cfg.get("backend_url", "http://127.0.0.1:18080"),
        ),
        speed=float(cfg.get("speed", 1.0)),
        timeout=float(cfg.get("request_timeout", 295.0)),
    )

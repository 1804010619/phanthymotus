"""Adapter between the shared ROS2 TTS plugin and the VITS2 CPU engine."""

from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from .onnx_cpu_engine import OnnxCpuEngine


SAMPLE_RATE = 16000
CHUNK_BYTES = 3200


class TTSAdapter(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        raise NotImplementedError

    def synthesize_stream(self, text: str):
        yield self.synthesize(text)


class Vits2OnnxCpuAdapter(TTSAdapter):
    def __init__(self, model_dir: str, speed: float = 1.0, num_threads: int = 6):
        if speed <= 0:
            raise ValueError("TTS speed must be greater than zero")

        root = Path(model_dir)
        package_dir = Path(__file__).resolve().parent
        os.environ["NLTK_DATA"] = str(root / "nltk_data")
        os.environ["EN_TN_CACHE_DIR"] = str(root / "tn_cache")
        os.environ["TN_CACHE_DIR"] = str(root / "tn_cache")
        os.environ["VITS2_FRONTEND_DATA_DIR"] = str(root / "frontend_data")

        config_path = package_dir / "config.json"
        self._engine = OnnxCpuEngine(
            config_path=config_path,
            model_dir=root / "onnx",
            num_threads=num_threads,
        )
        if self._engine.sample_rate != SAMPLE_RATE:
            raise RuntimeError(
                f"VITS2 sample rate must be {SAMPLE_RATE}, got {self._engine.sample_rate}"
            )
        self._length_scale = 1.0 / speed
        self._lock = threading.Lock()

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            raise ValueError("TTS text must not be empty")
        with self._lock:
            return self._engine.synthesize(text, length_scale=self._length_scale)

    def synthesize_stream(self, text: str):
        pcm = self.synthesize(text)
        for offset in range(0, len(pcm), CHUNK_BYTES):
            yield pcm[offset:offset + CHUNK_BYTES]


def build_adapter(cfg: dict) -> TTSAdapter:
    speaker_id = int(cfg.get("speaker_id", 0))
    if speaker_id != 0:
        raise ValueError("The VITS2 model supports only speaker_id=0")
    return Vits2OnnxCpuAdapter(
        model_dir=cfg.get("vits2_model_dir", "/models/vits2-mix"),
        speed=float(cfg.get("speed", 1.0)),
        num_threads=max(1, int(cfg.get("vits2_num_threads", 6))),
    )

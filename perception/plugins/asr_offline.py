"""Official Paraformer adapter for the default ASR plugin."""

from __future__ import annotations

import io
import json
import logging
import struct
import threading
import wave
from pathlib import Path


SAMPLE_RATE = 16000
log = logging.getLogger(__name__)


def _resolve_model_file(model_root: Path, configured_model: str) -> Path:
    if configured_model:
        configured_path = Path(configured_model)
        candidate = (
            configured_path
            if configured_path.is_absolute()
            else model_root / configured_path
        )
        if candidate.is_file():
            return candidate.resolve()

    preferred = model_root / "model.int8.onnx"
    if preferred.is_file():
        return preferred.resolve()

    onnx_files = sorted(model_root.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No ONNX model found in {model_root}")
    return onnx_files[0].resolve()


def _create_sherpa_recognizer(
    model_path: str,
    config: dict,
    sample_rate: int = SAMPLE_RATE,
    bits_per_sample: int = 16,
):
    del bits_per_sample
    import sherpa_onnx

    root = Path(model_path)
    model_root = root if root.is_dir() else root.parent
    configured_model = config.get("model") or config.get("model_path") or ""
    model_file = _resolve_model_file(model_root, configured_model or str(root))

    configured_tokens = Path(config.get("tokens", "tokens.txt"))
    tokens = (
        configured_tokens
        if configured_tokens.is_absolute()
        else model_root / configured_tokens
    ).resolve()
    if not tokens.is_file():
        raise FileNotFoundError(f"Token file not found: {tokens}")

    recognizer_config = config.get("recognizerConfig", {})
    kwargs = {
        "paraformer": str(model_file),
        "tokens": str(tokens),
        "num_threads": int(config.get("numThreads", 1)),
        "sample_rate": int(sample_rate),
        "feature_dim": int(config.get("featureConfig", {}).get("featureDim", 80)),
        "decoding_method": recognizer_config.get(
            "decodingMethod", "greedy_search"
        ),
        "debug": bool(config.get("debug", False)),
        "provider": config.get("provider", "cpu"),
        "rule_fsts": recognizer_config.get("ruleFsts", ""),
        "rule_fars": recognizer_config.get("ruleFars", ""),
    }
    try:
        return sherpa_onnx.OfflineRecognizer.from_paraformer(**kwargs)
    except TypeError:
        kwargs.pop("rule_fsts")
        kwargs.pop("rule_fars")
        return sherpa_onnx.OfflineRecognizer.from_paraformer(**kwargs)


class OfflineASRAdapter:
    """Decode complete WAV utterances with a cached offline recognizer."""

    _instances: dict[tuple[str, int, str], "OfflineASRAdapter"] = {}
    _lock = threading.Lock()

    def __init__(
        self,
        model_path: str,
        config: dict | None = None,
        num_threads: int = 1,
        provider: str = "cpu",
    ):
        model_root = Path(model_path)
        loaded_config: dict = {}
        config_path = model_root / "config.json"
        if config is not None:
            loaded_config = dict(config)
        elif config_path.is_file():
            raw_config = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(raw_config.get("sherpa"), dict):
                loaded_config = dict(raw_config["sherpa"])
                loaded_config.setdefault("model", raw_config.get("model_path", ""))
            else:
                loaded_config = dict(raw_config)

        loaded_config["numThreads"] = int(num_threads)
        loaded_config["provider"] = provider
        self._recognizer = _create_sherpa_recognizer(
            model_path, loaded_config, sample_rate=SAMPLE_RATE
        )

    @classmethod
    def get_instance(
        cls,
        model_path: str,
        config: dict | None = None,
        num_threads: int = 1,
        provider: str = "cpu",
    ) -> "OfflineASRAdapter":
        key = (str(Path(model_path).resolve()), int(num_threads), provider)
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = cls(
                    model_path,
                    config=config,
                    num_threads=num_threads,
                    provider=provider,
                )
            return cls._instances[key]

    @classmethod
    def clear_cache(cls) -> None:
        with cls._lock:
            cls._instances.clear()

    def transcribe(self, wav_bytes: bytes, language: str = "zh-CN") -> str:
        del language
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
                raise ValueError("Offline ASR expects mono 16-bit PCM WAV")
            sample_rate = wav_file.getframerate()
            raw_pcm = wav_file.readframes(wav_file.getnframes())

        sample_count = len(raw_pcm) // 2
        samples = [
            sample / 32768.0
            for sample in struct.unpack(f"<{sample_count}h", raw_pcm)
        ]
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self._recognizer.decode_stream(stream)

        result = getattr(stream, "result", None)
        if result is None and hasattr(self._recognizer, "get_result"):
            result = self._recognizer.get_result(stream)
        text = getattr(result, "text", result or "")
        return str(text).strip()

"""
utils/onnx_provider.py — Resolve the sherpa-onnx execution provider per model.

Two independent facts decide this, and both were measured on orin5 (Orin NX,
JetPack 5.11, CUDA 11.4, 6 CPU cores) against sherpa-onnx 1.13.6+cuda:

1. **Not every image can use the GPU at all.** jetson jp5.11 installs a
   CUDA-enabled wheel (onnxruntime-gpu 1.16.0); jp6.1 is CUDA 12.6 and that
   wheel's `libcudart.so.11.0` does not satisfy it, so it keeps the CPU wheel
   from PyPI, as do x86 dev hosts. See `Dockerfile.jetson`.

2. **The GPU only wins on fp32 weights.** ONNX Runtime's CUDA execution provider
   has no kernels for the quantised ops in an int8 model, so it partitions the
   graph and falls back to CPU node by node, adding a H2D/D2H copy at every
   boundary. Measured, same audio, steady-state median:

   | model                                   | CPU (2 thr) | CUDA    |         |
   |-----------------------------------------|-------------|---------|---------|
   | streaming paraformer `encoder.int8.onnx`| 3412 ms     | 10804 ms| 0.32x   |
   | offline SenseVoice `model.int8.onnx`    | 2021 ms     | 3787 ms | 0.53x   |
   | Matcha TTS fp32 (`model-steps-3.onnx`)  | 1751 ms     |  400 ms | 4.38x   |

So `hw_provider: auto` means "cuda when the wheel supports it *and* this model's
weights are fp32". Every ASR and KWS bundle deployed today ships int8 weights, so
auto keeps them on CPU; Matcha TTS is fp32 and gets the GPU.

Caveat worth knowing: the int8 rule is what the measurements support. A
*streaming* fp32 model is untested — the streaming penalty above (0.32x vs 0.53x
for the same dtype) shows sherpa's per-chunk decode adds its own GPU overhead, so
a streaming fp32 model might not reach anything like Matcha's 4.4x. Pin
`hw_provider: cpu` explicitly if you deploy one and it disappoints.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)

VALID_PROVIDERS = ("auto", "cuda", "cpu")


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    """True when the installed sherpa_onnx wheel bundles the CUDA provider.

    The marker is `sherpa_onnx/lib/libonnxruntime_providers_cuda.so`, which only
    a `-DSHERPA_ONNX_ENABLE_GPU=ON` build ships. This deliberately does not probe
    the driver or create a session: on Jetson the wheel is built for the same L4T
    release it is deployed on, and a probe would cost a full ONNX Runtime session
    at import time.
    """
    try:
        import sherpa_onnx
    except ImportError:
        return False
    marker = Path(sherpa_onnx.__file__).resolve().parent / "lib" / "libonnxruntime_providers_cuda.so"
    return marker.is_file()


def is_quantised(model_paths: Iterable[str]) -> bool:
    """True when any of these model files carries int8 weights.

    Filename, not graph inspection: `.int8.onnx` is the naming every sherpa-onnx
    bundle uses, and it is already what `asr.py` and `x_asr.py` match on to
    *prefer* the quantised file. Parsing the ONNX protobuf for QuantizeLinear
    nodes would be more precise but costs a read of a 165 MB file at startup.
    """
    return any("int8" in Path(p).name.lower() for p in model_paths if p)


def resolve_provider(value: str | None, model_paths: Iterable[str] = ()) -> str:
    """Map a configured `hw_provider` to a provider sherpa-onnx will accept.

    An explicit `cuda`/`cpu` is passed through unchanged — if someone pins cuda
    on an int8 model or a CPU wheel, that is a deliberate choice (or a
    misconfigured deployment that should stay visible), and silently rewriting it
    would hide either one.

    `model_paths` are the weight files the caller is about to load. Omitting them
    means "assume fp32", so a caller that forgets them errs toward the GPU rather
    than silently pinning CPU.
    """
    provider = (value or "auto").strip().lower()
    if provider != "auto":
        if provider not in VALID_PROVIDERS:
            log.warning("[onnx_provider] unknown hw_provider %r, using cpu", value)
            return "cpu"
        return provider

    paths = [p for p in model_paths if p]
    if not cuda_available():
        reason = "the installed sherpa_onnx wheel has no CUDA provider"
    elif is_quantised(paths):
        reason = ("int8 weights — ONNX Runtime's CUDA provider falls back to CPU "
                  "per node on quantised ops, which measured 2-3x slower")
    else:
        log.info("[onnx_provider] hw_provider=auto -> cuda (fp32 weights, CUDA wheel installed)")
        return "cuda"

    log.info("[onnx_provider] hw_provider=auto -> cpu (%s)", reason)
    return "cpu"

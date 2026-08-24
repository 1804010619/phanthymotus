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
   boundary. Measured, same audio, steady-state median, every provider swept
   across 1/2/4 `num_threads` (the ratio changes a lot with thread count, so a
   single-thread-count comparison is misleading):

   | model                                    | dtype       | CPU t=2 | CUDA t=2 | ratio | best CPU    | best CUDA  | ratio |
   |------------------------------------------|-------------|---------|----------|-------|-------------|------------|-------|
   | streaming paraformer `encoder.int8.onnx` | int8        | 3383 ms | 11125 ms | 0.30x | 3097 (t=4)  | 10524 (t=4)| 0.29x |
   | X-ASR offline transducer, beam search    | int8 + fp32 | 2832 ms |  5707 ms | 0.50x | 2503 (t=4)  |  5707 (t=2)| 0.44x |
   | offline SenseVoice `model.int8.onnx`     | int8        | 2022 ms |  2341 ms | 0.86x | 1354 (t=4)  |  1694 (t=4)| 0.80x |
   | Matcha TTS `model-steps-3.onnx` + vocos  | fp32        | 1784 ms |   416 ms | 4.28x | 1175 (t=4)  |   416 (t=1)| 2.83x |

   X-ASR is the case that justifies `is_quantised` treating *any* int8 file as
   disqualifying: its bundle is mixed precision (int8 encoder and joiner, fp32
   decoder) and CUDA still lost by 2x.

   The per-node fallback is not just an inference from that contrast: on the
   streaming int8 model the *CUDA* path speeds up 1.7x when given more **CPU**
   threads (17861 -> 11125 -> 10524 ms for 1/2/4). Thread count would barely
   matter if the graph were executing on the GPU. GPU contention is not the
   explanation either — the idle GR3D baseline was 0%, and GR3D sat at 88-92%
   during the CUDA runs, so the work does reach the GPU, just slowly.

So `hw_provider: auto` means "cuda when the wheel supports it *and* this model's
weights are fp32". Every ASR and KWS bundle deployed today ships int8 weights, so
auto keeps them on CPU; Matcha TTS is fp32 and gets the GPU.

Caveat worth knowing: the int8 rule is what the measurements support, and only for
these shapes. A *streaming* fp32 model is untested, and the streaming-vs-offline
gap at the same dtype above is large (0.30x vs 0.86x) — sherpa's per-chunk decode
carries substantial GPU overhead of its own, so a streaming fp32 model may land
nowhere near Matcha's speed-up. Pin `hw_provider: cpu` explicitly if you deploy
one and it disappoints. Note also that the deployed `num_threads` is 2; at 4
threads the CPU closes much of Matcha's gap (2.83x rather than 4.28x).
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

    *Any*, not all: X-ASR ships an int8 encoder and joiner beside an fp32 decoder,
    and CUDA still lost by 2x on it, so one quantised file is enough to disqualify
    the bundle.

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
                  "per node on quantised ops, measured up to 3x slower")
    else:
        log.info("[onnx_provider] hw_provider=auto -> cuda (fp32 weights, CUDA wheel installed)")
        return "cuda"

    log.info("[onnx_provider] hw_provider=auto -> cpu (%s)", reason)
    return "cpu"

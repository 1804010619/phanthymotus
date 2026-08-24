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
   boundary. Measured on an idle box (services stopped, idle GR3D verified 0%),
   same audio, steady-state median, every provider swept across 1/2/4
   `num_threads` (the ratio changes a lot with thread count, so a
   single-thread-count comparison is misleading):

   | model                                    | dtype       | CPU t=2 | CUDA t=2 | ratio  |
   |------------------------------------------|-------------|---------|----------|--------|
   | streaming paraformer `encoder.int8.onnx` | int8        | 3383 ms | 11125 ms |  0.30x |
   | X-ASR offline transducer, beam search    | int8 + fp32 | 2645 ms |  3294 ms |  0.80x |
   | offline SenseVoice `model.int8.onnx`     | int8        | 1996 ms |  2753 ms |  0.73x |
   | offline SenseVoice `model.onnx`          | fp32        | 4792 ms |   416 ms | 11.52x |
   | offline SenseVoice `model.fp16.onnx`     | fp16        | 8117 ms |   344 ms | 23.60x |
   | Matcha TTS `model-steps-3.onnx` + vocos  | fp32        | 1784 ms |   416 ms |  4.28x |

   The three SenseVoice rows are the same model in three dtypes, so this is a
   dtype result and not a per-model coincidence. X-ASR is why `is_quantised`
   disqualifies a bundle on *any* int8 file: it is mixed precision (int8 encoder
   and joiner, fp32 decoder) and CUDA still lost.

   Mechanism confirmed two ways. fp32-CUDA and fp16-CUDA are completely
   insensitive to thread count (417/416/415 and 343/344/346 ms at 1/2/4) with GR3D
   pinned at 95-97% — the graph really is on the GPU. int8-CUDA instead scales with
   *CPU* threads (3793 -> 2753 -> 1905 for SenseVoice, 17861 -> 11125 -> 10524 for
   streaming paraformer), which only makes sense if much of it is not.

   int8 on CUDA is also numerically noisier, because every int8<->fp32 partition
   boundary requantises. Same model and audio at num_threads=2, CPU vs CUDA:
   SenseVoice int8 differed on 3 of 4 clips (including `不然` -> `主然`, a real
   word error); SenseVoice fp32 and fp16 differed on 0 of 4, and fp16 matched fp32
   exactly. So keeping int8 on CPU buys output stability as well as speed.

So `hw_provider: auto` means "cuda when the wheel supports it *and* this model's
weights are fp32". Every ASR and KWS bundle deployed today ships int8 weights, so
auto keeps them on CPU; Matcha TTS is fp32 and gets the GPU.

Putting ASR on the GPU is a *model* change, not a config one: ship non-int8 weights
and auto routes them to CUDA by itself. Against the deployed int8-on-CPU (1996 ms),
fp32-CUDA is 4.80x and fp16-CUDA is 5.81x at half fp32's size — the trade-offs, the
fp16 conversion recipe, and what is still unmeasured (streaming) are written up in
perception/README.md.

KNOWN GAP: `is_quantised` only recognises int8. An fp16 model resolves to `cuda`,
which is right on a GPU wheel — but it would also resolve to `cuda` on a CPU-only
wheel, and fp16 on CPU is catastrophic (13051 ms vs int8's 3292 at one thread,
because ONNX Runtime has no fp16 CPU kernels and casts everything). Nothing ships
fp16 today; if that changes, this needs a guard that refuses fp16 without CUDA.

`int16` is not a middle ground worth trying: ONNX Runtime's int16 quantisation is
newer than int8, the CUDA provider has no kernels for it either, and the CPU side
lacks the dot-product paths that make int8 fast.
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
    and CUDA still lost on it, so one quantised file is enough to disqualify the
    bundle.

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
                  "per node on quantised ops; measured 1.25x-3.3x slower")
    else:
        log.info("[onnx_provider] hw_provider=auto -> cuda (fp32 weights, CUDA wheel installed)")
        return "cuda"

    log.info("[onnx_provider] hw_provider=auto -> cpu (%s)", reason)
    return "cpu"

"""
Host-side unit tests for utils/onnx_provider (no sherpa-onnx install required).

Run from the repo root:
    python -m pytest perception/tests -q

`sherpa_onnx` is replaced by a stub module whose __file__ points into a tmp_path
tree, so the CUDA-wheel marker can be made present or absent at will.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from utils import onnx_provider  # noqa: E402

INT8 = "/models/asr/encoder.int8.onnx"
FP32 = "/models/tts/model-steps-3.onnx"


@pytest.fixture(autouse=True)
def _clear_cache():
    """cuda_available is lru_cached for the process lifetime."""
    onnx_provider.cuda_available.cache_clear()
    yield
    onnx_provider.cuda_available.cache_clear()


def _install_fake_sherpa(monkeypatch, tmp_path, *, with_cuda: bool):
    pkg_dir = tmp_path / "sherpa_onnx"
    (pkg_dir / "lib").mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("")
    if with_cuda:
        (pkg_dir / "lib" / "libonnxruntime_providers_cuda.so").write_bytes(b"")
    module = types.ModuleType("sherpa_onnx")
    module.__file__ = str(pkg_dir / "__init__.py")
    monkeypatch.setitem(sys.modules, "sherpa_onnx", module)


# ── cuda_available ───────────────────────────────────────────────────────────

def test_cuda_available_true_when_marker_present(monkeypatch, tmp_path):
    _install_fake_sherpa(monkeypatch, tmp_path, with_cuda=True)
    assert onnx_provider.cuda_available() is True


def test_cuda_available_false_on_cpu_wheel(monkeypatch, tmp_path):
    _install_fake_sherpa(monkeypatch, tmp_path, with_cuda=False)
    assert onnx_provider.cuda_available() is False


def test_cuda_available_false_when_sherpa_missing(monkeypatch):
    """x86 dev hosts and the unit-test environment have no sherpa-onnx at all."""
    real_import = __import__

    def _fail(name, *args, **kwargs):
        if name == "sherpa_onnx":
            raise ImportError("no sherpa_onnx")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "sherpa_onnx", raising=False)
    monkeypatch.setattr("builtins.__import__", _fail)
    assert onnx_provider.cuda_available() is False


# ── is_quantised ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("paths, expected", [
    ((INT8,), True),
    ((FP32,), False),
    ((INT8, FP32), True),                       # any int8 file poisons the set
    (("/m/decoder-epoch-99-avg-1.onnx", "/m/joiner-epoch-99-avg-1.int8.onnx"), True),
    ((), False),
    ((None, ""), False),                        # unset paths are not int8 evidence
])
def test_is_quantised(paths, expected):
    assert onnx_provider.is_quantised(paths) is expected


# ── resolve_provider ─────────────────────────────────────────────────────────

def test_auto_picks_cuda_for_fp32_on_gpu_wheel(monkeypatch, tmp_path):
    _install_fake_sherpa(monkeypatch, tmp_path, with_cuda=True)
    assert onnx_provider.resolve_provider("auto", (FP32,)) == "cuda"


def test_auto_picks_cpu_for_int8_even_on_gpu_wheel(monkeypatch, tmp_path):
    """The measured reason: ONNX Runtime's CUDA EP has no int8 kernels and falls
    back per node, which came out 2-3x slower than 2-thread CPU on an Orin NX."""
    _install_fake_sherpa(monkeypatch, tmp_path, with_cuda=True)
    assert onnx_provider.resolve_provider("auto", (INT8,)) == "cpu"


def test_auto_with_no_paths_assumes_fp32(monkeypatch, tmp_path):
    """A caller that forgets model_paths errs toward the GPU, not toward a silent
    CPU pin — an unexpectedly slow ASR is harder to notice than a fast one."""
    _install_fake_sherpa(monkeypatch, tmp_path, with_cuda=True)
    assert onnx_provider.resolve_provider("auto") == "cuda"


@pytest.mark.parametrize("value", ["auto", "AUTO", " auto ", "", None])
def test_auto_and_unset_fall_back_to_cpu_on_cpu_wheel(monkeypatch, tmp_path, value):
    _install_fake_sherpa(monkeypatch, tmp_path, with_cuda=False)
    assert onnx_provider.resolve_provider(value, (FP32,)) == "cpu"


@pytest.mark.parametrize("value", ["cuda", "CUDA", " cuda "])
def test_explicit_cuda_passes_through_on_int8_and_cpu_wheel(monkeypatch, tmp_path, value):
    """A pinned provider is not second-guessed: either it is a deliberate choice
    or a misconfigured deployment, and both should stay visible."""
    _install_fake_sherpa(monkeypatch, tmp_path, with_cuda=False)
    assert onnx_provider.resolve_provider(value, (INT8,)) == "cuda"


def test_explicit_cpu_passes_through_for_fp32_on_gpu_wheel(monkeypatch, tmp_path):
    _install_fake_sherpa(monkeypatch, tmp_path, with_cuda=True)
    assert onnx_provider.resolve_provider("cpu", (FP32,)) == "cpu"


def test_unknown_value_falls_back_to_cpu(monkeypatch, tmp_path):
    _install_fake_sherpa(monkeypatch, tmp_path, with_cuda=True)
    assert onnx_provider.resolve_provider("trt", (FP32,)) == "cpu"

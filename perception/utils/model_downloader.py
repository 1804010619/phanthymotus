"""
utils/model_downloader.py — Auto-download sherpa-onnx models from COS if missing.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tarfile
import tempfile
import time
import zipfile
from urllib.error import URLError
from urllib.request import urlopen, urlretrieve

try:  # Linux only; the perception images are Linux, dev hosts may not be.
    import fcntl
except ImportError:  # pragma: no cover - Windows/macOS dev hosts
    fcntl = None

log = logging.getLogger(__name__)

COS_BASE = "https://agi-phanthy-dev-1252788780.cos.ap-beijing.myqcloud.com/public"


def _progress_hook(name: str):
    """Create a reporthook for urlretrieve that logs download progress."""
    last_pct = [0]
    def hook(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(int(block_num * block_size * 100 / total_size), 100)
            if pct >= last_pct[0] + 10:
                last_pct[0] = pct
                mb_done = block_num * block_size / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                log.info(f"[model_downloader] {name}: {pct}% ({mb_done:.1f}/{mb_total:.1f} MB)")
    return hook

MODELS = {
    "asr": {
        "url": f"{COS_BASE}/sherpa-onnx-streaming-paraformer-bilingual-zh-en.zip",
        "check_file": "tokens.txt",
    },
    "asr_en": {
        "url": f"{COS_BASE}/sherpa-onnx-streaming-zipformer-en-2023-06-26.zip",
        "check_file": "tokens.txt",
    },
    "asr_sensevoice": {
        "url": f"{COS_BASE}/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.zip",
        "check_file": "tokens.txt",
    },
    "asr_paraformer_offline": {
        "url": f"{COS_BASE}/sherpa-onnx-paraformer-zh-small-2024-03-09.tar.bz2",
        "check_file": "tokens.txt",
    },
    "asr_x_asr": {
        "url": f"{COS_BASE}/x-asr-zh-en-punct-int8-robot.zip",
        "check_file": "tokens.txt",
    },
    "tts": {
        "url": f"{COS_BASE}/matcha-icefall-zh-en.tar.bz2",
        "check_file": "model-steps-3.onnx",
    },
    "tts_vocoder": {
        "url": f"{COS_BASE}/vocos-16khz-univ.onnx",
        "check_file": "vocos-16khz-univ.onnx",
        "single_file": True,
    },
    "kws": {
        "url": f"{COS_BASE}/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2",
        "check_file": "tokens.txt",
    },
    "kws_zh": {
        "url": f"{COS_BASE}/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.zip",
        "check_file": "tokens.txt",
    },
    "kws_en": {
        "url": f"{COS_BASE}/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.zip",
        "check_file": "tokens.txt",
    },
    "vad": {
        "url": f"{COS_BASE}/silero_vad.onnx",
        "check_file": "silero_vad.onnx",
        "single_file": True,  # Not an archive, just a single file download
    },
    "denoise": {
        "url": f"{COS_BASE}/gtcrn_simple.onnx",
        "check_file": "gtcrn_simple.onnx",
        "single_file": True,
    },
}


def ensure_model(name: str, model_dir: str) -> None:
    """Ensure model files exist in model_dir. Download from COS if missing."""
    info = MODELS.get(name)
    if not info:
        raise ValueError(f"Unknown model name: {name}")

    check_path = os.path.join(model_dir, info["check_file"])
    if os.path.exists(check_path):
        log.info(f"[model_downloader] {name}: already exists at {model_dir}")
        return

    url = info["url"]
    os.makedirs(model_dir, exist_ok=True)
    log.info(f"[model_downloader] {name}: downloading from {url} ...")

    if info.get("single_file"):
        # Direct file download (not an archive)
        dest = os.path.join(model_dir, info["check_file"])
        urlretrieve(url, dest, reporthook=_progress_hook(name))
        log.info(f"[model_downloader] {name}: done.")
        return

    # Determine suffix from URL
    if url.endswith(".zip"):
        suffix = ".zip"
    else:
        suffix = ".tar.bz2"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        urlretrieve(url, tmp_path, reporthook=_progress_hook(name))
        log.info(f"[model_downloader] {name}: extracting to {model_dir} ...")

        if suffix == ".zip":
            _extract_zip(tmp_path, model_dir)
        else:
            _extract_tar(tmp_path, model_dir)

        log.info(f"[model_downloader] {name}: done.")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Verify
    if not os.path.exists(check_path):
        raise RuntimeError(
            f"[model_downloader] {name}: download completed but {info['check_file']} "
            f"not found in {model_dir}"
        )


def _extract_zip(zip_path: str, model_dir: str) -> None:
    """Extract zip, stripping common top-level directory prefix."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Filter out __MACOSX and directory entries
        names = [n for n in zf.namelist()
                 if not n.endswith('/') and not n.startswith('__MACOSX')]
        if not names:
            raise RuntimeError(f"Empty archive: {zip_path}")

        prefix = _common_prefix_from_names(names)
        for name in names:
            stripped = name[len(prefix):] if prefix else name
            if not stripped:
                continue
            dest = os.path.join(model_dir, stripped)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(name) as src, open(dest, 'wb') as dst:
                dst.write(src.read())


def _extract_tar(tar_path: str, model_dir: str) -> None:
    """Extract tar.bz2, stripping common top-level directory prefix."""
    with tarfile.open(tar_path, "r:bz2") as tf:
        members = tf.getmembers()
        if not members:
            raise RuntimeError(f"Empty archive: {tar_path}")

        names = [m.name for m in members if not m.isdir()]
        prefix = _common_prefix_from_names(names)
        for m in members:
            if m.isdir():
                continue
            if prefix:
                m.name = m.name[len(prefix):]
            if not m.name:
                continue
            m.name = m.name.lstrip("/")
            tf.extract(m, model_dir)


def _common_prefix_from_names(names: list[str]) -> str:
    """Find common top-level directory prefix from file name list."""
    dirs_with_slash = [n.split("/", 1) for n in names if "/" in n]
    if not dirs_with_slash:
        return ""
    first_parts = set(parts[0] for parts in dirs_with_slash)
    if len(first_parts) == 1:
        return first_parts.pop() + "/"
    return ""


# ── Verified bundles (OCR / obstacle TensorRT artefacts) ─────────────────────
# Pure additions consumed by the vision plugins' thin wrappers. The legacy
# ensure_model() above (sherpa-onnx archives, X-ASR) is intentionally left
# untouched. Every file in a verified bundle carries a pinned size and SHA256:
# existing files are re-verified before reuse, downloads are staged next to
# the destination, verified, and only then moved into place. Concurrent
# instances sharing /models serialize on a per-bundle file lock. Entries that
# ship one bundle per JetPack family use {"jp511": {...}, "jp61": {...}} keys
# selected by the TensorRT that is actually importable
# (see utils.tensorrt_runtime).


MODELS_ROOT = "/models"


def require_models_subpath(path: str, root: str = MODELS_ROOT) -> str:
    """Validate that a caller-supplied model_dir stays inside the models tree.

    model_dir is accepted over MCP config, and the downloader runs as root in
    the container — an unchecked value would let a caller create or overwrite
    files at arbitrary container paths. Returns the normalized absolute path.
    """
    normalized = os.path.normpath(os.path.join("/", str(path)))
    root_norm = os.path.normpath(root)
    if normalized != root_norm and not normalized.startswith(root_norm + os.sep):
        raise ValueError(
            f"model_dir must be under {root_norm}/: got {path!r}"
        )
    return normalized


def select_bundle_family(bundles: dict, family: str | None = None) -> str:
    """Pick the bundle key ("jp511"/"jp61") for the runtime TensorRT.

    An explicit ``family`` (or alias such as "61"/"511") wins; otherwise the
    family is derived from the importable TensorRT major version. Never
    depends on a Docker build argument or image ENV.
    """
    from utils.tensorrt_runtime import normalize_family, tensorrt_family

    if family is not None:
        key = normalize_family(family)
        if key is None:
            raise ValueError(f"Unknown model bundle family: {family!r}")
    else:
        key = tensorrt_family()
    if key not in bundles:
        raise RuntimeError(
            f"No model bundle for TensorRT family {key}; available: {sorted(bundles)}"
        )
    return key


def ensure_verified_bundle(
    name: str, model_dir: str, base_url: str, files: dict
) -> dict[str, str]:
    """Ensure a size/SHA256-pinned bundle is present and valid in model_dir.

    existing files → size check → SHA256 check → reuse
    otherwise      → lock → re-check → download (retry) → verify → replace
    Returns ``{filename: absolute path}``.
    """
    paths = {
        filename: os.path.join(model_dir, filename) for filename in files
    }
    if _bundle_matches(model_dir, files):
        log.info(f"[model_downloader] {name}: verified bundle already at {model_dir}")
        return paths

    os.makedirs(model_dir, exist_ok=True)
    # Platform instances share /models. Serialize the download so a cold
    # multi-instance launch fetches one copy instead of one per process; a
    # waiter re-checks the bundle once it gets the lock.
    lock_path = os.path.join(model_dir, f".{name.replace('/', '_')}.lock")
    with open(lock_path, "a+b") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if _bundle_matches(model_dir, files):
                log.info(f"[model_downloader] {name}: verified by another instance")
                return paths
            _download_verified_bundle(name, base_url, model_dir, files)
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return paths


def _file_matches(path: str, metadata: dict) -> bool:
    """Return whether one file exists and matches its pinned size and SHA256."""
    try:
        if not os.path.isfile(path):
            return False
        _verify_pinned_file(path, metadata)
    except (OSError, ValueError):
        return False
    return True


def _bundle_matches(model_dir: str, files: dict) -> bool:
    """Return whether every bundle file matches its pinned size and SHA256."""
    return all(
        _file_matches(os.path.join(model_dir, filename), metadata)
        for filename, metadata in files.items()
    )


def _verify_pinned_file(path: str, metadata: dict) -> None:
    actual_size = os.path.getsize(path)
    if actual_size != metadata["size"]:
        raise ValueError(
            f"size mismatch for {os.path.basename(path)}: "
            f"expected {metadata['size']}, got {actual_size}"
        )

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != metadata["sha256"]:
        raise ValueError(
            f"SHA256 mismatch for {os.path.basename(path)}: "
            f"expected {metadata['sha256']}, got {actual_sha256}"
        )


def _download_verified_bundle(
    name: str, base_url: str, model_dir: str, files: dict
) -> None:
    """Download and verify a multi-file model before replacing its destination."""
    os.makedirs(model_dir, exist_ok=True)
    staging_prefix = f".{name.replace('/', '_')}-"
    with tempfile.TemporaryDirectory(prefix=staging_prefix, dir=model_dir) as staging:
        for filename, metadata in files.items():
            if os.path.basename(filename) != filename:
                raise ValueError(f"Invalid model filename: {filename}")
            url = f"{base_url.rstrip('/')}/{filename}"
            destination = os.path.join(staging, filename)
            last_error = None
            for attempt in range(1, 4):
                try:
                    log.info(
                        f"[model_downloader] {name}: downloading {filename} "
                        f"(attempt {attempt}/3)"
                    )
                    with urlopen(url, timeout=120) as response, open(destination, "wb") as output:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                    _verify_pinned_file(destination, metadata)
                    os.chmod(destination, 0o644)
                    break
                except (URLError, TimeoutError, OSError, ValueError) as error:
                    last_error = error
                    if os.path.exists(destination):
                        os.unlink(destination)
                    if attempt < 3:
                        time.sleep(3)
            else:
                raise RuntimeError(
                    f"[model_downloader] {name}: failed to download {filename}"
                ) from last_error

        for filename in files:
            os.replace(
                os.path.join(staging, filename),
                os.path.join(model_dir, filename),
            )
    log.info(f"[model_downloader] {name}: verified bundle ready at {model_dir}")

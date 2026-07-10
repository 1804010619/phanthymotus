#!/usr/bin/env python3
"""Download and verify the offline ASR model used by the Jetson image."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path
from urllib.request import urlretrieve


DEFAULT_BASE_URL = (
    "http://172.28.4.81:34567/zengzhitao/embodied-ai/official_paraformer"
)

OFFICIAL_PARAFORMER_MANIFEST = {
    "config.json": "73c643043ccf143186ea8010d7cea803f49c0d31089a4d192808119e7c2880b3",
    "model.int8.onnx": "3ef6c19369b912f7caf3cef8e545c5ccd1a33d9d7ec792a46668dc41c4b229ec",
    "tokens.txt": "4b2d964e18b9cf139b473003b6698fb2ed9a2a5ec55b93daa677b28f578897aa",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(
    base_url: str,
    output_dir: str | Path,
    manifest: dict[str, str],
) -> None:
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    for filename, expected_sha256 in manifest.items():
        destination = destination_dir / filename
        url = f"{base_url.rstrip('/')}/{filename}"
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{filename}.", dir=destination_dir, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
            print(f"Downloading {url}", flush=True)
            urlretrieve(url, temporary_path)
            actual_sha256 = sha256_file(temporary_path)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"SHA256 mismatch for {filename}: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )
            temporary_path.replace(destination)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--output-dir", default="/models/sherpa-onnx/asr-offline"
    )
    args = parser.parse_args()
    download_model(args.base_url, args.output_dir, OFFICIAL_PARAFORMER_MANIFEST)


if __name__ == "__main__":
    main()

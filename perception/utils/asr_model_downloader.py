#!/usr/bin/env python3
"""Download the offline ASR model used by the Jetson image."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from urllib.request import urlretrieve


DEFAULT_BASE_URL = (
    "http://172.28.4.81:34567/zengzhitao/embodied-ai/official_paraformer"
)

OFFICIAL_PARAFORMER_FILES = (
    "config.json",
    "model.int8.onnx",
    "tokens.txt",
)


def download_model(
    base_url: str,
    output_dir: str | Path,
    filenames: tuple[str, ...],
) -> None:
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    for filename in filenames:
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
    download_model(args.base_url, args.output_dir, OFFICIAL_PARAFORMER_FILES)


if __name__ == "__main__":
    main()

#!/bin/bash
# Jetson TTS perception entrypoint — verify CUDA before loading sherpa GPU provider.
set -euo pipefail

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

if [ "${TTS_REQUIRE_CUDA:-1}" = "1" ]; then
    if ! python3 - <<'PY'
import sys
try:
    import torch
except Exception:
    sys.exit(1)
if not torch.cuda.is_available():
    sys.exit(1)
print(torch.cuda.get_device_name(0))
PY
    then
        echo "[entrypoint] FATAL: config uses hw_provider=cuda but CUDA is not available." >&2
        echo "[entrypoint] judgeflow must start the container with GPU runtime, for example:" >&2
        echo "  docker run --runtime nvidia \\" >&2
        echo "    -e NVIDIA_VISIBLE_DEVICES=all \\" >&2
        echo "    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \\" >&2
        echo "    ... <image>" >&2
        echo "[entrypoint] See deploy/judgeflow_tts_run.sh in this repository." >&2
        exit 125
    fi
fi

source /opt/ros/humble/install/setup.bash
source /ros_ws/install/setup.bash
exec python3 /work/main.py

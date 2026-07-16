#!/usr/bin/env bash
set -euo pipefail

export TTS_PLUGIN=${TTS_PLUGIN:-vits2_tts_trt}
export DEFAULT_TTS_PLUGIN=${DEFAULT_TTS_PLUGIN:-vits2_tts_trt}

set +u
source /opt/ros/jazzy/setup.bash
source /ros_ws/install/setup.bash
set -u

cd /work
exec /usr/bin/python3 /work/main.py

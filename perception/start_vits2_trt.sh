#!/usr/bin/env bash
set -euo pipefail

TRT_PORT=${VITS2_TRT_PORT:-18080}
TRT_ROOT=/work/plugins/vits2_tts_trt/runtime
TRT_PID=""
MAIN_PID=""

cleanup() {
    if [[ -n "$TRT_PID" ]]; then
        kill "$TRT_PID" >/dev/null 2>&1 || true
    fi
    if [[ -n "$MAIN_PID" ]]; then
        kill "$MAIN_PID" >/dev/null 2>&1 || true
    fi
    [[ -z "$TRT_PID" ]] || wait "$TRT_PID" >/dev/null 2>&1 || true
    [[ -z "$MAIN_PID" ]] || wait "$MAIN_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

cd "$TRT_ROOT"
/usr/bin/python3 -m uvicorn service:app \
    --host 127.0.0.1 --port "$TRT_PORT" --workers 1 \
    --loop uvloop --http httptools --no-access-log &
TRT_PID=$!

for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${TRT_PORT}/ready" | grep -q True; then
        break
    fi
    if ! kill -0 "$TRT_PID" >/dev/null 2>&1; then
        wait "$TRT_PID"
        exit 1
    fi
    sleep 1
done
curl -fsS "http://127.0.0.1:${TRT_PORT}/ready" | grep -q True

export TTS_PLUGIN=${TTS_PLUGIN:-vits2_tts_trt}
export VITS2_TRT_BACKEND_URL=${VITS2_TRT_BACKEND_URL:-http://127.0.0.1:${TRT_PORT}}
set +u
source /opt/ros/jazzy/setup.bash
source /ros_ws/install/setup.bash
set -u
cd /work
/usr/bin/python3 /work/main.py &
MAIN_PID=$!
set +e
wait -n "$TRT_PID" "$MAIN_PID"
STATUS=$?
set -e
kill "$TRT_PID" "$MAIN_PID" >/dev/null 2>&1 || true
wait "$TRT_PID" "$MAIN_PID" >/dev/null 2>&1 || true
exit "$STATUS"

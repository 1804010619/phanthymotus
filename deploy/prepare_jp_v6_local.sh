#!/usr/bin/env bash
# prepare_jp_v6_local.sh — 在 Jetson 本机构建 jp6-torch base（无需 TCR 凭证）
#
# Usage (on Jetson JP6 host):
#   ./prepare_jp_v6_local.sh
#   export BASE_IMAGE=local/phanthy-motus/jetson-base:jp6-torch
#   ./build_perception.sh --variant jetson
set -euo pipefail

# DaoCloud 镜像加速（与 prepare_jetson_base.sh 一致）；直连 Docker Hub 超时时用默认即可
ROS_BASE="${ROS_BASE:-docker.m.daocloud.io/dustynv/ros:humble-desktop-l4t-r36.4.0}"
PYTORCH_DONOR="${PYTORCH_DONOR:-docker.m.daocloud.io/dustynv/l4t-pytorch:r36.4.0}"
TARGET="${TARGET:-local/phanthy-motus/jetson-base:jp6-torch}"

echo "============================================"
echo "Building local Jetson JP6 PyTorch base"
echo "ROS base: ${ROS_BASE}"
echo "Donor:    ${PYTORCH_DONOR}"
echo "Target:   ${TARGET}"
echo "============================================"

docker pull "${ROS_BASE}"
docker pull "${PYTORCH_DONOR}"

TMPFILE="$(mktemp)"
cat > "${TMPFILE}" <<DOCKERFILE
FROM ${PYTORCH_DONOR} AS pytorch-donor
FROM ${ROS_BASE}
RUN rm -f /etc/apt/sources.list.d/* && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* && \
    apt-get -o Acquire::AllowInsecureRepositories=true update && \
    apt-get install -y --no-install-recommends --allow-unauthenticated libopenblas-base && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
COPY --from=pytorch-donor /usr/local/lib/python3.10/dist-packages/ /tmp/pt-donor/
RUN cd /tmp/pt-donor && \
    mkdir -p /usr/local/lib/python3.10/dist-packages && \
    for item in torch torchgen torchvision triton; do \
        if [ -e "\$item" ]; then cp -a "\$item" /usr/local/lib/python3.10/dist-packages/; fi; \
    done && \
  shopt -s nullglob && \
    for item in torch-*.dist-info torchvision-*.dist-info triton-*.dist-info; do \
        cp -a "\$item" /usr/local/lib/python3.10/dist-packages/; \
    done && \
    rm -rf /tmp/pt-donor
DOCKERFILE

docker build -f "${TMPFILE}" -t "${TARGET}" .
rm -f "${TMPFILE}"

echo ""
echo "Done. Local JP6 base:"
echo "  ${TARGET}"
echo ""
echo "Next:"
echo "  export BASE_IMAGE=${TARGET}"
echo "  ./deploy/build_perception.sh --variant jetson"

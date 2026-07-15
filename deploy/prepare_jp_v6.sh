#!/usr/bin/env bash
# prepare_jp_v6.sh — 构建含 GPU PyTorch 的 Jetson JP6 base 镜像并推送到 TCR
#
# 产出：jetson-base:jp6-torch 镜像，包含 JetPack 6.x (L4T r36.4) + PyTorch GPU
#
# Usage:
#   ./prepare_jp_v6.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_FILE="${SCRIPT_DIR}/.env"
if [ -f "${ENV_FILE}" ]; then
    source "${ENV_FILE}"
fi

if [ -z "${REGISTRY:-}" ] || [ -z "${REGISTRY_USER:-}" ] || [ -z "${REGISTRY_PASSWORD:-}" ] || [ -z "${IMAGE_NAMESPACE:-}" ]; then
    echo "[error] Registry not configured. This script requires a registry to push images."
    echo "        Copy deploy/.env.example to deploy/.env and fill in values."
    exit 1
fi

BASE_IMAGE="${REGISTRY}/${IMAGE_NAMESPACE}/jetson-base:humble-desktop-l4t-r36.4.0"
TARGET="${REGISTRY}/${IMAGE_NAMESPACE}/jetson-base:jp6-torch"
PYTORCH_DONOR="dustynv/l4t-pytorch:r36.4.0"

echo "============================================"
echo "Building Jetson JP6 PyTorch base image"
echo "Base:   ${BASE_IMAGE}"
echo "Donor:  ${PYTORCH_DONOR}"
echo "Target: ${TARGET}"
echo "============================================"

docker pull "${PYTORCH_DONOR}"

TMPFILE="$(mktemp)"
cat > "${TMPFILE}" <<DOCKERFILE
FROM ${PYTORCH_DONOR} AS pytorch-donor
FROM ${BASE_IMAGE}
RUN rm -f /etc/apt/sources.list.d/* && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* && \
    apt-get -o Acquire::AllowInsecureRepositories=true update && \
    apt-get install -y --no-install-recommends --allow-unauthenticated libopenblas-base && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
COPY --from=pytorch-donor /usr/local/lib/python3.10/dist-packages/ /tmp/pt-donor/
RUN cd /tmp/pt-donor && \
    mkdir -p /usr/local/lib/python3.10/dist-packages && \
    for item in torch torchgen torchvision triton; do \
        [ -e "\$item" ] && cp -a "\$item" /usr/local/lib/python3.10/dist-packages/; \
    done && \
    for item in torch-*.dist-info torchvision-*.dist-info triton-*.dist-info; do \
        [ -e "\$item" ] && cp -a "\$item" /usr/local/lib/python3.10/dist-packages/; \
    done && \
    rm -rf /tmp/pt-donor
DOCKERFILE

echo "${REGISTRY_PASSWORD}" | docker login "${REGISTRY}" -u "${REGISTRY_USER}" --password-stdin

docker build -f "${TMPFILE}" -t "${TARGET}" .
rm -f "${TMPFILE}"

echo "Pushing → ${TARGET}"
docker push "${TARGET}"

echo ""
echo "Done. Image available at:"
echo "  ${TARGET}"
echo ""
echo "Update Dockerfile.jetson BASE_IMAGE to:"
echo "  ARG BASE_IMAGE=${TARGET}"

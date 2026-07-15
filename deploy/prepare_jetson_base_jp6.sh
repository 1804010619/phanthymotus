#!/usr/bin/env bash
# prepare_jetson_base_jp6.sh — 中转 dustynv/ros JP6 镜像到腾讯云仓库
#
# Usage:
#   ./prepare_jetson_base_jp6.sh
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

SOURCE="docker.m.daocloud.io/dustynv/ros:humble-desktop-l4t-r36.4.0"
TARGET="${REGISTRY}/${IMAGE_NAMESPACE}/jetson-base:humble-desktop-l4t-r36.4.0"

echo "============================================"
echo "Pulling Jetson JP6 base from DaoCloud mirror"
echo "Source: ${SOURCE}"
echo "Target: ${TARGET}"
echo "============================================"

docker pull "${SOURCE}"

echo "Tagging → ${TARGET}"
docker tag "${SOURCE}" "${TARGET}"

echo "Pushing to registry..."
echo "${REGISTRY_PASSWORD}" | docker login "${REGISTRY}" -u "${REGISTRY_USER}" --password-stdin
docker push "${TARGET}"

echo ""
echo "Done. Jetson JP6 base image available at:"
echo "  ${TARGET}"

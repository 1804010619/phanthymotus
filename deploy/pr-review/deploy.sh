#!/usr/bin/env bash
# deploy.sh — Build and start PR Review Agent
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f .env ]; then
    echo "Error: .env not found. Copy .env.example and fill in values:"
    echo "  cp .env.example .env"
    exit 1
fi

# Setup QEMU for ARM64 cross-compilation (idempotent)
echo "Setting up QEMU for ARM64 cross-compilation..."
docker run --rm --privileged tonistiigi/binfmt --install arm64 2>/dev/null || true

# Setup buildx builder if not exists
if ! docker buildx inspect pr-review-builder &>/dev/null; then
    echo "Creating buildx builder..."
    docker buildx create --name pr-review-builder --use
else
    docker buildx use pr-review-builder
fi

# Build and start
echo "Building PR Review Agent..."
docker compose build

echo "Starting PR Review Agent..."
docker compose up -d

echo ""
echo "PR Review Agent is running on port 15690"
echo "  Status: curl http://localhost:15690/status"
echo "  Logs:   docker compose logs -f"

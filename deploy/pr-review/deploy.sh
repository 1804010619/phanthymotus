#!/usr/bin/env bash
# deploy.sh — build and start the PR Review Agent.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f .env ]; then
    echo "Error: .env not found. Create it first:"
    echo "  cp .env.example .env"
    echo "  \$EDITOR .env      # set GITHUB_TOKEN, REGISTRY_*, LLM_*"
    exit 1
fi

# Fail early on the settings that produce confusing runtime behaviour rather
# than a clean error.
missing=()
for var in GITHUB_TOKEN REGISTRY REGISTRY_USER REGISTRY_PASSWORD; do
    value="$(grep -E "^${var}=" .env | tail -1 | cut -d= -f2- || true)"
    if [ -z "$value" ] || [[ "$value" == your_* ]] || [[ "$value" == ghp_your_* ]]; then
        missing+=("$var")
    fi
done
if [ ${#missing[@]} -gt 0 ]; then
    echo "Error: these .env values are unset or still placeholders:"
    printf '  - %s\n' "${missing[@]}"
    exit 1
fi

# QEMU registration for ARM64 cross-compilation. Idempotent, and it persists
# on the host until reboot — the build scripts rely on it being present.
echo "==> Registering QEMU for ARM64 cross-compilation"
docker run --rm --privileged tonistiigi/binfmt --install arm64 >/dev/null 2>&1 \
    || echo "    warning: QEMU registration failed; ARM64 builds may not work"

echo "==> Building agent image"
docker compose build

echo "==> Starting agent"
docker compose up -d

echo
echo "PR Review Agent is running."
echo "  Status:  curl -s http://localhost:15690/status | python3 -m json.tool"
echo "  Jobs:    curl -s http://localhost:15690/jobs | python3 -m json.tool"
echo "  Logs:    docker compose -f $SCRIPT_DIR/docker-compose.yml logs -f"
echo
echo "Trigger a review by commenting /request_bot_review on a PR."
echo "Polling picks it up within POLL_INTERVAL_SECONDS (default 30s)."

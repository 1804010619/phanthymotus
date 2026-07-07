#!/bin/sh
# entrypoint.sh — merge service compose from new image and restart
# Env: COMPOSE_DIR, SERVICE, NEW_IMAGE
set -e

: "${COMPOSE_DIR:?COMPOSE_DIR required}"
: "${SERVICE:?SERVICE required}"
: "${NEW_IMAGE:?NEW_IMAGE required}"

echo "[restart] service=${SERVICE} image=${NEW_IMAGE}"
echo "[restart] compose_dir=${COMPOSE_DIR}"

COMPOSE_FILE="${COMPOSE_DIR}/docker-compose.yml"

# Extract service compose from image into a temp file
echo "[restart] extracting compose from image..."
CID=$(docker create "${NEW_IMAGE}")
docker cp "${CID}:/deploy/docker-compose.yml" /tmp/new-compose.yml 2>/dev/null || true
docker rm "${CID}" >/dev/null

# Merge: update the service definition in existing compose (preserving other services)
# If compose file doesn't exist, just use the extracted one directly.
if [ -f "${COMPOSE_FILE}" ] && [ -f /tmp/new-compose.yml ]; then
    # Use the agent-core container (which has python3+yaml) to merge
    # If python3 is available locally (alpine), use it; otherwise fall back to sed
    if command -v python3 >/dev/null 2>&1; then
        python3 - "${COMPOSE_FILE}" /tmp/new-compose.yml "${NEW_IMAGE}" "${SERVICE}" <<'PY'
import sys, yaml, os

compose_path, new_path, new_image, service = sys.argv[1:5]

with open(compose_path) as f:
    compose = yaml.safe_load(f) or {'services': {}}

with open(new_path) as f:
    new = yaml.safe_load(f) or {}

new_services = new.get('services', {})
if service in new_services:
    new_services[service]['image'] = new_image
    compose.setdefault('services', {})[service] = new_services[service]

with open(compose_path, 'w') as f:
    yaml.dump(compose, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
PY
    else
        # Fallback: no python3, just update image line via sed
        sed -i "/^  ${SERVICE}:/,/^  [^ ]/{s|image:.*|image: ${NEW_IMAGE}|}" "${COMPOSE_FILE}"
    fi
elif [ -f /tmp/new-compose.yml ]; then
    # No existing compose, use extracted one directly
    cp /tmp/new-compose.yml "${COMPOSE_FILE}"
    sed -i "s|__IMAGE__|${NEW_IMAGE}|" "${COMPOSE_FILE}"
fi

echo "[restart] starting service via docker compose..."
cd "${COMPOSE_DIR}"
docker compose up -d --no-deps --force-recreate "${SERVICE}"

echo "[restart] done. ${SERVICE} is up."

#!/bin/sh
# entrypoint.sh — extract compose from new image and restart service
# Env: COMPOSE_DIR, SERVICE, NEW_IMAGE
set -e

: "${COMPOSE_DIR:?COMPOSE_DIR required}"
: "${SERVICE:?SERVICE required}"
: "${NEW_IMAGE:?NEW_IMAGE required}"

echo "[restart] service=${SERVICE} image=${NEW_IMAGE}"
echo "[restart] compose_dir=${COMPOSE_DIR}"

# Extract updated docker-compose.yml from new image
echo "[restart] extracting compose from image..."
CID=$(docker create "${NEW_IMAGE}")
docker cp "${CID}:/deploy/docker-compose.yml" "${COMPOSE_DIR}/docker-compose.yml" 2>/dev/null || true
docker rm "${CID}" >/dev/null

# Update the image tag for the target service in compose file
# sed address range: from "  <service>:" to the next service definition, replace image line
sed -i "/^  ${SERVICE}:/,/^  [^ ]/{s|image:.*|image: ${NEW_IMAGE}|}" "${COMPOSE_DIR}/docker-compose.yml"

echo "[restart] starting service via docker compose..."
cd "${COMPOSE_DIR}"
docker compose up -d --no-deps --force-recreate "${SERVICE}"

echo "[restart] done. ${SERVICE} is up."

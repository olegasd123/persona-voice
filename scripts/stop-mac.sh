#!/usr/bin/env bash
# Persona-Voice - stop the dev stack on macOS. Fallback for when run-mac.sh was killed
# without running its own cleanup (closed terminal, crash).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Stopping Persona-Voice dev stack..."
docker compose -f docker-compose.livekit.yml down
echo "Stopped."

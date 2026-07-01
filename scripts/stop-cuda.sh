#!/usr/bin/env bash
# Persona-Voice - stop the production (CUDA) stack on Linux. Fallback for when run-cuda.sh was
# killed without running its own cleanup (closed terminal, crash), or to stop containers started
# by hand. Normally Ctrl+C in run-cuda.sh tears everything down.
# `set -uo pipefail` (no -e) so a failure in the first `down` doesn't skip the second.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Stopping Persona-Voice stack..."
docker compose -f docker-compose.livekit.yml down
docker compose down
echo "Stopped."

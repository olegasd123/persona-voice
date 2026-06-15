#!/usr/bin/env bash
# Dev convenience wrapper. For M0 it runs the config check; M3 wires up the live server.
set -euo pipefail

cd "$(dirname "$0")/.."

BACKEND="${BACKEND:-mac}"
echo "Persona-Voice dev (BACKEND=$BACKEND)"

python -m personavoice.server --check --backend "$BACKEND"

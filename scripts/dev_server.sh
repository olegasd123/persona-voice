#!/usr/bin/env bash
# Dev convenience wrapper. Runs the config check (run the live server with 'personavoice --serve').
set -euo pipefail

cd "$(dirname "$0")/.."

BACKEND="${BACKEND:-mac}"
echo "Persona-Voice dev (BACKEND=$BACKEND)"

python -m personavoice.server --check --backend "$BACKEND"

#!/usr/bin/env bash
# Persona-Voice - ONE-TIME setup for the dev stack on macOS (M4 Max).
#
# Run this once on a fresh checkout (or after a dependency / image change). It does the
# slow-but-cacheable work so the everyday ./scripts/run-mac.sh starts fast:
#
#   1. creates .venv312 (if missing) and installs the mac + livekit extras,
#   2. pulls the LiveKit Docker image,
#   3. builds the persona-voice token-server image.
#
# The brain on mac is LM Studio (or Ollama), started separately - no model is downloaded here.
# The host STT (whisper-mlx) + TTS (Kokoro) weights download from Hugging Face on the first
# run-mac (once, then cached).
#
# After this, start the stack with: ./scripts/run-mac.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- [1/3] virtualenv + Python deps --------------------------------------------------------
echo "[1/3] Python environment (.venv312)..."
if [[ ! -x ".venv312/bin/python" ]]; then
    echo "  Creating .venv312 (Python 3.12)..."
    if command -v uv >/dev/null 2>&1; then
        uv venv --python 3.12 .venv312
    elif command -v python3.12 >/dev/null 2>&1; then
        python3.12 -m venv .venv312
    else
        echo "  Need uv or python3.12 to create the venv. Install one and re-run." >&2
        exit 1
    fi
fi
PY=".venv312/bin/python"
echo "  Installing the mac + livekit extras..."
if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PY" -e '.[mac,livekit]'
else
    "$PY" -m pip install -e '.[mac,livekit]'
fi

# --- preflight -----------------------------------------------------------------------------
docker info >/dev/null 2>&1 || { echo "Docker is not running. Start Docker Desktop and retry."; exit 1; }

# --- [2/3] pull the LiveKit image ----------------------------------------------------------
echo "[2/3] Pulling the LiveKit image..."
docker compose -f docker-compose.livekit.yml pull livekit

# --- [3/3] build the token-server image ----------------------------------------------------
echo "[3/3] Building the persona-voice token-server image..."
docker compose -f docker-compose.livekit.yml build

echo
echo "Setup complete. Start LM Studio (load a model), then run:  ./scripts/run-mac.sh"

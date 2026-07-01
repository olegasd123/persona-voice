#!/usr/bin/env bash
# Persona-Voice - ONE-TIME setup for the production (CUDA) stack on Linux.
#
# Run this once on a fresh checkout (or after a dependency / image change). It does the
# slow-but-cacheable prep so the everyday ./scripts/run-cuda.sh starts fast - WITHOUT starting
# any server:
#
#   1. creates .venv312 (if missing) and installs the cuda + livekit + clone (Chatterbox) extras,
#   2. pulls the vLLM and LiveKit Docker images,
#   3. builds the persona-voice token-server image,
#   4. pre-downloads the host-side STT (faster-whisper) + TTS (Chatterbox) weights.
#
# The vLLM LLM model is NOT downloaded here: it caches into the persistent hf-cache volume on the
# first run-cuda (when vLLM actually starts). That keeps setup free of any running server.
#
# Prereqs: Docker + the NVIDIA Container Toolkit, and uv or python3.12 for the venv. The CUDA
# worker runs faster-whisper (STT) + Chatterbox (TTS) on the host GPU, which is why the clone
# extra (Chatterbox) is installed here and not only in the server image.
#
# After this, start the stack with: ./scripts/run-cuda.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- [1/4] virtualenv + Python deps --------------------------------------------------------
echo "[1/4] Python environment (.venv312)..."
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
echo "  Installing the cuda + livekit + clone extras (this can take a while)..."
if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PY" -e '.[cuda,livekit,clone]'
else
    "$PY" -m pip install -e '.[cuda,livekit,clone]'
fi

# --- preflight -----------------------------------------------------------------------------
docker info >/dev/null 2>&1 || { echo "Docker is not running. Start the Docker daemon and retry."; exit 1; }

# --- [2/4] pull Docker images --------------------------------------------------------------
echo "[2/4] Pulling Docker images (vLLM + LiveKit)..."
docker compose pull vllm
docker compose -f docker-compose.livekit.yml pull livekit

# --- [3/4] build the token-server image ----------------------------------------------------
echo "[3/4] Building the persona-voice token-server image..."
docker compose -f docker-compose.livekit.yml build

# --- [4/4] pre-download the host-side STT/TTS weights --------------------------------------
echo "[4/4] Pre-downloading host STT + TTS weights (best-effort)..."
"$PY" scripts/prefetch_host_models.py || echo "  (prefetch skipped/failed - the first run-cuda will download them instead)"

echo
echo "Setup complete. Start the stack with:  ./scripts/run-cuda.sh"
echo "  (the first run downloads the vLLM model into the cache; later runs are fast)"

#!/usr/bin/env bash
# Persona-Voice - start the DEV stack on macOS (M4 Max) and run the live agent.
#
#   ./scripts/run-dev-mac.sh
#
# What it does (one command, then leave it running):
#   1. starts the self-hosted LiveKit SFU + token server in Docker,
#   2. runs the conversation worker (BACKEND=mac) in the FOREGROUND.
#
# Press Ctrl+C to stop: the worker exits and the LiveKit containers are torn down (trap).
# If a run dies without cleanup, run ./scripts/stop-dev-mac.sh.
#
# The brain on mac is LM Studio (or Ollama), NOT vLLM - start LM Studio and load a model first
# (config/backends/mac.yaml -> openai/gpt-oss-20b by default). This script warns if it is down.
#
# Works on a fresh machine (Docker + a .venv312 present, no images yet): the token-server image
# is built on first run. Override the dialed LAN IP with PV_LIVEKIT_IP=<addr>; skip the
# dependency check with NO_INSTALL=1.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- venv python (created e.g. with: uv venv --python 3.12 .venv312 && uv pip install -e '.[mac,livekit]')
if [[ -x ".venv312/bin/python" ]]; then
    PY=".venv312/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PY="python"
else
    PY="python3"
fi

# --- LAN IP the phone dials (dynamic, override with PV_LIVEKIT_IP) --------------------------
find_lan_ip() {
    if [[ -n "${PV_LIVEKIT_IP:-}" ]]; then echo "$PV_LIVEKIT_IP"; return; fi
    local iface ip=""
    iface="$(route -n get default 2>/dev/null | awk '/interface:/{print $2}')"
    [[ -n "$iface" ]] && ip="$(ipconfig getifaddr "$iface" 2>/dev/null || true)"
    if [[ -z "$ip" ]]; then
        for i in en0 en1 en2; do
            ip="$(ipconfig getifaddr "$i" 2>/dev/null || true)"
            [[ -n "$ip" ]] && break
        done
    fi
    echo "$ip"
}
HOST_IP="$(find_lan_ip)"
[[ -n "$HOST_IP" ]] || { echo "Could not determine a LAN IP. Set PV_LIVEKIT_IP=<addr> (the address your phone dials)."; exit 1; }

# --- environment ---------------------------------------------------------------------------
export BACKEND=mac
export LIVEKIT_API_KEY="${LIVEKIT_API_KEY:-devkey}"      # dev keys; set real ones for a real deployment
export LIVEKIT_API_SECRET="${LIVEKIT_API_SECRET:-secret}"
export LIVEKIT_URL="ws://${HOST_IP}:7880"

# --- preflight -----------------------------------------------------------------------------
docker info >/dev/null 2>&1 || { echo "Docker is not running. Start Docker Desktop and retry."; exit 1; }

if [[ "${NO_INSTALL:-0}" != "1" ]]; then
    if ! "$PY" -c 'import livekit.agents' >/dev/null 2>&1; then
        echo "Installing the 'livekit' extra (one-time)..."
        "$PY" -m pip install -e '.[livekit]'
    fi
fi

# --- teardown on exit (Ctrl+C) -------------------------------------------------------------
_cleaned=0
cleanup() {
    [[ "$_cleaned" == 1 ]] && return
    _cleaned=1
    echo
    echo "Stopping Persona-Voice dev stack..."
    docker compose -f docker-compose.livekit.yml down
    echo "Stopped."
}
trap cleanup INT TERM EXIT

# --- run -----------------------------------------------------------------------------------
echo "LAN IP: $HOST_IP"
echo "[1/2] Starting LiveKit SFU + token server..."
docker compose -f docker-compose.livekit.yml up -d --build

LM_URL="${LMSTUDIO_BASE_URL:-http://localhost:1234/v1}"
if ! curl -sf "${LM_URL}/models" >/dev/null 2>&1; then
    echo "WARNING: LM Studio not reachable at ${LM_URL} - start it and load a model, or the LLM warm-up will wait."
fi

echo "[2/2] Starting the conversation worker (BACKEND=mac). Press Ctrl+C to stop everything."
echo "On the phone app, set the token server URL to:  http://${HOST_IP}:8080"
echo
"$PY" -m personavoice.server --serve

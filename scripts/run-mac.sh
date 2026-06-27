#!/usr/bin/env bash
# Persona-Voice - start the DEV stack on macOS (M4 Max) and run the live agent.
#
# This is the FAST path: it assumes setup-mac.sh has already done the one-time work
# (installed deps, built/pulled the Docker images). On a fresh checkout run setup-mac once,
# then:
#
#   ./scripts/setup-mac.sh   # one-time
#   ./scripts/run-mac.sh     # every run
#
# What it does (one command, then leave it running):
#   1. starts the self-hosted LiveKit SFU + token server in Docker,
#   2. runs the conversation worker (BACKEND=mac) in the FOREGROUND.
#
# Press Ctrl+C to stop: the worker exits and the LiveKit containers are torn down (trap).
# If a run dies without cleanup, run ./scripts/stop-mac.sh.
#
# The brain on mac is LM Studio (or Ollama), NOT vLLM - start LM Studio and load a model first
# (config/backends/mac.yaml -> openai/gpt-oss-20b by default). This script warns if it is down.
#
# Override the dialed LAN IP with PV_LIVEKIT_IP=<addr>.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- venv python (created by setup-mac.sh) --------------------------------------------------
if [[ -x ".venv312/bin/python" ]]; then
    PY=".venv312/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PY="python"
else
    echo "No .venv312 found. Run the one-time setup first:  ./scripts/setup-mac.sh"
    exit 1
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

export PERSONAVOICE_LOG_LEVEL="${PERSONAVOICE_LOG_LEVEL:-TRACE}"
export PERSONAVOICE_TOOL_MAX_ITERS="${PERSONAVOICE_TOOL_MAX_ITERS:-4}"
export PERSONAVOICE_TOOL_TIMEOUT="${PERSONAVOICE_TOOL_TIMEOUT:-10}"
export PERSONAVOICE_SEMANTIC_ENDPOINTING="${PERSONAVOICE_SEMANTIC_ENDPOINTING:-1}"
export PERSONAVOICE_ENDPOINTING_GRACE_MS="${PERSONAVOICE_ENDPOINTING_GRACE_MS:-2500}"

# --- preflight (cheap checks only; the one-time work lives in setup-mac.sh) -----------------
docker info >/dev/null 2>&1 || { echo "Docker is not running. Start Docker Desktop and retry."; exit 1; }

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
# No --build here: setup-mac.sh builds the token-server image.
echo "[1/2] Starting LiveKit SFU + token server..."
docker compose -f docker-compose.livekit.yml up -d

LM_URL="${LMSTUDIO_BASE_URL:-http://localhost:1234/v1}"
if ! curl -sf "${LM_URL}/models" >/dev/null 2>&1; then
    echo "WARNING: LM Studio not reachable at ${LM_URL} - start it and load a model, or the LLM warm-up will wait."
fi

echo "[2/2] Starting the conversation worker (BACKEND=mac). Press Ctrl+C to stop everything."
echo "On the phone app, set the token server URL to:  http://${HOST_IP}:8080"
echo
"$PY" -m personavoice.server --serve

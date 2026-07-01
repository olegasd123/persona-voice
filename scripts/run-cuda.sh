#!/usr/bin/env bash
# Persona-Voice - start the PRODUCTION (CUDA) stack on Linux and run the live agent.
#
# The Linux/CUDA counterpart to run-cuda.ps1. vLLM runs in Docker; the conversation worker runs
# on the HOST (faster-whisper STT + Chatterbox TTS on the same GPU), so both share the one card.
#
# This is the FAST path: it assumes setup-cuda.sh has already done the one-time work (venv +
# extras, pulled/built the Docker images, pre-downloaded the host STT/TTS weights). On a fresh
# checkout run setup once:
#
#   ./scripts/setup-cuda.sh
#
# Then start the stack:
#   ./scripts/run-cuda.sh            # auto-detect VRAM and pick the tier
#   ./scripts/run-cuda.sh 32         # force the 32 GB unquantized profile
#   ./scripts/run-cuda.sh 16         # force the 16 GB AWQ (4-bit) profile
#
# Supported VRAM tiers: 12 / 16 / 24 / 32 GB (see README "Run on CUDA"). What it does:
#   1. picks a VRAM-tier profile (model + VRAM fraction + context length),
#   2. starts vLLM (the CUDA LLM server) in Docker,
#   3. starts the LiveKit SFU + token server in Docker,
#   4. waits for vLLM to be healthy, then
#   5. runs the conversation worker in the FOREGROUND.
#
# Press Ctrl+C to stop: the worker exits and every container is torn down (trap).
#
# Overrides (env vars): PV_LIVEKIT_IP forces the LAN IP the phone dials; LIVEKIT_API_KEY /
# LIVEKIT_API_SECRET override the dev keys; PV_VLLM_TIMEOUT bounds the vLLM health wait (s).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- helpers -------------------------------------------------------------------------------
find_lan_ip() {
    # IPv4 the phone dials: src address of the default route, else the first non-loopback addr.
    if [[ -n "${PV_LIVEKIT_IP:-}" ]]; then echo "$PV_LIVEKIT_IP"; return; fi
    local ip=""
    # `|| true`: with `set -o pipefail` a missing `ip`/route would otherwise abort the script.
    ip="$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}' || true)"
    [[ -n "$ip" ]] || ip="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
    echo "$ip"
}

resolve_vram_profile() {
    # Total VRAM (MiB) of the first CUDA GPU; map to the largest tier that fits. Cards report a
    # little under nominal (a 16 GB card ~16376 MiB), so thresholds sit below each round number.
    # The warning goes to stderr so it does not pollute the tier captured by $(...).
    local raw mib
    # `|| true`: with `set -o pipefail` a missing nvidia-smi would otherwise abort the script.
    raw="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 || true)"
    mib="$(printf '%s' "$raw" | tr -cd '0-9')"
    [[ -z "$mib" ]] && mib=0
    if   (( mib >= 30000 )); then echo 32
    elif (( mib >= 22000 )); then echo 24
    elif (( mib >= 15000 )); then echo 16
    elif (( mib >= 11000 )); then echo 12
    else
        echo "WARNING: could not read VRAM (got '$raw'); defaulting to the 12 GB floor profile (may OOM)." >&2
        echo 12
    fi
}

wait_vllm() {
    local timeout="${1:-600}" deadline
    echo "Waiting for vLLM at http://localhost:8000/health (first run downloads the model; later runs just load it into VRAM)..."
    deadline=$(( $(date +%s) + timeout ))
    while (( $(date +%s) < deadline )); do
        if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
            echo "vLLM is ready."
            return 0
        fi
        sleep 3
    done
    echo "WARNING: vLLM not healthy after ${timeout}s; starting the worker anyway (it retries the LLM warm-up)."
}

# --- pick the VRAM-tier profile ------------------------------------------------------------
# Keep these rows in sync with the tier table in docker-compose.yml / README "Run on CUDA".
VRAM="${1:-auto}"
[[ "$VRAM" == "auto" ]] && VRAM="$(resolve_vram_profile)"
case "$VRAM" in
    32) VLLM_MODEL="Qwen/Qwen2.5-7B-Instruct";     VLLM_GPU_UTIL="0.72"; VLLM_MAX_LEN="16384" ;;  # unquantized 7B, big context
    24) VLLM_MODEL="Qwen/Qwen2.5-7B-Instruct";     VLLM_GPU_UTIL="0.65"; VLLM_MAX_LEN="8192"  ;;  # unquantized 7B
    16) VLLM_MODEL="Qwen/Qwen2.5-7B-Instruct-AWQ"; VLLM_GPU_UTIL="0.45"; VLLM_MAX_LEN="8192"  ;;  # 4-bit AWQ (default tier)
    12) VLLM_MODEL="Qwen/Qwen2.5-7B-Instruct-AWQ"; VLLM_GPU_UTIL="0.55"; VLLM_MAX_LEN="4096"  ;;  # 4-bit AWQ, floor
    *)  echo "Invalid VRAM tier '$VRAM' (use: auto | 12 | 16 | 24 | 32)."; exit 1 ;;
esac

# --- venv python (host worker: STT + TTS on the GPU) ---------------------------------------
if [[ -x ".venv312/bin/python" ]]; then
    PY=".venv312/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PY="python"
else
    echo "No .venv312 found. Run the one-time setup first:  ./scripts/setup-cuda.sh" >&2
    exit 1
fi

# --- LAN IP the phone dials (dynamic, override with PV_LIVEKIT_IP) --------------------------
HOST_IP="$(find_lan_ip)"
[[ -n "$HOST_IP" ]] || { echo "Could not determine a LAN IP. Set PV_LIVEKIT_IP=<addr> (the address your phone dials)."; exit 1; }

# --- environment ---------------------------------------------------------------------------
export LIVEKIT_API_KEY="${LIVEKIT_API_KEY:-devkey}"      # dev keys; set real ones for a real deployment
export LIVEKIT_API_SECRET="${LIVEKIT_API_SECRET:-secret}"
export LIVEKIT_URL="ws://${HOST_IP}:7880"

export BACKEND=cuda
export VLLM_MODEL VLLM_GPU_UTIL VLLM_MAX_LEN         # interpolated by docker-compose.yml for vLLM
export PERSONAVOICE_LLM_ADAPTER=vllm
export PERSONAVOICE_LLM_MODEL="$VLLM_MODEL"          # host worker must expect the served id
export PERSONAVOICE_LLM_BASE_URL="http://localhost:8000/v1"
export PERSONAVOICE_STT_DEVICE=cuda
export PERSONAVOICE_STT_COMPUTE=float16
export PERSONAVOICE_TTS_ADAPTER=chatterbox
export PERSONAVOICE_TTS_DEVICE=cuda

export PERSONAVOICE_LOG_LEVEL="${PERSONAVOICE_LOG_LEVEL:-TRACE}"
export PERSONAVOICE_DYNAMIC_EMOTION="${PERSONAVOICE_DYNAMIC_EMOTION:-true}"
export PERSONAVOICE_TOOL_MAX_ITERS="${PERSONAVOICE_TOOL_MAX_ITERS:-4}"
export PERSONAVOICE_TOOL_TIMEOUT="${PERSONAVOICE_TOOL_TIMEOUT:-10}"
export PERSONAVOICE_SEMANTIC_ENDPOINTING="${PERSONAVOICE_SEMANTIC_ENDPOINTING:-1}"
export PERSONAVOICE_ENDPOINTING_GRACE_MS="${PERSONAVOICE_ENDPOINTING_GRACE_MS:-2500}"
export PERSONAVOICE_BUSY_RETRY_AFTER="${PERSONAVOICE_BUSY_RETRY_AFTER:-10}"
export PERSONAVOICE_ADMISSION_503="${PERSONAVOICE_ADMISSION_503:-1}"

# Single shared GPU, one serialized session per worker: run every job in the worker process
# (THREAD) so the models prewarmed at startup are reused across calls. LiveKit's default on
# Linux/CUDA is PROCESS, which reloads STT+TTS+VAD per job (cold start every call, and a warm
# spare holds a second copy in VRAM). THREAD collapses that to one warm copy — the right shape
# for the 1-session-per-worker design. See _job_executor_type.
export PERSONAVOICE_JOB_EXECUTOR="${PERSONAVOICE_JOB_EXECUTOR:-thread}"

# Prometheus /metrics: the worker writes its per-turn counters here and the token-server
# container mounts + reads the same dir (see docker-compose.livekit.yml), so
# http://<host>:8080/metrics aggregates them. Absolute path so it matches the compose mount
# regardless of cwd, and exported BEFORE the worker starts (prometheus_client selects
# multi-process mode at import). The same var feeds the compose mount source, so both sides land
# on this dir.
export PROMETHEUS_MULTIPROC_DIR="${PROMETHEUS_MULTIPROC_DIR:-$REPO_ROOT/models/metrics}"
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

# Custom personas are a single JSON file shared between the dockerized token-server (writer) and
# this native worker (reader) — see the bind mount in docker-compose.livekit.yml. Seed an empty
# manifest if it's missing so Docker mounts a FILE, not an auto-created directory (which would
# break the store's file writes). Both sides default to models/user_personas.json.
USER_PERSONAS_FILE="${PERSONAVOICE_USER_PERSONAS:-$REPO_ROOT/models/user_personas.json}"
mkdir -p "$(dirname "$USER_PERSONAS_FILE")"
[[ -f "$USER_PERSONAS_FILE" ]] || printf '{"users": {}}\n' > "$USER_PERSONAS_FILE"

# --- preflight (cheap checks only; the one-time work lives in setup-cuda.sh) ----------------
docker info >/dev/null 2>&1 || { echo "Docker is not running. Start the Docker daemon and retry."; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || echo "WARNING: nvidia-smi not found — is the NVIDIA driver installed? vLLM needs a CUDA GPU."

# --- teardown on exit (Ctrl+C) -------------------------------------------------------------
_cleaned=0
cleanup() {
    [[ "$_cleaned" == 1 ]] && return
    _cleaned=1
    echo
    echo "Stopping Persona-Voice stack..."
    docker compose -f docker-compose.livekit.yml down
    docker compose down
    echo "Stopped."
}
trap cleanup INT TERM EXIT

# --- run -----------------------------------------------------------------------------------
echo "VRAM profile : ${VRAM} GB  (model=$VLLM_MODEL, gpu_util=$VLLM_GPU_UTIL, max_len=$VLLM_MAX_LEN)"
echo "LAN IP       : $HOST_IP"

echo
echo "[1/3] Starting vLLM (CUDA LLM server)..."
docker compose up -d vllm

echo "[2/3] Starting LiveKit SFU + token server..."
docker compose -f docker-compose.livekit.yml up -d \
    || { echo "Failed to start LiveKit / token server. If the image is missing, run ./scripts/setup-cuda.sh."; exit 1; }

wait_vllm "${PV_VLLM_TIMEOUT:-600}"

echo
echo "[3/3] Starting the conversation worker. Press Ctrl+C to stop everything."
echo "On the phone app, set the token server URL to:  http://${HOST_IP}:8080"
echo
"$PY" -m personavoice.server --serve

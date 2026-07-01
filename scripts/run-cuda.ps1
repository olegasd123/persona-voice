<#
  Persona-Voice - start the PRODUCTION (CUDA) stack on Windows and run the live agent.

  This is the FAST path: it assumes setup-cuda.ps1 has already done the one-time work
  (installed deps, pulled/built the Docker images, pre-downloaded the host STT/TTS weights).
  On first run, or after a fresh checkout, run setup-cuda once:

    .\scripts\setup-cuda.ps1

  Then start the stack (paths resolve relative to this script):
    .\scripts\run-cuda.ps1            # auto-detect VRAM and pick the tier
    .\scripts\run-cuda.ps1 -Vram 32   # force the 32 GB unquantized profile
    .\scripts\run-cuda.ps1 -Vram 16   # force the 16 GB AWQ (4-bit) profile

  Supported VRAM tiers: 12 / 16 / 24 / 32 GB (see README "Run on CUDA").

  What it does (one command = the whole "[1]" sequence):
    1. picks a VRAM-tier profile (model + VRAM fraction + context length),
    2. starts vLLM (the CUDA LLM server) in Docker,
    3. starts the LiveKit SFU + token server in Docker,
    4. waits for vLLM to be healthy, then
    5. runs the conversation worker in the FOREGROUND.

  Press Ctrl+C to stop: the worker exits and every container is torn down (the "[2]"
  sequence) by the finally{} block. If a run dies without cleanup, run stop-cuda.ps1.

  Overrides (env vars): PV_LIVEKIT_IP forces the LAN IP the phone dials; LIVEKIT_API_KEY /
  LIVEKIT_API_SECRET override the dev keys.
#>
[CmdletBinding()]
param(
    [ValidateSet('auto', '12', '16', '24', '32')]
    [string]$Vram = 'auto',
    [string]$HostIp = $env:PV_LIVEKIT_IP,
    [int]$VllmTimeoutSec = 600
)

$ErrorActionPreference = 'Stop'

# --- paths: everything is relative to this script ------------------------------------------
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$Py = Join-Path $RepoRoot '.venv312\Scripts\python.exe'

# --- helpers -------------------------------------------------------------------------------
function Find-LanIp {
    # IPv4 of the adapter that owns the default route (the active Wi-Fi / Ethernet link).
    $cfg = Get-NetIPConfiguration |
        Where-Object { $_.IPv4DefaultGateway -and $_.NetAdapter.Status -eq 'Up' } |
        Select-Object -First 1
    if ($cfg) { return $cfg.IPv4Address.IPAddress }
    # Fallback: first routable IPv4 (skip loopback + APIPA).
    return (Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } |
        Select-Object -First 1).IPAddress
}

function Resolve-VramProfile {
    # Total VRAM (MiB) of the first CUDA GPU; map to the largest tier that fits. Cards report a
    # little under nominal (a 16 GB card ~16376 MiB), so thresholds sit below each round number.
    $raw = $null
    try {
        $raw = nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null |
            Select-Object -First 1
    } catch { }
    $digits = "$raw" -replace '[^\d]', ''
    if (-not $digits) { $digits = '0' }
    $mib = [int]$digits
    if ($mib -ge 30000) { return '32' }
    if ($mib -ge 22000) { return '24' }
    if ($mib -ge 15000) { return '16' }
    if ($mib -ge 11000) { return '12' }
    Write-Warning "Could not read VRAM (got '$raw'); defaulting to the 12 GB floor profile (may OOM)."
    return '12'
}

function Wait-Vllm {
    param([int]$TimeoutSec)
    Write-Host 'Waiting for vLLM at http://localhost:8000/health (first run downloads the model; later runs just load it into VRAM)...' -ForegroundColor Cyan
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri 'http://localhost:8000/health' -UseBasicParsing -TimeoutSec 5
            if ($r.StatusCode -eq 200) { Write-Host 'vLLM is ready.' -ForegroundColor Green; return }
        } catch { }
        Start-Sleep -Seconds 3
    }
    Write-Warning "vLLM not healthy after $TimeoutSec s; starting the worker anyway (it retries the LLM warm-up)."
}

$script:CleanedUp = $false
function Stop-Stack {
    if ($script:CleanedUp) { return }
    $script:CleanedUp = $true
    Write-Host "`nStopping Persona-Voice stack..." -ForegroundColor Yellow
    docker compose -f docker-compose.livekit.yml down
    docker compose down
    Write-Host 'Stopped.' -ForegroundColor Yellow
}

# --- pick the VRAM-tier profile ------------------------------------------------------------
# Keep these rows in sync with the tier table in docker-compose.yml / README "Run on CUDA".
if ($Vram -eq 'auto') { $Vram = Resolve-VramProfile }
switch ($Vram) {
    '32' { $VllmModel = 'Qwen/Qwen2.5-7B-Instruct';     $VllmGpuUtil = '0.72'; $VllmMaxLen = '16384' }  # unquantized 7B, big context
    '24' { $VllmModel = 'Qwen/Qwen2.5-7B-Instruct';     $VllmGpuUtil = '0.65'; $VllmMaxLen = '8192'  }  # unquantized 7B
    '16' { $VllmModel = 'Qwen/Qwen2.5-7B-Instruct-AWQ'; $VllmGpuUtil = '0.45'; $VllmMaxLen = '8192'  }  # 4-bit AWQ (default tier)
    '12' { $VllmModel = 'Qwen/Qwen2.5-7B-Instruct-AWQ'; $VllmGpuUtil = '0.55'; $VllmMaxLen = '4096'  }  # 4-bit AWQ, floor
}

# --- LAN IP (dynamic) ----------------------------------------------------------------------
if (-not $HostIp) { $HostIp = Find-LanIp }
if (-not $HostIp) { throw 'Could not determine a LAN IP. Pass -HostIp <addr> (the address your phone dials).' }

# --- environment (deterministic; wins over any local .env, which dotenv loads non-override) -
if (-not $env:LIVEKIT_API_KEY)    { $env:LIVEKIT_API_KEY = 'devkey' }     # dev keys; set real ones in prod
if (-not $env:LIVEKIT_API_SECRET) { $env:LIVEKIT_API_SECRET = 'secret' }
$env:LIVEKIT_URL = "ws://${HostIp}:7880"

$env:BACKEND = 'cuda'
$env:VLLM_MODEL = $VllmModel
$env:VLLM_GPU_UTIL = $VllmGpuUtil
$env:VLLM_MAX_LEN = $VllmMaxLen
$env:PERSONAVOICE_LLM_ADAPTER = 'vllm'
$env:PERSONAVOICE_LLM_MODEL = $VllmModel
$env:PERSONAVOICE_LLM_BASE_URL = 'http://localhost:8000/v1'
$env:PERSONAVOICE_STT_DEVICE = 'cuda'
$env:PERSONAVOICE_STT_COMPUTE = 'float16'
$env:PERSONAVOICE_TTS_ADAPTER = 'chatterbox'
$env:PERSONAVOICE_TTS_DEVICE = 'cuda'

$env:PERSONAVOICE_LOG_LEVEL = 'TRACE'
$env:PERSONAVOICE_DYNAMIC_EMOTION = 'true'
$env:PERSONAVOICE_TOOL_MAX_ITERS = '4'
$env:PERSONAVOICE_TOOL_TIMEOUT = '10'
$env:PERSONAVOICE_SEMANTIC_ENDPOINTING = '1'
$env:PERSONAVOICE_ENDPOINTING_GRACE_MS = '2500'
$env:PERSONAVOICE_BUSY_RETRY_AFTER = '10'
$env:PERSONAVOICE_ADMISSION_503 = '1'

# Single shared GPU, one serialized session per worker: run every job in the worker process
# (THREAD) so the models prewarmed at startup are reused across calls. LiveKit's off-Windows
# default (PROCESS) isolates each job in its own subprocess, and with num_idle_processes=1 the
# active job + the warm spare then hold TWO copies of STT+TTS+VAD in VRAM (~4-5 GB wasted) — the
# process-global _WARMED cache only collapses to one copy under THREAD. See _job_executor_type.
$env:PERSONAVOICE_JOB_EXECUTOR = 'thread'

# Prometheus /metrics: the worker writes its per-turn counters here and the token-server
# container mounts + reads the same dir (see docker-compose.livekit.yml), so
# http://<host>:8080/metrics aggregates them. Absolute path so it matches the compose mount, and
# set BEFORE the worker starts (prometheus_client selects multi-process mode at import). The same
# var feeds the compose mount source, so both sides land on this dir.
if (-not $env:PROMETHEUS_MULTIPROC_DIR) { $env:PROMETHEUS_MULTIPROC_DIR = Join-Path $RepoRoot 'models\metrics' }
New-Item -ItemType Directory -Force -Path $env:PROMETHEUS_MULTIPROC_DIR | Out-Null

# --- preflight (cheap checks only; the one-time work lives in setup-cuda.ps1) ---------------
$dockerOk = $false
try { docker info 2>$null | Out-Null; $dockerOk = ($LASTEXITCODE -eq 0) } catch { $dockerOk = $false }
if (-not $dockerOk) { throw 'Docker is not running. Start Docker Desktop and retry.' }

if (-not (Test-Path $Py)) {
    throw "Virtualenv not found at $Py.`n  Run the one-time setup first:  .\scripts\setup-cuda.ps1"
}

# --- run -----------------------------------------------------------------------------------
try {
    Write-Host "VRAM profile : ${Vram} GB  (model=$VllmModel, gpu_util=$VllmGpuUtil, max_len=$VllmMaxLen)" -ForegroundColor Green
    Write-Host "LAN IP       : $HostIp" -ForegroundColor Green

    Write-Host "`n[1/3] Starting vLLM (CUDA LLM server)..." -ForegroundColor Cyan
    docker compose up -d vllm
    if ($LASTEXITCODE -ne 0) { throw 'Failed to start vLLM.' }

    # No --build here: setup-cuda.ps1 builds the token-server image. Rebuilding it on every
    # run was the bulk of the per-start overhead.
    Write-Host '[2/3] Starting LiveKit SFU + token server...' -ForegroundColor Cyan
    docker compose -f docker-compose.livekit.yml up -d
    if ($LASTEXITCODE -ne 0) { throw 'Failed to start LiveKit / token server. If the image is missing, run .\scripts\setup-cuda.ps1.' }

    Wait-Vllm -TimeoutSec $VllmTimeoutSec

    Write-Host "`n[3/3] Starting the conversation worker. Press Ctrl+C to stop everything." -ForegroundColor Cyan
    Write-Host "On the iPhone app, set the token server URL to:  http://${HostIp}:8080`n" -ForegroundColor Green
    & $Py -m personavoice.server --serve
}
finally {
    Stop-Stack
}

<#
  Persona-Voice - start the PRODUCTION stack on Windows (CUDA) and run the live agent.

  Usage (from anywhere; paths are resolved relative to this script):
    .\scripts\run-prod-windows.ps1            # auto-detect the GPU (RTX 5090 / RTX 4080)
    .\scripts\run-prod-windows.ps1 -Gpu 5090  # force the 32 GB unquantized profile
    .\scripts\run-prod-windows.ps1 -Gpu 4080  # force the 16 GB AWQ (4-bit) profile

  What it does (one command = the whole "[1]" sequence):
    1. picks a GPU profile (model + VRAM fraction),
    2. starts vLLM (the CUDA LLM server) in Docker,
    3. starts the LiveKit SFU + token server in Docker,
    4. waits for vLLM to be healthy, then
    5. runs the conversation worker in the FOREGROUND.

  Press Ctrl+C to stop: the worker exits and every container is torn down (the "[2]"
  sequence) by the finally{} block. If a run dies without cleanup, run stop-prod-windows.ps1.

  Works on a fresh machine (Docker + a .venv312 present, but no images yet): the LiveKit
  token-server image is built and the vLLM image is pulled on first run.

  Overrides (env vars): PV_LIVEKIT_IP forces the LAN IP the phone dials; LIVEKIT_API_KEY /
  LIVEKIT_API_SECRET override the dev keys; -NoInstall skips the dependency check.
#>
[CmdletBinding()]
param(
    [ValidateSet('auto', '5090', '4080')]
    [string]$Gpu = 'auto',
    [string]$HostIp = $env:PV_LIVEKIT_IP,
    [int]$VllmTimeoutSec = 600,
    [switch]$NoInstall
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

function Resolve-GpuProfile {
    $name = $null
    try { $name = nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1 } catch { }
    if ($name -match '5090') { return '5090' }
    if ($name -match '4080') { return '4080' }
    Write-Warning "GPU '$name' did not match a profile; defaulting to the 16 GB 4080-safe profile."
    return '4080'
}

function Wait-Vllm {
    param([int]$TimeoutSec)
    Write-Host "Waiting for vLLM at http://localhost:8000/health (first run downloads the model)..." -ForegroundColor Cyan
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

# --- pick the GPU profile ------------------------------------------------------------------
if ($Gpu -eq 'auto') { $Gpu = Resolve-GpuProfile }
switch ($Gpu) {
    '5090' { $VllmModel = 'Qwen/Qwen2.5-7B-Instruct';     $VllmGpuUtil = '0.6'  }  # unquantized, 32 GB
    '4080' { $VllmModel = 'Qwen/Qwen2.5-7B-Instruct-AWQ'; $VllmGpuUtil = '0.45' }  # 4-bit AWQ, 16 GB
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
$env:PERSONAVOICE_LLM_ADAPTER = 'vllm'
$env:PERSONAVOICE_LLM_MODEL = $VllmModel
$env:PERSONAVOICE_LLM_BASE_URL = 'http://localhost:8000/v1'
$env:PERSONAVOICE_STT_DEVICE = 'cuda'
$env:PERSONAVOICE_STT_COMPUTE = 'float16'
$env:PERSONAVOICE_TTS_ADAPTER = 'chatterbox'   # Windows-native; Orpheus needs its own in-process vLLM
$env:PERSONAVOICE_TTS_DEVICE = 'cuda'

# --- preflight -----------------------------------------------------------------------------
$dockerOk = $false
try { docker info 2>$null | Out-Null; $dockerOk = ($LASTEXITCODE -eq 0) } catch { $dockerOk = $false }
if (-not $dockerOk) { throw 'Docker is not running. Start Docker Desktop and retry.' }

if (-not (Test-Path $Py)) {
    throw "Virtualenv not found at $Py.`n  Create it:  py -3.12 -m venv .venv312`n  Install:    .\.venv312\Scripts\python.exe -m pip install -e `".[cuda,livekit]`""
}
if (-not $NoInstall) {
    $haveLivekit = $false
    try { & $Py -c 'import livekit.agents' 2>$null; $haveLivekit = ($LASTEXITCODE -eq 0) } catch { $haveLivekit = $false }
    if (-not $haveLivekit) {
        Write-Host "Installing the 'livekit' extra (one-time)..." -ForegroundColor Cyan
        & $Py -m pip install -e '.[livekit]'
        if ($LASTEXITCODE -ne 0) { throw 'pip install -e .[livekit] failed.' }
    }
}

# --- run -----------------------------------------------------------------------------------
try {
    Write-Host "GPU profile : $Gpu  (model=$VllmModel, gpu_util=$VllmGpuUtil)" -ForegroundColor Green
    Write-Host "LAN IP      : $HostIp" -ForegroundColor Green

    Write-Host "`n[1/3] Starting vLLM (CUDA LLM server)..." -ForegroundColor Cyan
    docker compose up -d vllm
    if ($LASTEXITCODE -ne 0) { throw 'Failed to start vLLM.' }

    Write-Host '[2/3] Starting LiveKit SFU + token server...' -ForegroundColor Cyan
    docker compose -f docker-compose.livekit.yml up -d --build
    if ($LASTEXITCODE -ne 0) { throw 'Failed to start LiveKit / token server.' }

    Wait-Vllm -TimeoutSec $VllmTimeoutSec

    Write-Host "`n[3/3] Starting the conversation worker. Press Ctrl+C to stop everything." -ForegroundColor Cyan
    Write-Host "On the iPhone app, set the token server URL to:  http://${HostIp}:8080`n" -ForegroundColor Green
    & $Py -m personavoice.server --serve
}
finally {
    Stop-Stack
}

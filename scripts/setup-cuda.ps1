<#
  Persona-Voice - ONE-TIME setup for the production (CUDA) stack on Windows.

  Run this once on a fresh machine (or after a dependency / image change). It does everything
  that is slow-but-cacheable, so the everyday `run-cuda.ps1` starts fast:

    1. creates .venv312 (if missing) and installs the cuda + livekit extras,
    2. pulls the vLLM and LiveKit Docker images,
    3. builds the persona-voice token-server image,
    4. primes the vLLM model into the shared hf-cache volume (downloads ~5-15 GB once),
    5. pre-downloads the host-side STT (faster-whisper) + TTS (Chatterbox) weights.

  Usage (paths resolve relative to this script):
    .\scripts\setup-cuda.ps1            # auto-detect the GPU (RTX 5090 / RTX 4080)
    .\scripts\setup-cuda.ps1 -Gpu 5090  # prime the 32 GB unquantized model
    .\scripts\setup-cuda.ps1 -Gpu 4080  # prime the 16 GB AWQ (4-bit) model
    .\scripts\setup-cuda.ps1 -SkipModelPrime   # deps + images only, don't download models

  After this, start the stack with: .\scripts\run-cuda.ps1
#>
[CmdletBinding()]
param(
    [ValidateSet('auto', '5090', '4080')]
    [string]$Gpu = 'auto',
    [int]$VllmTimeoutSec = 900,
    [switch]$SkipModelPrime
)

$ErrorActionPreference = 'Stop'

# --- paths: everything is relative to this script ------------------------------------------
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$Py = Join-Path $RepoRoot '.venv312\Scripts\python.exe'

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
    Write-Host 'Waiting for vLLM at http://localhost:8000/health (first run downloads the model)...' -ForegroundColor Cyan
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri 'http://localhost:8000/health' -UseBasicParsing -TimeoutSec 5
            if ($r.StatusCode -eq 200) { Write-Host 'vLLM model cached and ready.' -ForegroundColor Green; return $true }
        } catch { }
        Start-Sleep -Seconds 3
    }
    Write-Warning "vLLM not healthy after $TimeoutSec s; the model may still be downloading. Re-run setup-cuda or just start run-cuda."
    return $false
}

# --- preflight -----------------------------------------------------------------------------
$dockerOk = $false
try { docker info 2>$null | Out-Null; $dockerOk = ($LASTEXITCODE -eq 0) } catch { $dockerOk = $false }
if (-not $dockerOk) { throw 'Docker is not running. Start Docker Desktop and retry.' }

# --- [1/5] virtualenv + Python deps --------------------------------------------------------
Write-Host "`n[1/5] Python environment (.venv312)..." -ForegroundColor Cyan
if (-not (Test-Path $Py)) {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "Python launcher 'py' not found. Install Python 3.12, then re-run.`n  Or create the venv manually:  py -3.12 -m venv .venv312"
    }
    Write-Host '  Creating .venv312 (Python 3.12)...' -ForegroundColor Gray
    py -3.12 -m venv .venv312
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create .venv312.' }
}
Write-Host '  Installing the cuda + livekit extras (this can take a while)...' -ForegroundColor Gray
& $Py -m pip install -e '.[cuda,livekit]'
if ($LASTEXITCODE -ne 0) { throw 'pip install -e .[cuda,livekit] failed.' }

# --- [2/5] pull Docker images --------------------------------------------------------------
Write-Host "`n[2/5] Pulling Docker images (vLLM + LiveKit)..." -ForegroundColor Cyan
docker compose pull vllm
if ($LASTEXITCODE -ne 0) { throw 'Failed to pull the vLLM image.' }
docker compose -f docker-compose.livekit.yml pull livekit
if ($LASTEXITCODE -ne 0) { throw 'Failed to pull the LiveKit image.' }

# --- [3/5] build the token-server image ----------------------------------------------------
Write-Host "`n[3/5] Building the persona-voice token-server image..." -ForegroundColor Cyan
docker compose -f docker-compose.livekit.yml build
if ($LASTEXITCODE -ne 0) { throw 'Failed to build the token-server image.' }

# --- [4/5] prime the vLLM model into the hf-cache volume -----------------------------------
if ($SkipModelPrime) {
    Write-Host "`n[4/5] Skipping model prime (-SkipModelPrime). The model downloads on first run-cuda." -ForegroundColor Yellow
} else {
    if ($Gpu -eq 'auto') { $Gpu = Resolve-GpuProfile }
    switch ($Gpu) {
        '5090' { $env:VLLM_MODEL = 'Qwen/Qwen2.5-7B-Instruct';     $env:VLLM_GPU_UTIL = '0.6'  }
        '4080' { $env:VLLM_MODEL = 'Qwen/Qwen2.5-7B-Instruct-AWQ'; $env:VLLM_GPU_UTIL = '0.45' }
    }
    Write-Host "`n[4/5] Priming the vLLM model into the cache: $($env:VLLM_MODEL)" -ForegroundColor Cyan
    Write-Host '  (downloads once into the persistent hf-cache volume, then shuts vLLM back down)' -ForegroundColor Gray
    try {
        docker compose up -d vllm
        if ($LASTEXITCODE -ne 0) { throw 'Failed to start vLLM for model priming.' }
        Wait-Vllm -TimeoutSec $VllmTimeoutSec | Out-Null
    }
    finally {
        docker compose down | Out-Null
    }
}

# --- [5/5] pre-download the host-side STT/TTS weights --------------------------------------
Write-Host "`n[5/5] Pre-downloading host STT + TTS weights (best-effort)..." -ForegroundColor Cyan
& $Py (Join-Path $PSScriptRoot 'prefetch_host_models.py')

Write-Host "`nSetup complete. Start the stack with:  .\scripts\run-cuda.ps1" -ForegroundColor Green

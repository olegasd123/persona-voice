<#
  Persona-Voice - ONE-TIME setup for the production (CUDA) stack on Windows.

  Run this once on a fresh machine (or after a dependency / image change). It does the
  slow-but-cacheable prep so the everyday `run-cuda.ps1` starts fast - WITHOUT starting any
  server:

    1. creates .venv312 (if missing) and installs the cuda + livekit + clone (Chatterbox) extras,
    2. pulls the vLLM and LiveKit Docker images,
    3. builds the persona-voice token-server image,
    4. pre-downloads the host-side STT (faster-whisper) + TTS (Chatterbox) weights.

  The vLLM LLM model is NOT downloaded here: it caches into the persistent hf-cache volume on
  the first `run-cuda` (when vLLM actually starts). That keeps setup free of any running server.

  The CUDA worker runs faster-whisper (STT) + Chatterbox (TTS) on the host GPU, so the `clone`
  extra (Chatterbox) is installed here, not only in the server image.

  Usage (paths resolve relative to this script):
    .\scripts\setup-cuda.ps1

  After this, start the stack with: .\scripts\run-cuda.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

# --- paths: everything is relative to this script ------------------------------------------
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$Py = Join-Path $RepoRoot '.venv312\Scripts\python.exe'

# --- preflight -----------------------------------------------------------------------------
$dockerOk = $false
try { docker info 2>$null | Out-Null; $dockerOk = ($LASTEXITCODE -eq 0) } catch { $dockerOk = $false }
if (-not $dockerOk) { throw 'Docker is not running. Start Docker Desktop and retry.' }

# --- [1/4] virtualenv + Python deps --------------------------------------------------------
Write-Host "`n[1/4] Python environment (.venv312)..." -ForegroundColor Cyan
if (-not (Test-Path $Py)) {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "Python launcher 'py' not found. Install Python 3.12, then re-run.`n  Or create the venv manually:  py -3.12 -m venv .venv312"
    }
    Write-Host '  Creating .venv312 (Python 3.12)...' -ForegroundColor Gray
    py -3.12 -m venv .venv312
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create .venv312.' }
}
Write-Host '  Installing the cuda + livekit + clone extras (this can take a while)...' -ForegroundColor Gray
& $Py -m pip install -e '.[cuda,livekit,clone]'
if ($LASTEXITCODE -ne 0) { throw 'pip install -e .[cuda,livekit,clone] failed.' }

# --- [2/4] pull Docker images --------------------------------------------------------------
Write-Host "`n[2/4] Pulling Docker images (vLLM + LiveKit)..." -ForegroundColor Cyan
docker compose pull vllm
if ($LASTEXITCODE -ne 0) { throw 'Failed to pull the vLLM image.' }
docker compose -f docker-compose.livekit.yml pull livekit
if ($LASTEXITCODE -ne 0) { throw 'Failed to pull the LiveKit image.' }

# --- [3/4] build the token-server image ----------------------------------------------------
Write-Host "`n[3/4] Building the persona-voice token-server image..." -ForegroundColor Cyan
docker compose -f docker-compose.livekit.yml build
if ($LASTEXITCODE -ne 0) { throw 'Failed to build the token-server image.' }

# --- [4/4] pre-download the host-side STT/TTS weights --------------------------------------
Write-Host "`n[4/4] Pre-downloading host STT + TTS weights (best-effort)..." -ForegroundColor Cyan
& $Py (Join-Path $PSScriptRoot 'prefetch_host_models.py')

Write-Host "`nSetup complete. Start the stack with:  .\scripts\run-cuda.ps1" -ForegroundColor Green
Write-Host '  (the first run downloads the vLLM model into the cache; later runs are fast)' -ForegroundColor Gray

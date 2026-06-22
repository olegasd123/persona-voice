<#
  Persona-Voice - stop the production stack on Windows.

  Normally Ctrl+C in run-cuda.ps1 tears everything down. Use this as a fallback when a
  run was killed without cleanup (closed window, crash), or to stop containers started by hand.
#>
$ErrorActionPreference = 'Continue'

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host 'Stopping Persona-Voice stack...' -ForegroundColor Yellow
docker compose -f docker-compose.livekit.yml down
docker compose down
Write-Host 'Stopped.' -ForegroundColor Yellow

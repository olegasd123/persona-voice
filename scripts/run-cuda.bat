@echo off
REM Persona-Voice - start the production stack (Windows / CUDA).
REM Thin wrapper so the script is double-clickable; the real logic lives in the .ps1 next to it
REM (PowerShell gives reliable Ctrl+C -> teardown, which a plain .bat cannot).
REM
REM Run setup-cuda.bat once on a fresh machine first; then:
REM   run-cuda.bat            auto-detect VRAM and pick the tier
REM   run-cuda.bat 32         force the 32 GB unquantized profile
REM   run-cuda.bat 16         force the 16 GB AWQ profile (tiers: 12 / 16 / 24 / 32)
REM
REM Press Ctrl+C to stop the stack. (Answer N to any "Terminate batch job?" prompt - the
REM containers are already torn down by the PowerShell finally block at that point.)
setlocal
if "%~1"=="" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-cuda.ps1"
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-cuda.ps1" -Vram %~1
)
endlocal

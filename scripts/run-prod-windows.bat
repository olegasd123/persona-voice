@echo off
REM Persona-Voice - start the production stack (Windows / CUDA).
REM Thin wrapper so the script is double-clickable; the real logic lives in the .ps1 next to it
REM (PowerShell gives reliable Ctrl+C -> teardown, which a plain .bat cannot).
REM
REM   run-prod-windows.bat            auto-detect the GPU (RTX 5090 / RTX 4080)
REM   run-prod-windows.bat 5090       force the 32 GB unquantized profile
REM   run-prod-windows.bat 4080       force the 16 GB AWQ profile
REM
REM Press Ctrl+C to stop the stack. (Answer N to any "Terminate batch job?" prompt - the
REM containers are already torn down by the PowerShell finally block at that point.)
setlocal
if "%~1"=="" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-prod-windows.ps1"
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-prod-windows.ps1" -Gpu %~1
)
endlocal

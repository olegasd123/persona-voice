@echo off
REM Persona-Voice - ONE-TIME setup for the production stack (Windows / CUDA).
REM Thin wrapper so the script is double-clickable; the real logic lives in the .ps1 next to it.
REM
REM Run this once on a fresh machine (installs deps, pulls/builds images, primes models):
REM   setup-cuda.bat            auto-detect the GPU (RTX 5090 / RTX 4080)
REM   setup-cuda.bat 5090       prime the 32 GB unquantized model
REM   setup-cuda.bat 4080       prime the 16 GB AWQ model
REM
REM Then start the stack with run-cuda.bat.
setlocal
if "%~1"=="" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-cuda.ps1"
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-cuda.ps1" -Gpu %~1
)
endlocal

@echo off
REM Persona-Voice - ONE-TIME setup for the production stack (Windows / CUDA).
REM Thin wrapper so the script is double-clickable; the real logic lives in the .ps1 next to it.
REM
REM Run this once on a fresh machine. It installs deps, pulls/builds the Docker images, and
REM pre-downloads the host STT/TTS weights. No server is started here - the vLLM model caches
REM on the first run-cuda.
REM
REM Then start the stack with run-cuda.bat.
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-cuda.ps1"
endlocal

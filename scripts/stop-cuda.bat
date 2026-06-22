@echo off
REM Persona-Voice - stop the production stack (Windows). Fallback for when a run was killed
REM without running its own cleanup. Wraps stop-cuda.ps1.
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop-cuda.ps1"
endlocal

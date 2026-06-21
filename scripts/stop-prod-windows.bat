@echo off
REM Persona-Voice - stop the production stack (Windows). Fallback for when a run was killed
REM without running its own cleanup. Wraps stop-prod-windows.ps1.
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop-prod-windows.ps1"
endlocal

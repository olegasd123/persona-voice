# Run / stop scripts

One command starts the whole stack and runs the live conversation worker in the foreground.
**Press `Ctrl+C` to stop** — the worker exits and every container it started is torn down.

| Platform | Start | Stop (fallback) |
|----------|-------|-----------------|
| Windows prod (CUDA) | `scripts\run-prod-windows.bat` (or `.ps1`) | `scripts\stop-prod-windows.bat` |
| macOS dev (M4 Max) | `scripts/run-dev-mac.sh` | `scripts/stop-dev-mac.sh` |

The stop scripts are only needed if a run was killed without cleanup (closed window / crash).

## Windows production

```powershell
.\scripts\run-prod-windows.ps1            # auto-detect the GPU
.\scripts\run-prod-windows.ps1 -Gpu 5090  # force the 32 GB profile
.\scripts\run-prod-windows.ps1 -Gpu 4080  # force the 16 GB profile
```

The GPU profile is auto-detected with `nvidia-smi` (override with `-Gpu`):

| Profile | vLLM model | `VLLM_GPU_UTIL` |
|---------|------------|-----------------|
| **5090** (32 GB) | `Qwen/Qwen2.5-7B-Instruct` (unquantized) | `0.6` |
| **4080** (16 GB) | `Qwen/Qwen2.5-7B-Instruct-AWQ` (4-bit) | `0.45` |

It starts vLLM + LiveKit + the token server in Docker, waits for vLLM to be healthy, then runs
the worker (`personavoice.server --serve`). TTS is forced to Chatterbox (Windows-native).

> The `.bat` files are thin wrappers around the `.ps1` scripts (PowerShell gives reliable
> `Ctrl+C` → teardown). Running the `.ps1` directly avoids the `Terminate batch job?` prompt.

## macOS development

```bash
./scripts/run-dev-mac.sh
```

Starts the LiveKit SFU + token server, then runs the worker with `BACKEND=mac`. The brain is
**LM Studio** (or Ollama) — start it and load a model first (the script warns if it is down). No
vLLM container is used on the Mac.

## What's automatic

- **Dynamic LAN IP** — the address your phone dials is detected from the default-route adapter.
  Override with `PV_LIVEKIT_IP=<addr>` (PowerShell: `-HostIp <addr>`).
- **Token server URL** for the phone app is printed on startup as `http://<LAN_IP>:8080`.
- **Fresh machine** — the LiveKit token-server image is built and the vLLM image pulled on first
  run; the `livekit` Python extra is installed if missing (skip with `-NoInstall` / `NO_INSTALL=1`).
- **LiveKit keys** default to the dev `devkey` / `secret`; override with `LIVEKIT_API_KEY` /
  `LIVEKIT_API_SECRET` for a real deployment.

## Prerequisites

- Docker Desktop running.
- The `.venv312` virtualenv present. To create it:
  - Windows: `py -3.12 -m venv .venv312 && .\.venv312\Scripts\python.exe -m pip install -e ".[cuda,livekit]"`
  - macOS: `uv venv --python 3.12 .venv312 && uv pip install -e '.[mac,livekit]'`

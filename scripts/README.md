# Setup / run / stop scripts

There are two phases. **`setup-*` is one-time** (installs deps, pulls + builds the Docker
images, pre-downloads the host model weights) — run it on a fresh machine or after a
dependency change. It does not start any server.
**`run-*` is every-run** — it starts the stack and the live conversation worker in the
foreground; **press `Ctrl+C` to stop** and every container it started is torn down.

To override environment variables use:

```powershell
$env:PERSONAVOICE_LOG_LEVEL = "TRACE"
```

```sh
export PERSONAVOICE_LOG_LEVEL="TRACE"
```

Splitting the two keeps `run-*` fast: it no longer rebuilds the token-server image or
re-checks/installs Python deps on every start.

| Platform | Setup (one-time) | Start | Stop (fallback) |
|----------|------------------|-------|-----------------|
| Windows prod (CUDA) | `scripts\setup-cuda.bat` (or `.ps1`) | `scripts\run-cuda.bat` (or `.ps1`) | `scripts\stop-cuda.bat` |
| macOS dev (M4 Max) | `scripts/setup-mac.sh` | `scripts/run-mac.sh` | `scripts/stop-mac.sh` |

The stop scripts are only needed if a run was killed without cleanup (closed window / crash).

## Windows production

```powershell
.\scripts\setup-cuda.ps1            # ONCE: deps + images + host STT/TTS weights (no server started)
.\scripts\run-cuda.ps1              # then, every run (auto-detect the GPU)
.\scripts\run-cuda.ps1 -Gpu 5090    # force the 32 GB profile
.\scripts\run-cuda.ps1 -Gpu 4080    # force the 16 GB profile
```

The GPU profile is auto-detected with `nvidia-smi` (override with `-Gpu`):

| Profile | vLLM model | `VLLM_GPU_UTIL` |
|---------|------------|-----------------|
| **5090** (32 GB) | `Qwen/Qwen2.5-7B-Instruct` (unquantized) | `0.6` |
| **4080** (16 GB) | `Qwen/Qwen2.5-7B-Instruct-AWQ` (4-bit) | `0.45` |

`run-cuda` starts vLLM + LiveKit + the token server in Docker, waits for vLLM to be healthy,
then runs the worker (`personavoice.server --serve`). TTS is forced to Chatterbox
(Windows-native). `setup-cuda` does not start any server: it installs deps, pulls/builds the
images, and pre-downloads the host STT/TTS weights. The vLLM model caches into the shared
`hf-cache` volume on the **first `run-cuda`** (when vLLM actually starts) and is reused after.

> The `.bat` files are thin wrappers around the `.ps1` scripts (PowerShell gives reliable
> `Ctrl+C` → teardown). Running the `.ps1` directly avoids the `Terminate batch job?` prompt.

## macOS development

```bash
./scripts/setup-mac.sh   # ONCE: deps + LiveKit image + token-server image
./scripts/run-mac.sh     # then, every run
```

`run-mac` starts the LiveKit SFU + token server, then runs the worker with `BACKEND=mac`. The
brain is **LM Studio** (or Ollama) — start it and load a model first (the script warns if it
is down). No vLLM container is used on the Mac. The host STT (whisper-mlx) + TTS (Kokoro)
weights download on the first run.

## What's automatic

- **Dynamic LAN IP** — the address your phone dials is detected from the default-route adapter.
  Override with `PV_LIVEKIT_IP=<addr>` (PowerShell: `-HostIp <addr>`).
- **Token server URL** for the phone app is printed on startup as `http://<LAN_IP>:8080`.
- **LiveKit keys** default to the dev `devkey` / `secret`; override with `LIVEKIT_API_KEY` /
  `LIVEKIT_API_SECRET` for a real deployment.

## Prerequisites

- Docker Desktop running.
- Python 3.12 (`setup-*` creates `.venv312` and installs the right extras for you; on Windows
  it uses the `py -3.12` launcher, on macOS it prefers `uv`, else `python3.12`).

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
| Windows (CUDA) | `scripts\setup-cuda.bat` (or `.ps1`) | `scripts\run-cuda.bat` (or `.ps1`) | `scripts\stop-cuda.bat` |
| Linux (CUDA) | `scripts/setup-cuda.sh` | `scripts/run-cuda.sh` | `scripts/stop-cuda.sh` |
| macOS (M4 Max) | `scripts/setup-mac.sh` | `scripts/run-mac.sh` | `scripts/stop-mac.sh` |

The stop scripts are only needed if a run was killed without cleanup (closed window / crash).

## Windows (CUDA)

```powershell
.\scripts\setup-cuda.ps1            # ONCE: deps + images + host STT/TTS weights (no server started)
.\scripts\run-cuda.ps1              # then, every run (auto-detect VRAM, pick the tier)
.\scripts\run-cuda.ps1 -Vram 32     # force the 32 GB profile
.\scripts\run-cuda.ps1 -Vram 16     # force the 16 GB profile
```

The VRAM tier is auto-detected from the card's `memory.total` via `nvidia-smi` (override with
`-Vram 12|16|24|32`):

| Tier | vLLM model | `VLLM_GPU_UTIL` | `VLLM_MAX_LEN` |
|------|------------|-----------------|----------------|
| **32 GB** | `Qwen/Qwen2.5-7B-Instruct` (unquantized) | `0.72` | `16384` |
| **24 GB** | `Qwen/Qwen2.5-7B-Instruct` (unquantized) | `0.65` | `8192` |
| **16 GB** | `Qwen/Qwen2.5-7B-Instruct-AWQ` (4-bit) | `0.45` | `8192` |
| **12 GB** | `Qwen/Qwen2.5-7B-Instruct-AWQ` (4-bit) | `0.55` | `4096` |

`run-cuda` starts vLLM + LiveKit + the token server in Docker, waits for vLLM to be healthy,
then runs the worker (`personavoice.server --serve`). TTS is forced to Chatterbox
(Windows-native). `setup-cuda` does not start any server: it installs deps, pulls/builds the
images, and pre-downloads the host STT/TTS weights. The vLLM model caches into the shared
`hf-cache` volume on the **first `run-cuda`** (when vLLM actually starts) and is reused after.

> The `.bat` files are thin wrappers around the `.ps1` scripts (PowerShell gives reliable
> `Ctrl+C` → teardown). Running the `.ps1` directly avoids the `Terminate batch job?` prompt.

## Linux (CUDA)

The Linux counterpart to the Windows `.ps1` scripts.

```bash
./scripts/setup-cuda.sh        # ONCE: venv + extras + Docker images + host STT/TTS weights
./scripts/run-cuda.sh          # then, every run (auto-detect VRAM, pick the tier)
./scripts/run-cuda.sh 32       # force the 32 GB profile (tiers: 12 | 16 | 24 | 32)
```

Same tier table as Windows above — the VRAM tier is auto-detected from the card's `memory.total`
via `nvidia-smi` (override with the positional arg). `setup-cuda` creates `.venv312`, installs the
`cuda,livekit,clone` extras (the worker runs faster-whisper + Chatterbox on the host GPU), pulls
the vLLM/LiveKit images, builds the token-server image, and pre-downloads the host STT/TTS
weights — without starting anything. `run-cuda` then starts vLLM + LiveKit + the token server in
Docker, waits for vLLM to be healthy, and runs the worker on the host. It sets
`PERSONAVOICE_JOB_EXECUTOR=thread` so the GPU models load once and stay warm across calls (the
Linux/CUDA default is per-job process isolation, i.e. a cold start every call). Ctrl+C tears down
every container it started (trap); `stop-cuda.sh` is the fallback if a run is killed uncleanly.
Prereqs: Docker + the NVIDIA Container Toolkit, and `uv` or `python3.12` for the venv.

## macOS (M4 Max)

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

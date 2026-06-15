# Persona-Voice

A real-time, open-source **speech-to-speech persona system** built as a modular cascade
(**STT → LLM brain → TTS**). It emulates people for practical use: a PM/HR interviewer for
practice, an English conversation teacher, or a casual companion.

It runs as a server on an **RTX 4080 (16 GB)** in production and is fully developable on a
**Mac M4 Max** via swappable Mac-native backends. The iPhone client is thin — it only
captures and plays audio.

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full design and milestones.

## Status

Current milestone: **M2 — Backend parity (Mac ↔ CUDA) ✅ done**. The CUDA cascade is now
implemented alongside the Mac one: `faster_whisper` / `parakeet` (STT), `vllm` (LLM,
OpenAI-compatible — it shares the streaming path with `lmstudio`), and `orpheus` /
`chatterbox` (TTS). A `docker-compose.yml` brings up the vLLM server plus the persona-voice
server on the 4080, and `scripts/bench_latency.py` reports per-stage timings on either
machine.

> The CUDA code is complete, lazy-imported, and unit-tested at the logic level on the Mac;
> the end-to-end run **on the 4080** is pending access to the GPU box.

**M1 (done):** the Mac backend runs end-to-end — `whisper_mlx` (STT), `lmstudio` / `ollama`
(LLM), `kokoro` (TTS) wired through a turn-based `pipeline` and the `personavoice-demo` CLI
(speak into a wav → hear a reply). Verified on an M4 Max at ~5.8 s/turn warm (STT 1.2 s ·
LLM 0.8 s · TTS 3.8 s).

## Quickstart

Requires Python 3.11+.

```bash
# 1. Create a virtualenv and install (core + dev tools).
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

# 2. Configure.
cp .env.example .env        # defaults to BACKEND=mac

# 3. Validate config + load (stub) adapters — the M0 acceptance test.
python -m personavoice.server --check
#   ...or: BACKEND=cuda python -m personavoice.server --check
#   ...or: ./scripts/dev_server.sh

# 4. Run the tests.
pytest
```

To install backend model libraries (heavier, platform-specific):

```bash
pip install -e '.[mac]'     # M4 Max: mlx-whisper, mlx-lm, kokoro, httpx, ...
pip install -e '.[cuda]'    # RTX 4080: faster-whisper, ...
```

## Offline voice loop (M1)

Run a single turn through the cascade on the Mac: a spoken `.wav` question in, a synthesized
spoken reply out. Requires the `mac` extra and a local LLM server.

> **Python 3.12 for the `mac` extra.** Kokoro pulls spaCy/`blis`, which has no wheels for
> Python 3.13/3.14 (it fails to compile). Use Python 3.12, e.g. with [uv](https://docs.astral.sh/uv/):
> ```bash
> uv venv --python 3.12 .venv312 && source .venv312/bin/activate
> uv pip install -e '.[mac]'
> ```

```bash
# LLM: start LM Studio's local server and load a model (default config expects
# openai/gpt-oss-20b). Confirm it's up:  curl http://localhost:1234/v1/models
# (Prefer Ollama? set adapter: ollama in config/backends/mac.yaml, then `ollama serve`.)

# Speak into a prerecorded wav and hear the reply:
personavoice-demo --wav question.wav --persona companion --play

# ...or record from the mic (needs sounddevice + a mic):
personavoice-demo --record 5 --persona language_teacher --play
```

It prints the transcript, the persona's reply, and per-stage timings, and writes the spoken
reply to `reply.wav` (override with `--out`). Streaming, barge-in, and the live LiveKit
server arrive in M3.

> **Reasoning models:** local models like Qwen3 / gpt-oss emit a hidden thinking trace
> (`reasoning_content`) that the adapter never speaks. Keep it short with
> `extra_body: {reasoning_effort: low}` (already set in `mac.yaml`) so replies start fast and
> don't exhaust `max_tokens` before producing any spoken content.

## Run on the 4080 (CUDA, M2)

The `cuda` backend mirrors the Mac cascade: `faster_whisper`/`parakeet` (STT), `vllm` (LLM),
`orpheus`/`chatterbox` (TTS). The LLM is served by a separate **vLLM** process so the app
image stays light; the STT/TTS models run in the persona-voice container.

```bash
# Bring up vLLM (Qwen2.5-7B-AWQ) + the persona-voice server on the GPU box.
docker compose up --build
#   vllm:        http://localhost:8000/v1   (OpenAI-compatible)
#   personavoice: validates the cuda config against vLLM (the live server lands in M3)
```

Both services share the single 4080. The compose file documents the VRAM budget (≈10–12 GB:
vLLM 4-bit ~6–7 GB + STT ~2 GB + Chatterbox ~2–3 GB, within 16 GB) and caps vLLM's
`--gpu-memory-utilization` so STT/TTS fit. Orpheus is more expressive but runs its own
in-process vLLM (tight on one card) — prefer **Chatterbox** (MIT, torch) for the single-GPU
stack by setting `adapter: chatterbox` in `config/backends/cuda.yaml`.

Without Docker, run the pieces directly: `vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ
--quantization awq`, then `BACKEND=cuda personavoice-demo --wav question.wav` (needs the
`cuda` extra plus `chatterbox-tts`).

## Benchmarking latency (M2)

`scripts/bench_latency.py` runs the turn-based pipeline N times and reports per-stage
min/median/mean/max — use it to compare the Mac and the 4080 and to catch regressions:

```bash
python scripts/bench_latency.py --backend mac  --wav question.wav --runs 5
python scripts/bench_latency.py --backend cuda --wav question.wav --runs 5 --json cuda.json
```

These are full-stage, turn-based wall times — not the streaming "time to first audio" budget
(that arrives with the streaming pipeline in M3).

## Layout

```
config/
  backends/{mac,cuda}.yaml   # which adapter + model per stage; BACKEND switch
  personas/*.yaml            # the four personas (prompt + voice + behavior)
src/personavoice/
  models.py                  # Persona, VoiceRef, Transcript, Msg, configs
  adapters/                  # stt/ llm/ tts/ — base classes + per-backend impls + factory
  persona/                   # loader, prompt builder, registry
  server/                    # settings, config loading, `--check` entrypoint
  orchestrator/ memory/ voice/   # filled in M1/M3, M8, M5/M9
scripts/
  download_models.py         # pinned model manifest + downloader
  bench_latency.py           # per-stage latency benchmark (mac/cuda)
docker-compose.yml           # 4080 stack: vLLM + persona-voice server
Dockerfile                   # CUDA server image (STT + TTS)
tests/
```

## Backends

| Stage | Mac (dev) | CUDA (prod) |
|-------|-----------|-------------|
| STT   | `whisper_mlx` | `faster_whisper` / `parakeet` |
| LLM   | `lmstudio` / `ollama` / `mlx_lm` | `vllm` |
| TTS   | `kokoro` (fast, no clone) / `f5_mlx` (clone) | `orpheus` / `chatterbox` |

Select with `BACKEND=mac|cuda`. Each backend's `config/backends/<backend>.yaml` names the
adapter, model, and per-adapter options (env interpolation via `${VAR:-default}` is
supported).

## Models & licenses

`python scripts/download_models.py --backend <mac|cuda> --list` prints the pinned manifest.

| Stage | Mac | CUDA | License |
|-------|-----|------|---------|
| STT   | `mlx-community/whisper-large-v3-turbo` | `Systran/faster-whisper-large-v3` | MIT |
| LLM   | LM Studio / Ollama (any loaded model) | `Qwen/Qwen2.5-7B-Instruct` (vLLM) | model-dependent |
| TTS   | `hexgrad/Kokoro-82M` | `canopylabs/orpheus-3b-0.1-ft` | Apache-2.0 (see caveats) |

Licenses verified against each model card (2026-06). Two caveats that affect
redistribution / commercial use:

- **Orpheus-3b-0.1-ft** is tagged Apache-2.0 but its weights are fine-tuned from
  **Llama-3.2-3B**, so Meta's Llama 3.2 Community License also applies. **Chatterbox**
  (`ResembleAI/chatterbox`, MIT) is the clean-license CUDA alternative.
- **F5-TTS** weights (the M5 Mac cloning option) are **CC-BY-NC** (non-commercial) due to
  the Emilia training set, even though the F5 *code* is MIT. For commercial cloning use
  Chatterbox (MIT) on CUDA, or an Apache-licensed OpenF5 checkpoint.

> Licenses drift — **re-verify before any redistribution**.

## Development

```bash
pip install -e '.[dev]'
pre-commit install      # ruff lint+format on commit
ruff check . && ruff format --check .
pytest
```

# Persona-Voice

A real-time, open-source **speech-to-speech persona system** built as a modular cascade
(**STT → LLM brain → TTS**). It emulates people for practical use: a PM/HR interviewer for
practice, an English conversation teacher, or a casual companion.

It runs as a server on an **RTX 4080 (16 GB)** in production and is fully developable on a
**Mac M4 Max** via swappable Mac-native backends. The iPhone client is thin — it only
captures and plays audio.

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full design and milestones.

## Status

Current milestone: **M0 — Foundations** (repo skeleton, config, adapter interfaces,
backend factory, model manifest). Adapters are stubs until M1/M2 implement real inference.

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
pip install -e '.[mac]'     # M4 Max: mlx-whisper, mlx-lm, kokoro, ...
pip install -e '.[cuda]'    # RTX 4080: faster-whisper, ...
```

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
scripts/download_models.py   # pinned model manifest + downloader
tests/
```

## Backends

| Stage | Mac (dev) | CUDA (prod) |
|-------|-----------|-------------|
| STT   | `whisper_mlx` | `faster_whisper` / `parakeet` |
| LLM   | `ollama` / `mlx_lm` | `vllm` |
| TTS   | `kokoro` (fast, no clone) / `f5_mlx` (clone) | `orpheus` / `chatterbox` |

Select with `BACKEND=mac|cuda`. Each backend's `config/backends/<backend>.yaml` names the
adapter, model, and per-adapter options (env interpolation via `${VAR:-default}` is
supported).

## Models & licenses

`python scripts/download_models.py --backend <mac|cuda> --list` prints the pinned manifest.

| Stage | Mac | CUDA | License |
|-------|-----|------|---------|
| STT   | `mlx-community/whisper-large-v3-turbo` | `Systran/faster-whisper-large-v3` | MIT |
| LLM   | `qwen2.5:7b-instruct` (Ollama) | `Qwen/Qwen2.5-7B-Instruct` (vLLM) | Apache-2.0 |
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

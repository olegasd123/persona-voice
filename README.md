# Persona-Voice

A real-time, open-source **speech-to-speech persona system** built as a modular cascade
(**STT → LLM brain → TTS**). It emulates people for practical use: a PM/HR interviewer for
practice, an English conversation teacher, or a casual companion.

It runs as a server on an **RTX 4080 (16 GB)** in production and is fully developable on a
**Mac M4 Max** via swappable Mac-native backends. The iPhone client is thin — it only
captures and plays audio.

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full design and milestones.

## Status

Current milestone: **M4 — Persona system (done)**. The four personas (PM/HR interviewer,
language teacher, companion) are selectable at runtime and each behaves *and sounds*
distinct. A **voice registry** (`config/voices.yaml`, `voice/registry.py`) maps each
persona's logical voice to a backend-native preset, so on the Mac they speak as
`af_heart` / `af_sarah` / `af_nicole` / `am_michael` (verified: four distinct Kokoro voices),
and on the 4080 as Orpheus presets. The LiveKit agent **selects the persona** from job/room
metadata (`{"persona": "hr_interviewer"}`) or `PERSONAVOICE_PERSONA`, and a client can
**hot-swap** the persona mid-call via a data message — the registry hot-reloads so edited
persona files take effect without a restart. `temperature` / `turn_style` are wired through
to the LLM and prompt.

**M3 (partial):** the reply **streams** — LLM tokens are chunked into sentences
(`orchestrator/chunker.py`) and spoken as they're generated, so the persona starts talking
before the full reply exists. A `TurnController` (`orchestrator/turn.py`) makes **barge-in**
one cancellable task, and the **LiveKit agent** (`orchestrator/agent.py`, lazy-imported
behind the `livekit` extra) wires WebRTC transport + Silero VAD endpointing + barge-in.
`personavoice-stream-demo` runs the streaming loop on the Mac without LiveKit.

> Streaming is verified e2e on the M4 Max — a spoken question streamed back as 11 sentence
> wavs, warm **first_token 0.76 s · first_audio 5.44 s · total 8.95 s** (speech starts at
> 5.44 s while the rest generates). The live **browser/LiveKit** back-and-forth + barge-in
> need a running LiveKit server (and the 4080 for the ≤900 ms first-audio budget) — that's
> the open M3 step, analogous to M2's on-4080 run.

**M2 (done):** the CUDA cascade mirrors the Mac one — `faster_whisper` / `parakeet` (STT),
`vllm` (LLM, OpenAI-compatible, sharing the streaming path with `lmstudio`), `orpheus` /
`chatterbox` (TTS), with `docker-compose.yml` (vLLM + server) and `scripts/bench_latency.py`.
Code-complete, lazy-imported, unit-tested on the Mac; the on-4080 run is pending the GPU box.

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
reply to `reply.wav` (override with `--out`). For the streaming version (and barge-in / the
live LiveKit server), see **Streaming voice loop (M3)** below.

> **Reasoning models:** local models like Qwen3 / gpt-oss emit a hidden thinking trace
> (`reasoning_content`) that the adapter never speaks. Keep it short with
> `extra_body: {reasoning_effort: low}` (already set in `mac.yaml`) so replies start fast and
> don't exhaust `max_tokens` before producing any spoken content.

## Streaming voice loop (M3)

Same cascade, but the reply is **streamed sentence-by-sentence** instead of waiting for the
whole thing. Run it on the Mac with no LiveKit server:

```bash
# Speak into a wav; hear each sentence as soon as it's synthesized.
personavoice-stream-demo --wav question.wav --persona companion --play
personavoice-stream-demo --record 5 --persona language_teacher --play
```

It writes one wav per sentence to `reply_stream/` and prints **time-to-first-token** and
**time-to-first-audio** (the perceived latency — when speech starts), which on the M4 Max is
well below the turn-based total because TTS overlaps LLM generation.

### Live LiveKit agent

For a live, barge-in conversation over WebRTC, run the LiveKit Agents worker. It needs the
`livekit` extra and a LiveKit server (cloud or self-hosted):

```bash
pip install -e '.[livekit]'          # livekit-agents + Silero VAD plugin
# Set LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET in .env, then:
personavoice --serve                 # production worker (BACKEND from env)
personavoice-agent dev               # hot-reload dev worker
```

The agent (`orchestrator/agent.py`) uses Silero VAD to endpoint each utterance, transcribes
it, and streams the persona's reply onto the published audio track; when VAD detects the
user starting to speak, `TurnController` cancels the in-flight LLM+TTS and flushes the queue
(**barge-in**). Connect any LiveKit client (the [Agents Playground](https://agents-playground.livekit.io/)
or the iOS app in M6) to converse. This live run is the open M3 acceptance step.

## Personas & voices (M4)

A persona is a YAML file (`config/personas/*.yaml`): a system prompt plus knobs for the LLM
(`temperature`, `max_tokens`), behavior (`turn_style`, `follow_up_probability`), and a
logical `voice.ref`. Four ship today — `pm_interviewer`, `hr_interviewer`,
`language_teacher`, `companion` — and any demo or the live agent takes `--persona <id>`.

Each persona references a **logical voice** (`voices/companion_soft`); the **voice registry**
(`config/voices.yaml`) maps that to a concrete preset per TTS backend, so the personas sound
distinct on whichever backend is active:

```yaml
# config/voices.yaml
companion_soft:
  emotion: warm
  presets: { kokoro: af_heart, orpheus: tara }   # Mac / 4080
pm_calm:
  presets: { kokoro: am_michael, orpheus: leo }
```

A clone-only backend (Chatterbox/F5) has no preset and falls back to its default voice until
M5 wires zero-shot cloning (which will use a `sample:` per voice). `python -m
personavoice.server --check` lists the loaded personas and voices and warns if a persona
won't sound distinct on the active backend.

**Selecting a persona on the live agent.** The LiveKit agent picks the persona from, in
priority order: the dispatch's job metadata, the room metadata
(`{"persona": "hr_interviewer"}`), then the `PERSONAVOICE_PERSONA` env var (else the first
registered persona). A connected client can also **switch persona mid-call** by publishing a
data message — a bare id (`hr_interviewer`) or `{"persona": "hr_interviewer"}`. The swap
keeps the conversation history, and the persona registry reloads from disk first, so editing
a persona file takes effect without restarting the worker.

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

These are full-stage, turn-based wall times. For the streaming "time to first audio", use
`personavoice-stream-demo` (see **Streaming voice loop (M3)**), which reports it directly.

## Layout

```
config/
  backends/{mac,cuda}.yaml   # which adapter + model per stage; BACKEND switch
  personas/*.yaml            # the four personas (prompt + voice + behavior)
  voices.yaml                # voice registry: logical voice -> per-backend preset (M4)
src/personavoice/
  models.py                  # Persona, VoiceRef, VoiceDef, Transcript, Msg, configs
  adapters/                  # stt/ llm/ tts/ — base classes + per-backend impls + factory
  persona/                   # loader, prompt builder, registry
  voice/                     # voice registry (M4); zero-shot cloning in M5
  server/                    # settings, config loading, `--check` / `--serve` entrypoint
  orchestrator/              # pipeline (M1) + chunker/streaming/turn/agent (M3 streaming)
  memory/                    # filled in M8
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

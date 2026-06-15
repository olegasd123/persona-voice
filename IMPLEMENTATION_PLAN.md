# Persona-Voice — Implementation Plan

A real-time, open-source **speech-to-speech persona system** built as a **modular cascade**
(STT → LLM brain → TTS) that can emulate people for practical use cases:

- **PM / HR interviewer** — job-interview practice
- **Language teacher** — conversational practice (English first)
- **Companion (BF/GF)** — casual / romantic conversation

It runs as a **server on an RTX 4080 (16 GB)** with a **thin iPhone client**, and is fully
developable on a **Mac M4 Max** via swappable Mac-native backends.

---

## 1. Design principles

1. **Cascade, not end-to-end.** Three swappable parts give us voice cloning, per-persona
   fine-tuning, and a clear training path — none of which open end-to-end S2S models do well.
2. **Backend abstraction first.** Every stage (STT / LLM / TTS) sits behind an interface with
   a `BACKEND=mac|cuda` switch. Same repo runs on the M4 Max (MLX / llama.cpp / whisper.cpp)
   and the 4080 (vLLM / CUDA). Persona logic, orchestration, and the iPhone client never change.
3. **Stream everything.** Stream STT while the user talks; pipe LLM tokens into TTS
   sentence-by-sentence. This is what keeps perceived latency low.
4. **Thin client, fat server.** iPhone only captures/plays audio over WebRTC. All models run
   server-side (4080 in prod, Mac in dev).
5. **Personas are config + adapters.** A persona = system prompt + (optional) LoRA adapter +
   voice + behavior knobs, defined in YAML and hot-swappable.

---

## 2. Target architecture

```
┌─────────────┐   WebRTC (Opus)   ┌──────────────────────────────────────────────┐
│  iPhone app │  ───────────────▶ │            Server (4080 / M4 Max)            │
│  mic + spkr │  ◀─────────────── │                                              │
└─────────────┘   audio stream    │  LiveKit Agent (transport, VAD, turn-taking) │
                                   │      │                                       │
                                   │      ▼                                       │
                                   │  ┌───────┐   ┌────────────┐   ┌──────────┐   │
                                   │  │  STT  │──▶│ LLM (brain)│──▶│   TTS    │   │
                                   │  │adapter│   │  + persona │   │ adapter  │   │
                                   │  └───────┘   │  LoRA      │   │ + voice  │   │
                                   │              └─────┬──────┘   └──────────┘   │
                                   │                    │                          │
                                   │              ┌─────▼──────┐                   │
                                   │              │   Memory   │ (RAG + profile)   │
                                   │              └────────────┘                   │
                                   └──────────────────────────────────────────────┘
```

### Component matrix (CUDA prod ↔ Mac dev)

| Stage      | RTX 4080 (prod)                    | M4 Max (dev)                          | VRAM/RAM (4080) |
|------------|------------------------------------|---------------------------------------|-----------------|
| **STT**    | `faster-whisper` / NVIDIA Parakeet | `mlx-whisper` / `whisper.cpp`         | ~2 GB           |
| **LLM**    | Qwen2.5-7B / Llama-3.1-8B via vLLM | **LM Studio** / Ollama / `mlx-lm`     | ~5–6 GB         |
| **TTS**    | Orpheus / Chatterbox               | `f5-tts-mlx` / Chatterbox-MPS / Kokoro| ~3–4 GB         |
| **Glue**   | LiveKit Agents                     | LiveKit Agents (identical)            | —               |
| **Train**  | Unsloth / LLaMA-Factory (QLoRA)    | `mlx-lm` LoRA (light)                 | —               |

### Latency budget (turn-based, 4080)

| Step                         | Target  |
|------------------------------|---------|
| Endpointing (VAD silence)    | ~200 ms |
| STT finalize                 | ~150 ms |
| LLM time-to-first-token      | ~200 ms |
| TTS first audio chunk        | ~200 ms |
| WebRTC round-trip (local)    | ~50 ms  |
| **Total to first audio**     | **~600–900 ms** |

Mac dev expectation: **~1.5–2.5 s** (TTS is the bottleneck; use Kokoro for fast dev iterations).

---

## 3. Proposed repository layout

```
persona-voice/
├── IMPLEMENTATION_PLAN.md
├── README.md
├── pyproject.toml
├── docker-compose.yml              # 4080 server stack
├── .env.example                    # BACKEND=mac|cuda, model paths, keys
├── config/
│   ├── backends/{mac.yaml,cuda.yaml}
│   └── personas/
│       ├── pm_interviewer.yaml
│       ├── hr_interviewer.yaml
│       ├── language_teacher.yaml
│       └── companion.yaml
├── src/personavoice/
│   ├── adapters/
│   │   ├── stt/{base.py,whisper_mlx.py,faster_whisper.py,parakeet.py}
│   │   ├── llm/{base.py,ollama.py,vllm.py,mlx_lm.py}
│   │   └── tts/{base.py,f5_mlx.py,orpheus.py,chatterbox.py,kokoro.py}
│   ├── orchestrator/{agent.py,pipeline.py,turn.py}
│   ├── persona/{loader.py,prompt.py,registry.py}
│   ├── memory/{rag.py,profile.py,store.py}
│   ├── voice/{clone.py,registry.py}
│   └── server/{app.py,ws.py,config.py}
├── training/
│   ├── persona_lora/{datasets/,configs/,train.py,merge.py}
│   └── voice/{clone_zeroshot.py,finetune.py}
├── ios/PersonaVoice/              # SwiftUI + LiveKit iOS SDK
├── scripts/{dev_server.sh,bench_latency.py,download_models.py}
└── tests/
```

### Adapter interfaces (the contract every backend implements)

```python
class STTAdapter(Protocol):
    async def stream(self, audio: AsyncIterator[bytes]) -> AsyncIterator[Transcript]: ...

class LLMAdapter(Protocol):
    async def stream_chat(self, messages: list[Msg], persona: Persona) -> AsyncIterator[str]: ...

class TTSAdapter(Protocol):
    async def stream_tts(self, text: AsyncIterator[str], voice: VoiceRef) -> AsyncIterator[bytes]: ...
    async def clone_voice(self, sample_wav: bytes, name: str) -> VoiceRef: ...
```

### Persona config (example)

```yaml
# config/personas/hr_interviewer.yaml
id: hr_interviewer
name: "HR Interviewer"
llm:
  base_model: qwen2.5-7b-instruct
  lora: adapters/hr_interviewer        # optional; null until trained
  temperature: 0.7
voice:
  ref: voices/hr_warm                  # zero-shot clone or fine-tuned
  emotion: neutral-warm
system_prompt: |
  You are a senior HR interviewer at a mid-size tech company...
behavior:
  turn_style: concise
  follow_up_probability: 0.6
memory:
  enabled: true
  scope: per_user
```

---

## 4. Milestones

Effort estimates assume **one developer**, part-time. Each milestone is independently
demoable. Treat M0–M5 as the MVP path; M6+ are productionization and the training stack.
Mark a milestone as `[Done]` when it's completed.

### M0 — Foundations *(≈ 3–5 days)* `[Done]`
**Goal:** repo skeleton, config, adapter interfaces, model download, both environments boot.
- [x] Init repo, `pyproject.toml`, linting (ruff), pre-commit, `.env.example`.
- [x] Define `STT/LLM/TTS` adapter base classes + `Persona`/`VoiceRef` data models.
- [x] `config/backends/{mac,cuda}.yaml` and a backend factory keyed on `BACKEND`.
- [x] `scripts/download_models.py` (Qwen2.5-7B, Whisper, one TTS) for both backends.
- [x] **Verify latest model versions + licenses** and pin them. Findings: Qwen2.5-7B, Kokoro = Apache-2.0; Whisper = MIT; Chatterbox = MIT. Caveats: Orpheus-3b weights derive from Llama-3.2 (Llama 3.2 license also applies); F5-TTS weights are CC-BY-NC (non-commercial) — code is MIT. See README "Models & licenses".
- **Acceptance:** `python -m personavoice.server --check` validates config and loads stub adapters on both Mac and 4080. ✅ Passes on Mac (cuda config validated; run on the 4080 to confirm there). 17 tests green, ruff + mypy clean.

### M1 — Walking skeleton (offline voice loop) *(≈ 1 week)* `[Done]`
**Goal:** prove the cascade end-to-end, file-based, turn-based, on the Mac.
- [x] Implement Mac adapters: `mlx-whisper` (STT), `LM Studio` (LLM, OpenAI-compatible; `Ollama`
      also available), `Kokoro` (TTS, no clone yet). Heavy ML libs are lazy-imported so
      `--check`/tests still run without the `mac` extra.
- [x] Shared `audio.py` (wav decode/encode/resample) — audio flows between stages as wav bytes.
- [x] `pipeline.py`: wav in → transcript → LLM reply → wav out (turn-based, with per-stage timings).
- [x] CLI demo (`personavoice-demo`): speak into a wav (or `--record` from the mic), hear a reply.
- **Acceptance:** ✅ a recorded question returns coherent spoken audio on the M4 Max. Verified the
  full STT→LLM→TTS loop: spoken question → Whisper transcript → in-character persona reply (companion
  & language_teacher) → Kokoro audio played back. Warm steady-state timings: **STT 1.2 s · LLM 0.8 s
  · TTS 3.8 s · total ~5.8 s** (turn-based; M3 streaming will cut perceived latency). 36 unit tests
  green; ruff + mypy clean.
- **Setup notes (M4 Max):**
  - The `mac` extra (kokoro → spaCy → blis) has **no wheels for Python 3.13/3.14**; use **Python 3.12**.
    Provisioned via `uv venv --python 3.12 .venv312 && uv pip install -e '.[mac]'`.
  - LLM: load a model in **LM Studio** and start its server (default config: `openai/gpt-oss-20b`).
  - First run downloads Whisper + Kokoro + the spaCy English model (one-time; cold start is slow).
  - *LLM note:* local reasoning models (Qwen3 / gpt-oss) stream a hidden `reasoning_content` trace the
    adapter never speaks; keep it short via `extra_body: {reasoning_effort: low}` (set in `mac.yaml`)
    so spoken content starts fast and `max_tokens` isn't spent on thinking.

### M2 — Backend parity (Mac ↔ CUDA) *(≈ 4–6 days)* `[Done]`
**Goal:** same pipeline runs on the 4080.
- [x] CUDA adapters: `faster_whisper` + `parakeet` (STT), `vllm` (LLM), `orpheus` + `chatterbox`
      (TTS). The OpenAI-compatible streaming/reasoning logic is shared in
      `adapters/llm/_openai_compat.py`, so `vllm` and `lmstudio` are thin subclasses. All heavy
      libs are lazy-imported so `--check`/tests still run without the `cuda` extra.
- [x] `docker-compose.yml` (vLLM server + persona-voice server) and a CUDA `Dockerfile`. VRAM
      budget documented: vLLM 4-bit ~6–7 GB + STT ~2 GB + Chatterbox ~2–3 GB ≈ 10–12 GB, within
      16 GB; vLLM's `--gpu-memory-utilization` is capped so STT/TTS fit. Orpheus runs its own
      in-process vLLM (tight on one card) → Chatterbox is the single-GPU default.
- [x] `scripts/bench_latency.py` — runs the turn-based pipeline N times and reports per-stage
      min/median/mean/max for either backend (`--json` to persist).
- **Acceptance:** ⚠️ *partial on the Mac.* The CUDA cascade is code-complete, lazy-imported, and
  unit-tested at the logic level (52 tests green; ruff + mypy clean), and `BACKEND=cuda
  server --check` validates the stack on the Mac. The **on-4080 end-to-end demo + real latency
  report are pending access to the GPU box** — run `docker compose up` and `bench_latency.py
  --backend cuda` there to close this out.
- **Notes:**
  - vLLM/Orpheus/Chatterbox are installed in the server image (CUDA toolchain), not in the
    `cuda` wheel extra, which stays light (`faster-whisper`, `soundfile`, `numpy`, `httpx`).
  - `cuda.yaml`'s LLM `model` must match the id vLLM serves (incl. the `-AWQ` suffix);
    `quantization`/`max-model-len` are vLLM *server* flags, set in `docker-compose.yml`.

### M3 — Real-time orchestration & streaming *(≈ 1.5 weeks)*
**Goal:** live, low-latency, turn-based conversation with streaming + barge-in.
- [ ] Integrate **LiveKit Agents**: WebRTC transport, Silero VAD, turn detection.
- [ ] Streaming STT (partials), token→sentence chunker feeding streaming TTS.
- [ ] Barge-in: cancel in-flight TTS/LLM when user starts speaking.
- [ ] Hit latency budget (≤ ~900 ms to first audio on 4080).
- **Acceptance:** a browser/LiveKit test client holds a natural spoken back-and-forth; barge-in works.

### M4 — Persona system *(≈ 1 week)*
**Goal:** the four personas, selectable at runtime.
- [ ] Persona loader + prompt builder + registry; hot-swap without restart.
- [ ] Author system prompts + behavior knobs for PM, HR, Teacher, Companion.
- [ ] Per-persona voice + temperature/turn-style wiring.
- **Acceptance:** switch persona via config/API; each behaves and sounds distinctly in live conversation.

### M5 — Voice cloning (zero-shot) *(≈ 4–6 days)*
**Goal:** clone a voice from a short sample and use it per persona.
- [ ] `voice/clone.py` zero-shot from ~10 s sample (Orpheus/Chatterbox on CUDA, F5-mlx on Mac).
- [ ] Voice registry + assign clones to personas.
- **Acceptance:** record a sample → persona speaks in that voice in live conversation.
- **→ MVP complete: a working, persona-driven, voice-cloning S2S server.**

### M6 — iPhone client *(≈ 1.5–2 weeks)*
**Goal:** native app that talks to the server.
- [ ] SwiftUI app + **LiveKit iOS SDK** (mic capture, audio playback, push-to-talk + VAD modes).
- [ ] Persona picker, connection settings (server URL/token), basic transcript view.
- [ ] Background-audio/interruption handling, reconnection.
- **Acceptance:** full spoken conversation from the iPhone to the 4080 server over the network.

### M7 — Persona fine-tuning (LoRA) *(≈ 1.5 weeks)*
**Goal:** train per-persona brains beyond prompting.
- [ ] `training/persona_lora/`: dataset format (in-character dialogues), curation scripts.
- [ ] QLoRA training on 4080 (Unsloth/LLaMA-Factory); light LoRA on Mac via `mlx-lm`.
- [ ] Adapter hot-swap at inference; eval harness comparing prompt-only vs LoRA.
- **Acceptance:** an HR/Teacher LoRA measurably improves in-character behavior; adapters load at runtime.

### M8 — Memory / learn-from-conversations *(≈ 1.5 weeks)*
**Goal:** continuity across sessions; foundation for long-term learning.
- [ ] Per-user transcript store; rolling profile/summary; RAG retrieval into context.
- [ ] Privacy controls: consent, local encryption, per-user delete/export.
- [ ] Pipeline to distill logs into periodic SFT/DPO LoRA refreshes (manual trigger first).
- **Acceptance:** the assistant recalls prior-session facts; user can wipe their data.

### M9 — Voice fine-tuning (high fidelity) *(≈ 1 week)*
**Goal:** custom voices beyond zero-shot for key personas.
- [ ] `training/voice/finetune.py` on a target speaker dataset (CUDA).
- [ ] Quality A/B vs zero-shot; fold winners into the voice registry.
- **Acceptance:** a fine-tuned voice is clearly higher fidelity than its zero-shot clone.

### M10 — Hardening, eval & latency optimization *(≈ 1–1.5 weeks)*
**Goal:** make it robust and fast enough to use daily.
- [ ] Latency tuning: quantization, KV-cache, TTS chunk sizing, speculative endpointing.
- [ ] Automated eval: latency, WER (STT), persona adherence, voice MOS spot-checks.
- [ ] Observability (per-stage timings, logs), graceful error/reconnect, load test.
- [ ] Security pass: auth on the server API, TLS, rate limiting, secrets handling.
- **Acceptance:** sustained multi-turn sessions within latency budget; eval dashboard green.

---

## 5. Sequencing & critical path

```
M0 ─▶ M1 ─▶ M2 ─▶ M3 ─▶ M4 ─▶ M5  ══▶  MVP
                              └─▶ M6 (iPhone, can start after M3)
M5 ─▶ M7 ─▶ M8 ─▶ M9 ─▶ M10  (training + production track)
```

- **M6 (iPhone)** can begin in parallel once **M3** exposes a stable LiveKit endpoint.
- **M7–M9 (training)** require the 4080 and only need the MVP (through M5) as a baseline.
- Recommended first deliverable to *use*: **M0→M5 + M6** = a usable companion/interview app.

---

## 6. Risks & mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Mac TTS too slow for live dev | Slow iteration | Use Kokoro (fast, no clone) for dev; switch to cloning on the 4080. |
| CUDA-only deps break on Mac | Lost dev time | Adapter abstraction + pinned Mac forks; CI smoke-test both backends. |
| 16 GB VRAM pressure (STT+LLM+TTS together) | OOM on 4080 | 4-bit LLM, quantized STT, smaller TTS; measure in M2; offload if needed. |
| Latency budget missed | Unnatural feel | Stream everything; tune chunking/endpointing in M3 & M10. |
| Voice-clone misuse / consent | Legal/ethical | Consent gating, watermark option, restrict cloning to user-provided samples. |
| Conversation data privacy | Trust/legal | Encryption at rest, per-user delete/export, explicit opt-in for training (M8). |
| Model license drift | Compliance | License audit in M0; re-check before any redistribution. |

---

## 7. Tech stack summary

- **Orchestration:** LiveKit Agents (WebRTC, VAD, turn detection, iOS SDK). Pipecat as fallback.
- **STT:** faster-whisper / Parakeet (CUDA) · mlx-whisper / whisper.cpp (Mac).
- **LLM:** Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct · vLLM (CUDA) · LM Studio / Ollama / mlx-lm (Mac).
- **TTS:** Orpheus / Chatterbox (CUDA) · f5-tts-mlx / Chatterbox-MPS / Kokoro (Mac).
- **Training:** Unsloth / LLaMA-Factory (QLoRA, CUDA) · mlx-lm LoRA (Mac).
- **Client:** SwiftUI + LiveKit iOS SDK.
- **Infra:** Docker Compose (server), Python 3.11+, uv/poetry.

**Language decision:** Python throughout for server + ML + training (single language, all reference
models work out of the box, fine-tuning is native, LiveKit Agents Python SDK ties it together).
The iOS client is Swift/SwiftUI. If a polyglot orchestrator is ever wanted, the adapter layer
already isolates models so they can be exposed as HTTP/gRPC workers — but that's not the plan.

---

## 8. Open decisions (revisit as we build)

- Final TTS pick for prod: **Orpheus** (expressive, emotion tags; but weights inherit the Llama-3.2 license) vs **Chatterbox** (emotion exaggeration control; clean MIT) — decide in M2/M5 after a quality+latency bake-off. License leans Chatterbox if redistribution/commercial matters. *M2 update:* both adapters are implemented; **Chatterbox is the default for the single-4080 dockerized stack** (no second in-process vLLM, fits VRAM, MIT). Orpheus stays available as the expressive option; final quality A/B in M5. Mac cloning via F5-TTS is CC-BY-NC (non-commercial) — fine for dev, not for shipping.
- LLM base: **Qwen2.5-7B** vs **Llama-3.1-8B** — decide in M4 on persona quality.
- Memory store: lightweight (SQLite + FAISS) vs managed vector DB — decide in M8.
- Endpointing strategy: pure VAD vs semantic turn detection — tune in M3/M10.
```

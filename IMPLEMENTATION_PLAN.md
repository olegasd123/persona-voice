# Persona-Voice — Implementation Plan

A real-time, open-source **speech-to-speech persona system** built as a **modular cascade**
(STT → LLM brain → TTS) that can emulate people for practical use cases:

- **PM / HR interviewer** — job-interview practice
- **Language teacher** — conversational practice (English first)
- **Companion (BF/GF)** — casual / romantic conversation

It runs as a **server on an RTX 4080 (16 GB)** with a **thin cross-platform client (iOS + Android)**, and is fully
developable on a **Mac M4 Max** via swappable Mac-native backends.

---

## 1. Design principles

1. **Cascade, not end-to-end.** Three swappable parts give us voice cloning, per-persona
   fine-tuning, and a clear training path — none of which open end-to-end S2S models do well.
2. **Backend abstraction first.** Every stage (STT / LLM / TTS) sits behind an interface with
   a `BACKEND=mac|cuda` switch. Same repo runs on the M4 Max (MLX / llama.cpp / whisper.cpp)
   and the 4080 (vLLM / CUDA). Persona logic, orchestration, and the client never change.
3. **Stream everything.** Stream STT while the user talks; pipe LLM tokens into TTS
   sentence-by-sentence. This is what keeps perceived latency low.
4. **Thin client, fat server.** The client only captures/plays audio over WebRTC. All models run
   server-side (4080 in prod, Mac in dev).
5. **Personas are config + adapters.** A persona = system prompt + (optional) LoRA adapter +
   voice + behavior knobs, defined in YAML and hot-swappable.

---

## 2. Target architecture

```
┌─────────────┐   WebRTC (Opus)   ┌──────────────────────────────────────────────┐
│ Flutter app │  ───────────────▶ │            Server (4080 / M4 Max)            │
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
├── client/                         # Flutter app (iOS + Android) — LiveKit Flutter SDK
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
- [x] `scripts/bench_latency.py` — turn-based per-stage min/median/mean/max for either backend
      (`--json` to persist), plus `--stream` for the M3 time-to-first-audio landmarks
      (STT finalize → LLM TTFT → first TTS chunk) measured through the streaming pipeline.
- **Acceptance:** ✅ *closed on real CUDA hardware — RTX 5090 (32 GB, Blackwell sm_120).* The
  CUDA cascade is code-complete, lazy-imported, and unit-tested at the logic level (ruff + mypy
  clean), and `BACKEND=cuda server --check` PASSes. **Verified end-to-end on the GPU** via the
  Windows-native dev path: faster-whisper STT + LM Studio brain (`qwen2.5-vl-7b`) + Chatterbox
  TTS (vLLM/Orpheus stay the dockerized prod default — see the Windows setup note). Warm
  turn-based `bench_latency --backend cuda`: **STT 0.105 s · LLM 0.62 s · TTS (whole reply)
  5.1 s · total ~5.8 s** — STT is ~11× faster than the Mac's 1.2 s; STT+LLM are well under
  budget and full-reply TTS is the bottleneck (the streaming first-audio number that actually
  matters is in M3). *Note:* the box turned out to be a 5090/32 GB, not the planned 4080/16 GB,
  so the VRAM-pressure risk is moot — there's ample headroom for an unquantized 7B + larger TTS.
- **Notes:**
  - vLLM/Orpheus/Chatterbox are installed in the server image (CUDA toolchain), not in the
    `cuda` wheel extra, which stays light (`faster-whisper`, `soundfile`, `numpy`, `httpx`).
  - `cuda.yaml`'s LLM `model` must match the id vLLM serves (incl. the `-AWQ` suffix);
    `quantization`/`max-model-len` are vLLM *server* flags, set in `docker-compose.yml`.
  - **Setup notes (Windows / RTX 5090):** the system Python (3.14) is too new for the ML
    wheels — provision **Python 3.12** (`uv venv --python 3.12 .venv312`). Blackwell (sm_120)
    needs the **cu128** PyTorch wheels (`uv pip install torch torchaudio --index-url
    https://download.pytorch.org/whl/cu128` → torch 2.11+cu128); ctranslate2 4.8 already runs
    faster-whisper on sm_120 (float16). `chatterbox-tts` pins `torch==2.6.0` (no sm_120) —
    install it with `uv pip install chatterbox-tts --override constraints-cuda.txt` to keep the
    cu128 torch (it runs fine against torch 2.11 / transformers 5.2). `cuda.yaml`'s
    adapter/model/base_url are now env-overridable (`PERSONAVOICE_LLM_ADAPTER`,
    `PERSONAVOICE_TTS_ADAPTER`, `PERSONAVOICE_LLM_BASE_URL`, …), so the **same file** drives both
    the vLLM/Orpheus Docker prod stack (defaults) and this LM-Studio+Chatterbox dev box (set the
    vars in `.env`). LM Studio: `lms load qwen/qwen2.5-vl-7b` + start the server; swap to a text
    `Qwen2.5-7B-Instruct` later via one env var for a cleaner (vision-free) brain.
    *(M7 update: this dev box now serves the brain from **vLLM** like prod — `docker compose up
    -d vllm` serving the unquantized `Qwen/Qwen2.5-7B-Instruct` on the 5090's 32 GB — so STT/LLM/
    TTS all match the prod cascade and the M7 served-LoRA A/B works. LM Studio stays a documented
    fallback in `.env`/`.env.example`.)*
  - **Low-latency TTS (Kokoro) on CUDA:** to hit the sub-900 ms first-audio budget, also
    `uv pip install kokoro --override constraints-cuda.txt`, then `uv pip install pip` +
    `python -m spacy download en_core_web_sm` (the uv venv ships no pip, and Kokoro's misaki G2P
    fetches a spaCy model at runtime). Select it with `PERSONAVOICE_TTS_ADAPTER=kokoro` +
    `PERSONAVOICE_TTS_MODEL=hexgrad/Kokoro-82M`. Kokoro auto-uses CUDA, voices all four personas
    distinctly, but can't clone — Chatterbox stays the cloning default.

### M3 — Real-time orchestration & streaming *(≈ 1.5 weeks)* `[Partial]`
**Goal:** live, low-latency, turn-based conversation with streaming + barge-in.
- [x] Integrate **LiveKit Agents**: WebRTC transport, Silero VAD endpointing, barge-in
      (`orchestrator/agent.py`). LiveKit + plugins are lazy-imported (the `livekit` extra),
      so `--check`/tests/demos never need them. Launch with `personavoice --serve` (or
      `personavoice-agent dev`).
- [x] Token→sentence chunker (`orchestrator/chunker.py`) feeding streaming TTS: the base
      `stream_tts` now speaks each sentence as it's generated, so every backend streams for
      free on top of its one-shot `synthesize`. STT-in is VAD-segmented per utterance
      (Whisper isn't a partial-decoding model), then transcribed — true partials deferred.
- [x] Barge-in: `orchestrator/turn.py` `TurnController` drives the response as one
      cancellable task; VAD speech-start cancels in-flight LLM+TTS and flushes the output
      queue. Cancellation semantics are unit-tested with fakes.
- [x] `StreamingPipeline` (`orchestrator/streaming.py`) + `personavoice-stream-demo`: a
      runnable Mac streaming loop (no LiveKit needed) that synthesizes/plays the reply
      sentence-by-sentence and reports time-to-first-token / time-to-first-audio.
- [x] **Latency budget hit on CUDA (RTX 5090)** with `bench_latency --stream`. With **Kokoro**
      (fast, no clone): warm **e2e first-audio ≈ 0.58 s** = STT 0.104 s + LLM TTFT 0.33 s + first
      Kokoro chunk ~0.14 s, whole reply spoken in ~0.96 s — **under the 900 ms target**. With
      **Chatterbox** (the cloning backend): ≈ 1.3 s, its first-sentence synth (~0.8 s) being the
      whole gap. Both TTS are env-selectable (`PERSONAVOICE_TTS_ADAPTER`), so low-latency vs
      cloning is a one-line config switch; Kokoro covers all four persona presets distinctly.
      The *live-agent* number (adds VAD endpointing + WebRTC RTT) still needs a running LiveKit
      SFU.
- **Acceptance:** ⚠️ *partial.* Streaming verified e2e on the M4 Max: a spoken question →
  Whisper transcript → gpt-oss-20b reply **streamed as 11 sentence wavs** with **warm
  first_token 0.76 s · first_audio 5.44 s · total 8.95 s** — the persona starts speaking at
  5.44 s while the rest of the reply is still being generated (a turn-based loop emits no
  audio until the whole reply is generated *and* synthesized). 90 tests green (5 numpy/
  soundfile tests skip in the light dev env); ruff + mypy clean. The **CUDA latency budget is
  now measured and hit** (RTX 5090; see the item above — warm e2e first-audio ≈ 0.58 s with
  Kokoro, ≈ 1.3 s with the cloning Chatterbox). **Still pending:** the live browser/LiveKit
  back-and-forth + barge-in need a running LiveKit server — `personavoice --serve` against a
  LiveKit instance closes this out.

### M4 — Persona system *(≈ 1 week)* `[Done]`
**Goal:** the four personas, selectable at runtime.
- [x] Persona loader + prompt builder + registry; hot-swap without restart. The
      `PersonaRegistry.reload()` re-reads files from disk; the live agent reloads then calls
      `PersonaAgent.set_persona()` on a data message, swapping persona (and voice) mid-call
      while keeping conversation history.
- [x] Author system prompts + behavior knobs for PM, HR, Teacher, Companion (in
      `config/personas/*.yaml`).
- [x] Per-persona voice + temperature/turn-style wiring. `temperature`/`top_p`/`max_tokens`
      flow into the OpenAI-compatible payload; `turn_style` becomes a prompt directive
      (`persona/prompt.py`). A **voice registry** (`config/voices.yaml`, `voice/registry.py`)
      maps each persona's logical `voice.ref` to a backend-native preset, so the four
      personas sound distinct (Kokoro `af_heart`/`af_sarah`/`af_nicole`/`am_michael` on Mac,
      Orpheus presets on CUDA). Clone-only backends fall back to default until M5.
- [x] Runtime selection via API: the LiveKit agent picks the persona from job/room metadata
      (`{"persona": "..."}`) or `PERSONAVOICE_PERSONA` (pure, unit-tested resolver); mid-call
      switch via a data message.
- **Acceptance:** ✅ switch persona via config (`--persona`) / API (agent metadata + data
      message); each behaves and **sounds distinctly** — verified on the M4 Max: the four
      personas synthesize as four distinct Kokoro voices (4/4 unique audio, incl. a male
      voice for PM). 112 unit tests green (5 numpy/soundfile tests skip in the light env);
      ruff + mypy clean. The live LiveKit metadata-select / mid-call-swap path rides on the
      same open M3 step (needs a running LiveKit server).

### M5 — Voice cloning (zero-shot) *(≈ 4–6 days)* `[Partial]`
**Goal:** clone a voice from a short sample and use it per persona.
- [x] `voice/clone.py` zero-shot from a ~10 s sample. Cloning here is **reference
      conditioning**: the cloning TTS speaks *in the voice of* a stored reference WAV passed at
      generation time. `Chatterbox` (CUDA, MIT) and `F5-mlx` (Mac) honor `voice.sample_path`
      (F5 also takes `ref_text`, auto-filled by the cascade's STT); the base
      `TTSAdapter.clone_voice` persists the sample → `VoiceRef`. **Orpheus is preset-only**
      (`orpheus-speech`'s engine takes no reference sample), so cloning on CUDA is Chatterbox —
      `supports_cloning=False` on Orpheus. All model libs are lazy-imported (`clone`/`clone-mac`
      extras) so `--check`/tests run without them.
- [x] Voice registry + assign clones to personas. `ClonesStore` (`<clones_dir>/clones.json`)
      persists clones + **per-persona assignments**; `VoiceRegistry.resolve_for_persona` prefers
      an assigned clone on a cloning backend, else the static preset. Non-destructive (never
      rewrites persona/voices YAML). `personavoice-clone` CLI records/loads a sample, clones,
      assigns, and (`--say --play`) speaks a line in the new voice. `server --check` lists
      clones + assignments and suppresses the distinctness warning for an assigned clone.
- **Acceptance:** ⚠️ *partial (audible clone now verified on the M4 Max).* The cloning system
      is unit-tested end-to-end (sample validation, store persistence/reload, cloner + STT
      ref-text, per-persona resolution, pipeline speaking the assigned clone, CLI, `--check`):
      **147 tests green** (0 skip in `.venv312`); ruff + mypy clean; both `BACKEND=mac|cuda
      server --check` PASS. **Verified audibly with F5 on the M4 Max:** a ~12 s female sample →
      `personavoice-clone --sample … --assign companion --say "…" --play` synthesized **15.4 s
      of speech in the cloned voice in ~20 s** (the companion persona resolved to the clone via
      the registry). Two real-API fixes landed during this: f5's reference transcript kwarg is
      `ref_audio_text` (not `ref_text`), and f5 requires the reference at **24 kHz** — the
      adapter now resamples any sample on the fly. **Still open:** the *live-conversation* clone
      rides on the same open M3 LiveKit-server step.
- **→ MVP complete: a working, persona-driven, voice-cloning S2S server.**

### M6 — Cross-platform client (iOS + Android) *(≈ 2–3 weeks)* `[Done on iOS; Android device run out of scope]`
**Goal:** one Flutter app, from a single codebase, that talks to the server on both iOS and Android.
**Scope note:** the iOS path is live-verified on a real device (below). A real **Android device run
is out of scope** for this project — no Android hardware on hand — so the Android-native layers stay
written-but-device-unverified; revisit if/when a device is available.
- [x] **Token server** (the missing client prerequisite): `server/token_server.py` mints
      short-lived LiveKit join tokens and serves `/personas`, so a thin client can join a room
      without ever seeing the LiveKit secret ("fat server, thin client"). Tokens are stdlib-only
      HS256 JWTs in LiveKit's documented format (`server/tokens.py`) — no extra needed — and are
      unit-tested end-to-end (mint/verify/expiry/tamper + the HTTP routes over a real socket).
      Run with `personavoice --token-server`. Self-hosted SFU + token server in
      `docker-compose.livekit.yml`.
- [x] Flutter app + **LiveKit Flutter SDK** (`livekit_client`): mic capture, audio playback,
      mute toggle — one shared UI for both platforms (`client/`).
- [x] Persona picker, connection settings (token-server URL / api token / identity, persisted),
      basic transcript view (renders LiveKit `TranscriptionEvent`s). Mid-call persona switch
      rides the M4 data-message path; selection on connect sends `{"persona": <id>}`.
- [x] **Mic modes + assistant transcript + audio routing.** Client: open-mic (server VAD) /
      push-to-talk (hold to talk) toggle, loudspeaker by default
      (`AudioOutputOptions(speakerOn: true)`), and reconnect/resume surfaced in the status line
      (`RoomReconnecting/Resuming/Reconnected` → "Reconnecting…"). Server: the agent now
      **publishes its spoken reply as a live transcript** — `StreamingPipeline.stream_response`
      gained an `on_sentence` tap, and `agent.make_transcript_publisher` pushes one growing
      `rtc.Transcription` segment in step with the audio (best-effort; flips to final on
      completion *and* on barge-in), so the assistant's words now appear in the client view.
- [x] **Audio-session interruption + route/Bluetooth-change plumbing** (the platform-channel
      layer). A native `EventChannel` (`persona_voice/audio_session`) surfaces OS audio events to
      Dart: iOS `AudioSessionMonitor.swift` (`AVAudioSession` interruption + route-change
      notifications, registered in `AppDelegate`) and Android `AudioSessionMonitor.kt`
      (`OnAudioFocusChangeListener` + `AudioDeviceCallback`, registered in `MainActivity`).
      `services/audio_session.dart` decodes them into typed events; `VoiceSession` mutes the mic
      on an incoming-call/Siri interruption and **auto-resumes** a previously-live mic (open-mic)
      when it clears, and tracks the active route for the UI. The handler + parser are pure and
      unit-tested room-less; the native monitors only *observe* (LiveKit/WebRTC still owns the
      session). *iOS native compiles + runs on a real device; Android (no SDK/device) is out of
      scope.* (Already done: mic permissions, iOS background-audio modes, `minSdk 23` / iOS 13,
      speakerphone routing, reconnection state.)
- [x] Native telephony **CallKit (iOS) / ConnectionService (Android)** — presenting *our*
      conversation *as* a system call (lock-screen UI, OS call list). The platform layer is
      written: a `MethodChannel` (`persona_voice/telephony`, app→OS: `startCall`/
      `reportConnected`/`endCall`/`setMuted`) + `EventChannel` (`persona_voice/telephony_events`,
      OS→app: `endCall`/`setMuted`), bridged in `services/telephony.dart` (pure
      `parseCallControlEvent` + UUID call-id + a `SystemCallController` interface). iOS
      `CallKitController.swift` models it as an outgoing `CXStartCallAction` and forwards
      `CXEndCallAction`/`CXSetMutedCallAction` from the `CXProviderDelegate`; Android
      `PersonaConnectionService.kt` is a **self-managed** `ConnectionService` (API 26+; no-op on
      23–25) placed via `TelecomManager.placeCall`, relaying `onDisconnect`/
      `onCallAudioStateChanged` through a bridge singleton. `VoiceSession` starts/ends the call
      across connect/teardown, hangs up on a system end, mirrors a system mute onto the mic, and
      pushes app mutes back (with an echo guard). Distinct from the interruption handling above
      (being interrupted *by* a call). *iOS CallKit native compiles + ran on a real device (basic
      connect/talk/hang-up; deeper flows not yet); Android ConnectionService is out of scope (no
      device).* The Dart layer is analyzed + unit-tested.
- **Acceptance:** ✅ *met on iOS.* Token server verified (live-smoked: `/healthz`, `/personas`,
      `POST /token` mint a valid JWT). Server **188 tests green** (the streaming `on_sentence` tap
      and the transcript-publisher); ruff + mypy clean. Flutter client statically verified —
      `flutter analyze` clean, `flutter test` green (**37 tests**: token client + models, the pure
      `sessionStatusLabel`, mic-mode/PTT transitions on a room-less `VoiceSession`, the audio-event
      + call-control parsers, the interruption/route handler, and the system-call behavior with a
      fake controller). **Live on-device run verified (2026-06-16) on a real iPhone 17 Pro / iOS
      26.5:** a full spoken conversation (connect → talk to the Companion persona → streamed reply →
      hang up) over the LAN against the self-hosted LiveKit SFU + token server + `personavoice
      --serve` agent (BACKEND=mac). Getting there required three iOS fixes: the `permission_handler`
      **Podfile macro** (`GCC_PREPROCESSOR_DEFINITIONS << 'PERMISSION_MICROPHONE=1'`, else the mic
      request returns `denied` with no prompt — the actual blocker), Info.plist `NSAllowsLocalNetworking`
      + `NSLocalNetworkUsageDescription` for the LAN token server / SFU, and running `flutter run
      --release` (a debug build needs the host↔device VM-service handshake and crashes without it).
      **Out of scope:** the **Android** on-device run (no Android device on hand) — the Kotlin
      audio-session + ConnectionService layers stay written-but-device-unverified; the deeper iOS
      CallKit / Bluetooth-route behaviors also got only light on-device exercise.

### M7 — Persona fine-tuning (LoRA) *(≈ 1.5 weeks)* `[Done]`
**Goal:** train per-persona brains beyond prompting.
- [x] `training/persona_lora/`: dataset format (in-character dialogues) + curation. The format is
      the OpenAI **messages-JSONL** shape (the cascade's `Msg`), so it's consumed unchanged by
      mlx-lm's "chat" loader and converts to ShareGPT for LLaMA-Factory (`training/dataset.py`,
      validated end-to-end). **Curation is self-chat** (`training/curate.py`): the persona LLM
      answers while a *user simulator* (the same LLM, roles flipped) plays a realistic partner —
      role-flipping + simulator prompt are pure/tested; the only impurity is the injected LLM.
- [x] LoRA training, **backend-dispatched** (`training/config.py` + `train.py`): `mlx_lm lora` on
      Mac, `llamafactory-cli train` QLoRA (4-bit) on CUDA. The plan (trainer config dict +
      command) is built purely and unit-tested for both; `train.py` writes the config and shells
      out (`--dry-run` previews). `merge.py` fuses the adapter back (`mlx_lm fuse` / LLaMA-Factory
      `export`) for standalone serving. CUDA libs stay out of the light wheel (training image),
      same policy as vLLM.
- [x] Adapter hot-swap at inference + eval harness. **Hot-swap:** vLLM serves adapters via
      `--lora-modules <name>=<path>`; a persona whose `llm.lora` basename matches routes there
      automatically (`_openai_compat.lora_request_model` → request `model`; `supports_lora=True`
      on vLLM, off on LM Studio). **Eval** (`training/eval.py`): deterministic persona-adherence
      proxies (turn-style fit, spoken-clean rate, question rate vs `follow_up_probability`,
      keyword coverage) → a `persona_adherence` composite, with a prompt-only-vs-LoRA `compare`
      and a `--bare-prompt` mode that drops the derived directives (turn-style + the "speak
      without markdown" nudge) to measure what the LoRA internalizes *beyond* prompting.
- **Acceptance:** ✅ *Mac LoRA path verified live (mlx-lm), then closed end-to-end on CUDA
      (RTX 5090).* On the M4 Max the whole loop ran via mlx-lm 0.31.3 + LM Studio (**val loss
      2.74 → 2.27**; adapter loads + merges; eval `persona_adherence 0.95`). **The CUDA brain is
      now served by vLLM** (replacing the LM-Studio dev stopgap — see the M2 note), which closes
      the served-LoRA A/B that LM Studio couldn't do. Full CUDA loop, run for real on the 5090:
      **curated** 40 spoken-clean HR-interviewer dialogues via vLLM → **trained** a real QLoRA
      (LLaMA-Factory, 4-bit bnb, in a Blackwell-patched trainer image; **train_loss 0.31**,
      adapter 40.4 M params / 0.53 %, 45 s) → vLLM **hot-loaded** it (`--lora-modules
      hr_interviewer=/models/adapters/hr_interviewer`) so prompt-only + LoRA are served
      side-by-side → ran the **served A/B**. Result: with *full* prompting the strong
      Qwen2.5-7B-Instruct base is already near-ceiling, so the proxies show **parity** (≈ ±0.04);
      under a **bare prompt** the LoRA **measurably wins** — `persona_adherence` **+0.066**,
      `spoken_clean_rate` **+0.25** (base leaks markdown 25 % without the directive; the LoRA
      stays clean), `turn_style_fit` **+0.14** (concise; base rambles 66→55 words). I.e. the LoRA
      *internalizes* the persona's spoken-clean, concise, questioning behavior beyond prompting —
      the point of M7. **344 tests green** (was 266; +the spoken-clean/bare-prompt/role-anchor
      additions); ruff + mypy clean; both `BACKEND=mac|cuda server --check` PASS.
- **Setup notes (CUDA / RTX 5090):** vLLM and LLaMA-Factory run in Docker (the prod design;
      `vllm/vllm-openai:latest` v0.23 serves Blackwell sm_120 fine). **The stock
      `hiyouga/llamafactory:latest` ships torch 2.6.0/cu124, which only supports up to sm_90 and
      refuses to run on a 5090** — `training/persona_lora/Dockerfile.blackwell` overlays cu128
      torch 2.7.1 + bitsandbytes ≥0.47 (a 16 GB 4080 / sm_89 runs the stock image as-is). Train
      with the trainer container mounting the repo + sharing the vLLM HF-cache volume (no
      re-download); serve the A/B with `docker compose -f docker-compose.yml -f
      docker-compose.lora.yml up -d vllm`. Three data-quality fixes landed here: the self-chat
      **user simulator was inverting roles** (it inherited the persona's own prompt and started
      *interviewing*) → it now references the persona by name only + seeds a counterpart opener;
      curate gained a **`--spoken-clean`** pass (strip markdown from targets, since the curated
      base leaks it ~20 %); and dataset read/write are now **UTF-8** (Windows wrote cp1252, which
      the Linux trainer couldn't decode). The committed CUDA defaults stay 4080-safe (AWQ brain +
      4-bit QLoRA); this 5090 box serves the unquantized brain via `.env`.

### M8 — Memory / learn-from-conversations *(≈ 1.5 weeks)* `[Done]`
**Goal:** continuity across sessions; foundation for long-term learning.
- [x] **Per-user transcript store + rolling profile + RAG into context** (`src/personavoice/memory/`).
      `store.py` `MemoryStore` persists per-user turns (`transcript.jsonl`), a `profile.json`, and a
      `consent.json` under `<models>/memory/<user>/`. `profile.py` distills turns into a rolling
      `UserProfile` (durable facts + summary) via an injected LLM (prompt-building + JSON parsing are
      pure/tested; only `ProfileBuilder.update` calls the model). `rag.py` retrieves relevant prior
      turns — `KeywordRetriever` (pure cosine over term counts, **zero deps**, default) or an optional
      `EmbeddingRetriever` (lazy sentence-transformers, `memory-embeddings` extra) — and
      `recall_context` assembles the spoken-friendly block (profile + top-k turns) injected as a
      second system message (`persona/prompt.py build_messages(memory_context=...)`).
- [x] **Consent-gated facade + cascade wiring.** `conversation.py` `ConversationMemory` is the
      runtime API: nothing is recalled/recorded without granted consent, and it's keyed off
      `persona.memory.enabled` so it's wired in unconditionally and stays dormant until opt-in.
      `StreamingPipeline` (and so the LiveKit agent) recall before the turn, record after, and
      distill the profile on a background cadence (`summarize_every`); the agent keys memory by the
      room/job `{"user": ...}` metadata or the participant identity, and flushes distillation on
      teardown (`aclose`). `personavoice-stream-demo --user <id>` is the offline (no-LiveKit) path.
- [x] **Privacy controls: consent, local encryption, per-user delete/export.** Consent is explicit
      (recording) with a separate stricter `allow_training` opt-in. **At-rest encryption** is optional
      Fernet (`store.py` `FernetCipher`, lazy `cryptography`/`memory` extra, keyed from
      `PERSONAVOICE_MEMORY_KEY`) — each JSONL line/blob is an independent token, so append stays
      line-at-a-time; `NullCipher` (plaintext) is the dev default and `server --check` flags it.
      `personavoice-memory` is the ops/privacy CLI: `--list/--show/--grant/--revoke/--consolidate/
      --export/--delete/--distill/--gen-key`.
- [x] **Distill logs → SFT LoRA refreshes (manual trigger).** `distill.py` groups a user's
      transcripts by session, coerces them into well-formed training examples (drops dangling turns,
      merges fragments), prepends the persona prompt, and emits the **M7 messages-JSONL** dataset —
      consumed unchanged by `personavoice-train`. Strictly opt-in (`allow_training`), manual trigger
      (`personavoice-memory --distill`). *DPO/preference-pair distillation deferred* (we don't collect
      preferences yet); SFT refresh is the M8 deliverable.
- **Acceptance:** ✅ **met + live-verified on the M4 Max.** The assistant **recalls prior-session
      facts**: a real conversation → `consolidate` distilled (via **gpt-oss-20b** in LM Studio) a
      profile ("name is Sam", "has a dog named Rex", "high-school biology teacher", "hikes on
      weekends") → a fresh facade (next-session/process) recalled that block for a later query. The
      user **can wipe their data** (`--delete`) and export it (`--export`); encryption round-trips
      under a real Fernet key (`--check` reports `encrypted`, and a bad key fails the check). **341
      tests green** (was 266; +75 for M8 — store/profile/rag/conversation/distill/cli + the streaming
      integration; 0 skip in `.venv312` once `cryptography` is installed); ruff + mypy clean; both
      `BACKEND=mac|cuda server --check` PASS with the memory section. **Open (same rider as M3/M4):**
      the *live LiveKit* memory path (metadata-keyed user, mid-call recall) needs a running LiveKit
      server; the served-LoRA A/B from a distilled dataset rides the 4080 (M7's open step).

### M9 — Voice fine-tuning (high fidelity) *(≈ 1 week)* `[Partial]`
**Goal:** custom voices beyond zero-shot for key personas.
- [x] Voice fine-tune on a target speaker dataset (CUDA), in `src/personavoice/training/voice/`
      (mirrors the M7 LoRA stack: pure, unit-tested logic; heavy trainers shelled out, lazy).
      `dataset.py` is the target-speaker dataset — a `metadata.csv` of `audio|text` (the LJSpeech
      / F5 `prepare_csv_wavs` shape) with validation + STT auto-transcription. `config.py`
      `VoiceTrainConfig` → an **engine-dispatched** `VoiceTrainPlan`: **f5** (`f5-tts_finetune-cli`,
      the canonical trainer; CC-BY-NC weights → dev/personal) or **chatterbox** (MIT, a
      configurable community `trainer_script`). `finetune.py` writes any config and runs the
      prep + trainer commands (`--dry-run` previews). One CLI: `personavoice-voice-train`
      (`dataset`/`run`/`eval`/`register`/`list`). Voice fine-tuning is CUDA-centric (the Mac path
      stays zero-shot cloning).
- [x] Quality **A/B vs zero-shot** (`evaluate.py`): synthesize the same probes with the
      fine-tuned voice and the M5 clone, embed both + held-out **real** target clips (Resemblyzer,
      `voice-eval` extra), and compare **speaker similarity** — PASS when the fine-tune clears the
      target by a `--margin`. Cosine/mean/`score_ab` are pure + unit-tested with toy embeddings.
      **Fold winners into the voice registry**: `voice/finetuned.py` `FinetunedVoicesStore` is a
      non-destructive overlay (`finetuned.json` + per-persona assignments) that **outranks a
      clone** in `VoiceRegistry.resolve_for_persona` (**fine-tuned ▶ clone ▶ preset**); the cloning
      adapters (Chatterbox `from_local`, F5 `model_name`) load the trained checkpoint via the new
      `VoiceRef.model_path`. `server --check` lists fine-tuned voices + assignments and suppresses
      the distinctness warning for an assigned voice.
- **Acceptance:** ⚠️ *code-complete; the audible on-GPU A/B rides the RTX 5090 (same close-out
      pattern as M2/M7).* The whole stack is unit-tested at the logic level (dataset validation,
      engine plan-builders, prep→train ordering + failures, A/B scoring, store persistence,
      registry precedence over a clone, adapter `model_path` wiring, `--check`): **407 tests
      green** (was 344; +63 for M9; 3 skip in `.venv312`), ruff + mypy clean, both
      `BACKEND=mac|cuda server --check` PASS. CLI flows smoke-tested (`run --dry-run` for both
      engines, `dataset` from a sample manifest, `register`/`list`/`assign`). **Still open:** run
      a real fine-tune on the 5090 (F5-TTS in the CUDA training image) and confirm the A/B shows
      the fine-tuned voice **clearly higher fidelity** than its zero-shot clone (speaker-similarity
      delta over margin + a human MOS spot-check).
- **Setup notes (CUDA / RTX 5090):** the voice trainers stay out of the light `cuda` wheel (same
      policy as vLLM/LLaMA-Factory) — install F5-TTS (`pip install f5-tts`) in the CUDA training
      image; on Blackwell (sm_120) use the cu128 torch wheels (the M2/M7 notes apply). Chatterbox
      ships **no official finetune CLI**, so its `trainer_script` points at a community trainer
      (documented in `training/voice/README.md`). Licensing: an F5 fine-tune inherits **CC-BY-NC**
      (dev/personal); fine-tune **Chatterbox** (MIT) for anything shipped — same consent gate as M5.

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
                              └─▶ M6 (client, can start after M3)
M5 ─▶ M7 ─▶ M8 ─▶ M9 ─▶ M10  (training + production track)
```

- **M6 (client)** can begin in parallel once **M3** exposes a stable LiveKit endpoint.
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

- **Orchestration:** LiveKit Agents (WebRTC, VAD, turn detection, Flutter + native client SDKs). Pipecat as fallback.
- **STT:** faster-whisper / Parakeet (CUDA) · mlx-whisper / whisper.cpp (Mac).
- **LLM:** Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct · vLLM (CUDA) · LM Studio / Ollama / mlx-lm (Mac).
- **TTS:** Orpheus / Chatterbox (CUDA) · f5-tts-mlx / Chatterbox-MPS / Kokoro (Mac).
- **Training:** Unsloth / LLaMA-Factory (QLoRA, CUDA) · mlx-lm LoRA (Mac).
- **Client:** Flutter (Dart) + LiveKit Flutter SDK — one codebase for iOS + Android.
- **Infra:** Docker Compose (server), Python 3.11+, uv/poetry.

**Language decision:** Python throughout for server + ML + training (single language, all reference
models work out of the box, fine-tuning is native, LiveKit Agents Python SDK ties it together).
The client is a single Flutter (Dart) codebase shared across iOS and Android. If a polyglot orchestrator is ever wanted, the adapter layer
already isolates models so they can be exposed as HTTP/gRPC workers — but that's not the plan.

---

## 8. Open decisions (revisit as we build)

- Final TTS pick for prod: **Orpheus** (expressive, emotion tags; but weights inherit the Llama-3.2 license) vs **Chatterbox** (emotion exaggeration control; clean MIT) — decide in M2/M5 after a quality+latency bake-off. License leans Chatterbox if redistribution/commercial matters. *M2 update:* both adapters are implemented; **Chatterbox is the default for the single-4080 dockerized stack** (no second in-process vLLM, fits VRAM, MIT). Orpheus stays available as the expressive option. *M5 update:* the `orpheus-speech` engine exposes **preset voices only** (no reference-sample input), so **zero-shot cloning on CUDA goes through Chatterbox** — Orpheus is now `supports_cloning=False`. This makes Chatterbox the prod default unless an expressive, non-cloned preset is wanted. Mac cloning via F5-TTS is CC-BY-NC (non-commercial) — fine for dev, not for shipping.
- LLM base: **Qwen2.5-7B** vs **Llama-3.1-8B** — decide in M4 on persona quality.
- Memory store: lightweight (SQLite + FAISS) vs managed vector DB — decide in M8.
- Endpointing strategy: pure VAD vs semantic turn detection — tune in M3/M10.
- **Client framework: Flutter (decided).** One Flutter/Dart codebase ships the thin client to
  both iOS and Android via the official LiveKit Flutter SDK (`livekit_client`). Chosen over
  native ×2 (the UI surface is tiny — connect, persona picker, talk button, transcript, so
  two codebases aren't worth it) and over React Native / web-PWA (Flutter's media/audio
  support and LiveKit SDK fit a real-time voice app best). This is purely client-side — the
  server speaks WebRTC/LiveKit, so any LiveKit client connects unchanged. Remaining
  platform-specific work is the audio/telephony plumbing (audio session, background audio,
  CallKit/ConnectionService), which the LiveKit SDK wraps over the native WebRTC stacks.
```

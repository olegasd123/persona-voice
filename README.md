# Persona-Voice

A real-time, open-source **speech-to-speech persona system**. You talk, and a chosen
character talks back — in its own voice and style. It is built as a **modular cascade**:

```
speech in  →  STT  →  LLM brain (+ persona)  →  TTS  →  speech out
```

Three swappable stages give you voice cloning, per-persona fine-tuning, and a clear training
path — things that end-to-end speech models do not do well today.

Four personas ship out of the box:

- **PM interviewer** and **HR interviewer** — practice for job interviews.
- **Language teacher** — conversation practice (English first).
- **Companion** — casual, friendly chat.

The system runs as a **server on a CUDA GPU** in production (designed for a 16 GB card such as
an RTX 4080; verified on an RTX 5090). The same code is fully developable on a **Mac (M-series)**
through Mac-native backends. The client is thin: a single **Flutter** app for iOS and Android
that only captures and plays audio. All the models run on the server.

## What it can do

- **Hold a real-time spoken conversation.** The reply **streams** sentence by sentence, so the
  persona starts speaking before the full answer is ready. You can **interrupt it** (barge-in)
  and it stops and listens.
- **Switch personas at runtime** — by config, by API metadata, or mid-call from the client.
  Each persona has its own prompt, behavior knobs, and a distinct voice.
- **Clone a voice** from a short (~10 s) sample and make a persona speak in it.
- **Fine-tune a persona "brain"** with a LoRA adapter when prompting is not enough.
- **Fine-tune a high-fidelity voice** for a target speaker, beyond a zero-shot clone.
- **Call tools mid-conversation** — a persona can opt into function calls (e.g. checking the
  clock) that run server-side before it answers, on a tool-capable LLM backend.
- **Remember a user across sessions** — a consent-gated, per-user memory that recalls earlier
  facts in later calls.
- **Run the same code on two very different machines** (a Mac for development, a CUDA GPU for
  production) by flipping one `BACKEND` switch.

## How it works

```
┌─────────────┐   WebRTC (Opus)   ┌──────────────────────────────────────────────┐
│ Flutter app │  ───────────────▶ │            Server (CUDA GPU / Mac)           │
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

Five design ideas hold it together:

1. **A cascade, not one big model.** Three separate stages (STT, LLM, TTS) are easy to swap,
   clone, and train.
2. **Backends sit behind an interface.** Every stage has a base class and a `BACKEND=mac|cuda`
   switch. The persona logic, the orchestration, and the client never change between machines.
3. **Stream everything.** Transcribe while the user talks; feed LLM tokens into the TTS sentence
   by sentence. This is what keeps the perceived latency low.
4. **Thin client, fat server.** The client only moves audio over WebRTC. All models run on the
   server.
5. **Personas are config plus adapters.** A persona is a system prompt, optional LoRA adapter,
   a voice, and a few behavior knobs — written in YAML and hot-swappable.

### Backends (which model runs where)

| Stage | Mac (dev) | CUDA (prod) | VRAM (CUDA) |
|-------|-----------|-------------|-------------|
| **STT** | `whisper_mlx` | `faster_whisper` / `parakeet` | ~2 GB |
| **LLM** | `lmstudio` / `ollama` / `mlx_lm` | `vllm` | ~5–6 GB |
| **TTS** | `kokoro` (fast, no clone) | `chatterbox` (clone) | ~3–4 GB |
| **Glue** | LiveKit Agents | LiveKit Agents (identical) | — |
| **Train** | `mlx-lm` LoRA | LLaMA-Factory QLoRA / Chatterbox voice tuning | — |

Select a backend with `BACKEND=mac|cuda`. Each backend's `config/backends/<backend>.yaml` names
the adapter, the model, and the per-adapter options. Environment interpolation
(`${VAR:-default}`) is supported, so one file can drive both the production stack and a dev box.

### Latency budget (CUDA, time to first audio)

| Step | Target |
|------|--------|
| Endpointing (VAD silence) | ~200 ms |
| STT finalize | ~150 ms |
| LLM time-to-first-token | ~200 ms |
| TTS first audio chunk | ~200 ms |
| WebRTC round-trip (local) | ~50 ms |
| **Total to first audio** | **~600–900 ms** |

On CUDA (RTX 5090) the streaming path hits this: with **Kokoro** (fast, no clone) the warm
time-to-first-audio is about **0.58 s**; with **Chatterbox** (the cloning backend) about
**1.3 s**. Both are selected by one environment variable, so "low latency" vs "cloning" is a
one-line switch. A Mac is slower (TTS is the bottleneck) — use Kokoro for fast dev iterations.

## Project status

The system is feature-complete across the whole cascade: offline loop, streaming, the live
LiveKit agent, four personas, voice cloning, persona LoRA training, cross-session memory, voice
fine-tuning, and a hardening pass (eval, observability, security). It is verified at the logic
level by the test suite (`pytest`, plus `ruff` and `mypy`), and the heavy or hardware-bound steps
are verified on real hardware (a Mac M4 Max for dev, an RTX 5090 for CUDA).

One thing needs a running **LiveKit server** to exercise live: the browser/phone back-and-forth
with barge-in over WebRTC. That path is **verified on a real iPhone** (a full spoken conversation
over the LAN against a self-hosted LiveKit SFU). The **Android** client code is written and
analyzed, but it is **not verified on a device** (no Android hardware on hand).

## Quickstart

Requires Python 3.11+ (use **Python 3.12** for the Mac model libraries — see the note below).

```bash
# 1. Create a virtualenv and install the core package plus dev tools.
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

# 2. Configure.
cp .env.example .env        # defaults to BACKEND=mac

# 3. Validate the config and load the adapters.
python -m personavoice.server --check
#   ...or: BACKEND=cuda python -m personavoice.server --check

# 4. Run the tests.
pytest
```

The core install is intentionally light, so `--check` and the tests run on any machine without
heavy ML wheels. To install the backend model libraries:

```bash
pip install -e '.[mac]'     # Mac: mlx-whisper, mlx-lm, kokoro, ...
pip install -e '.[cuda]'    # CUDA: faster-whisper, ...
```

> **Use Python 3.12 for the `mac` extra.** Kokoro pulls in spaCy/`blis`, which has no wheels for
> Python 3.13/3.14. With [uv](https://docs.astral.sh/uv/):
> ```bash
> uv venv --python 3.12 .venv312 && source .venv312/bin/activate
> uv pip install -e '.[mac]'
> ```

## Usage

### Offline voice loop (one machine, no LiveKit)

Run a single turn through the cascade: a spoken `.wav` question in, a synthesized reply out.
Needs the `mac` extra and a local LLM server.

```bash
# LLM: start LM Studio's local server and load a model (the default config expects
# openai/gpt-oss-20b). Check it is up:  curl http://localhost:1234/v1/models
# (Prefer Ollama? set adapter: ollama in config/backends/mac.yaml, then `ollama serve`.)

# Speak into a prerecorded wav and hear the reply:
personavoice-demo --wav question.wav --persona companion --play

# ...or record from the mic (needs sounddevice + a mic):
personavoice-demo --record 5 --persona language_teacher --play
```

It prints the transcript, the persona's reply, and per-stage timings, and writes the spoken reply
to `reply.wav` (override with `--out`).

> **Reasoning models:** local models like Qwen3 / gpt-oss emit a hidden thinking trace
> (`reasoning_content`) that the adapter never speaks. Keep it short with
> `extra_body: {reasoning_effort: low}` (already set in `mac.yaml`) so replies start fast and do
> not spend `max_tokens` on thinking.

### Streaming voice loop (one machine, no LiveKit)

Same cascade, but the reply is **streamed sentence by sentence** instead of waiting for the whole
answer. The persona starts speaking sooner.

```bash
# Hear each sentence as soon as it is synthesized.
personavoice-stream-demo --wav question.wav --persona companion --play
personavoice-stream-demo --record 5 --persona language_teacher --play
```

It writes one wav per sentence to `reply_stream/` and prints **time-to-first-token** and
**time-to-first-audio** (the moment speech starts), which is well below the turn-based total
because TTS overlaps LLM generation.

### Live LiveKit agent (real-time, with barge-in)

For a live conversation over WebRTC, run the LiveKit Agents worker. It needs the `livekit` extra
and a LiveKit server (cloud or self-hosted):

```bash
pip install -e '.[livekit]'          # livekit-agents + Silero VAD plugin
# Set LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET in .env, then:
personavoice --serve                 # production worker (BACKEND from env)
personavoice-agent dev               # hot-reload dev worker
```

The agent uses Silero VAD to find the end of each utterance, transcribes it, and streams the
reply onto the published audio track. When VAD detects the user starting to speak, the
`TurnController` cancels the in-flight LLM + TTS and clears the output queue (**barge-in**).
Connect any LiveKit client — the [Agents Playground](https://agents-playground.livekit.io/) or
the Flutter app — to talk.

**Semantic endpointing** (optional, `PERSONAVOICE_SEMANTIC_ENDPOINTING=1`) layers a cheap
"did they actually finish?" gate on top of the VAD silence. Pure VAD treats every pause as the
end of a turn, so a mid-thought pause ("I think… *pause* …it's fine") gets answered too early.
With it on, a transcript that reads as unfinished — trails off into an ellipsis ("I think…"),
ends on a trailing conjunction/preposition/article or a filler like "um", or trails off on a
hedging lead-in like "I guess" / "the thing is" (`orchestrator/completion.py`) — is **held** and
merged with the next utterance into one turn. A grace window (`PERSONAVOICE_ENDPOINTING_GRACE_MS`, default 1500 ms) bounds the
cost of a misjudged hold: if no continuation arrives, the held text is answered as-is. The gate is
deliberately conservative (it only holds on clear continuation cues), so the worst case is that
delay, never a dropped turn.

### Token server and self-hosted LiveKit

A real client cannot join a room without a token, and it must never see the LiveKit secret. The
**token server** mints short-lived join tokens and tells the client where to connect ("fat
server, thin client"). It is pure standard library (no extra needed):

```bash
# Set LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET in .env, then:
personavoice --token-server          # HTTP on PERSONAVOICE_HOST:PERSONAVOICE_PORT (default :8080)
```

| Route | Purpose |
|-------|---------|
| `GET /healthz` | liveness; adds `sessions_active` / `sessions_max` when the worker is reporting load |
| `GET /metrics` | Prometheus exposition (per-turn latency/outcome counters + session load); empty unless the `metrics` extra is installed |
| `GET /personas[?user=]` | `{"personas": [{"id","name","description","voice","custom"}], "default": <id>}` — curated + the user's custom personas |
| `POST /personas?user=` | create a custom persona (body = a persona draft) |
| `POST /personas/draft?user=` | draft a persona from `{"description": ...}` via the cascade LLM — validated, **not** persisted (drop into the edit form) |
| `PUT /personas/{id}?user=` / `DELETE /personas/{id}?user=` | edit / delete one of the user's own personas (curated are read-only) |
| `GET /loras` | served LoRA adapters for a custom persona's `llm.lora` (`{"loras": [...], "supports_lora"}`) |
| `GET /voices` | selectable voice catalog for the active backend (`{"voices": [...], "supports_cloning"}`) |
| `POST /token` | body `{"room"?, "identity"?, "name"?, "persona"?, "voice"?, "cefr"?, "demeanor"?, "user"?}` → `{"url","token","room","identity","name","persona","voice","cefr","demeanor","user"}` |
| `POST /voices/clone` | `?name=&authorized=1` + the wav as the raw body → enroll a clone |
| `DELETE /voices/clone/{name}` | remove a cloned voice |

`room`/`identity` are generated when omitted. `identity` is the unique LiveKit participant id
(`sub`); `name` is a cosmetic display-name claim that falls back to `identity`; `user` scopes a
caller's custom personas and memory (see below). The persona and the per-session options (voice /
CEFR / demeanor — see below) are validated, embedded in the token metadata, and echoed back, so
the client can apply them with a data message after connecting. If `PERSONAVOICE_API_TOKEN` is
set, requests need `Authorization: Bearer …`. Tokens are standard HS256 JWTs in LiveKit's
documented format (`server/tokens.py`).

### Session options (voice / CEFR / demeanor)

Three per-conversation knobs, chosen before a call and swappable mid-call, layered on top of the
persona (the client only *chooses*; the server *applies*):

- **voice** — speak with any voice from the library (`GET /voices`) instead of the persona's
  assigned voice, for this session only.
- **cefr** — a CEFR level (`A1`…`C2`) that calibrates language difficulty for learners.
- **demeanor** — `kind`, `natural` (the persona as authored), or `rude` (blunt but **bounded** —
  brusque, never abusive).

They ride the existing metadata/data-message rail: pass them to `POST /token` (validated +
embedded in metadata) or send a `{"voice":…, "cefr":…, "demeanor":…}` data message to change them
mid-call. The offline demos take `--voice/--cefr/--demeanor` so the behavior is testable without a
client (e.g. `personavoice-stream-demo --wav q.wav --persona language_teacher --cefr a2`).

**Moderation.** A pluggable input/output guard (`safety/`) wraps the LLM turn — **on by default**
(the dependency-free rule guard), disabled with `PERSONAVOICE_MODERATION=none`. The rule guard
short-circuits a crisis/self-harm utterance to a calm, resource-pointing reply, blocks any reply
that *encourages* self-harm, and bounds abusive output — which is what *guarantees* the `rude`
demeanor stays "brusque, not abusive". On the streaming path the output bound runs per sentence,
just before each is voiced, so it applies live without buffering the whole reply.

**Dynamic emotion (per-utterance prosody).** Off by default; set `PERSONAVOICE_DYNAMIC_EMOTION=1`
to have the persona color each reply individually. The model is asked to prefix a reply with one
`[emotion]` tag (`neutral`, `happy`, `excited`, `calm`, `sad`, `serious`, `warm`, `curious`,
`sympathetic`); the pipeline strips the tag before TTS, history, and the transcript, and applies it
to the voice for that turn only. It's audible on a cloning backend (Chatterbox maps it to its
`exaggeration` knob); a preset-only backend (Kokoro) ignores it gracefully, so the tag is a no-op
there. Static per-persona `emotion` (in `config/voices.yaml` / a persona's `voice.emotion`) is
unchanged and still the baseline when no tag is emitted.

**Self-host the SFU.** `docker-compose.livekit.yml` brings up a LiveKit server (dev keys
`devkey`/`secret`) plus the token server:

```bash
docker compose -f docker-compose.livekit.yml up --build       # LiveKit SFU + token server
LIVEKIT_URL=ws://localhost:7880 LIVEKIT_API_KEY=devkey \
  LIVEKIT_API_SECRET=secret BACKEND=mac personavoice --serve   # the conversation agent
```

`LIVEKIT_URL` is the address the **client** dials, so from a phone use the machine's LAN IP
(`ws://192.168.x.y:7880`) and open UDP 7882. Use TLS (`wss://`) and a real key/secret in
production.

### Tool / function calling

A persona can **call a tool mid-turn** instead of answering blind — the companion can check the
wall clock, and the registry is the seam for a teacher's dictionary or an interviewer's question
bank. A persona opts in by listing tool names in its YAML:

```yaml
# config/personas/companion.yaml
tools:
  - get_current_time
```

When a persona lists tools, the turn runs a bounded loop: the model is offered the tool schemas,
and if it calls one the server executes it, feeds the result back, and lets the model answer.
**Nothing is voiced during the tool round-trips** — TTS is deferred until the follow-up reply
streams. A persona with no `tools:` is a pure conversationalist and is never sent any schema, so
existing personas are unchanged.

- **Backend support.** Routed on the OpenAI-compatible LLM backends (vLLM prod, LM Studio dev);
  vLLM needs `--enable-auto-tool-choice --tool-call-parser <model-parser>` — the compose file
  already passes these (`hermes` for Qwen2.5, overridable via `VLLM_TOOL_PARSER`). Other backends
  (Ollama, mlx-lm) ignore the schemas and just answer — the capability degrades gracefully.
- **Bounds.** `PERSONAVOICE_TOOL_MAX_ITERS` (default 4) caps the model↔tool round-trips before a
  plain answer is forced; `PERSONAVOICE_TOOL_TIMEOUT` (default 10 s) bounds each tool call. A
  hallucinated tool, bad arguments, a timeout, or a handler error become a short error string fed
  back to the model rather than failing the turn.
- **Built-ins.** Only safe, side-effect-free, offline tools ship (`get_current_time`); anything
  networked or side-effecting (weather, web search) must be added behind explicit gating. New
  tools are registered in `orchestrator/tools.py`. `personavoice.server --check` lists the
  registry and warns when a persona references an unknown tool or one the active backend can't
  route.

### Flutter client (iOS + Android)

`client/` is the cross-platform app. It fetches the persona list, asks for a token, connects,
publishes the mic, shows a live transcript, and switches persona mid-call. The mic runs in
**open-mic** mode (the server's VAD decides turns) or **push-to-talk** (hold to talk); audio goes
to the loudspeaker by default; the status line shows LiveKit reconnects. The agent also publishes
its spoken reply as a transcript, so the assistant's words appear in the client view.

The **live voice loop is verified on a real iPhone** against the self-hosted LiveKit stack. The
Android code is written and analyzed but **not verified on a device**. See
[client/README.md](client/README.md) for the full setup, the native audio-session and telephony
plumbing, and the iOS build notes.

### Personas and voices

A persona is a YAML file (`config/personas/*.yaml`): a system prompt plus knobs for the LLM
(`temperature`, `max_tokens`), behavior (`turn_style`, `follow_up_probability`), and a logical
`voice.ref`. The four shipped personas are `pm_interviewer`, `hr_interviewer`,
`language_teacher`, and `companion`. Any demo or the live agent takes `--persona <id>`.

Each persona points at a **logical voice** (for example `voices/companion_soft`). The **voice
registry** (`config/voices.yaml`) maps that to a concrete preset per TTS backend, so the personas
sound distinct on whichever backend is active:

```yaml
# config/voices.yaml
companion_soft:
  emotion: warm
  presets: { kokoro: af_heart }   # Mac preset
pm_calm:
  presets: { kokoro: am_michael }
```

A cloning backend like Chatterbox has no preset and falls back to its default voice, unless a
**clone is assigned** to the persona (see below). `python -m personavoice.server --check` lists
the loaded personas, voices, and clones, and warns if a persona will not sound distinct on the
active backend.

**Selecting a persona on the live agent.** The agent picks the persona in this order: the
dispatch's job metadata, the room metadata (`{"persona": "hr_interviewer"}`), then the
`PERSONAVOICE_PERSONA` environment variable (otherwise the first registered persona). A connected
client can also **switch persona mid-call** by publishing a data message — a bare id
(`hr_interviewer`) or `{"persona": "hr_interviewer"}`. The swap keeps the conversation history,
and the registry reloads from disk first, so editing a persona file takes effect without a
restart.

**Custom personas (multi-user).** Beyond the curated YAML, users can author their own personas at
runtime over HTTP, scoped per `user_id` (the `?user=` query / `{"user": …}` token metadata): `POST
/personas?user=`, `PUT`/`DELETE /personas/{id}?user=`, and `GET /personas?user=` (which merges the
curated set with that user's own, each tagged `custom`). A draft is just a persona body — `name`
and `system_prompt` are enough; the voice ref and LLM base model default to the curated default,
and a `session_defaults` block bakes in CEFR/demeanor/voice the agent layers under any explicit
session option. They persist to `PERSONAVOICE_USER_PERSONAS` (`<models>/user_personas.json`) and
the agent resolves a `{"persona": <id>, "user": <uid>}` against this store — **curated always win
on id clash**, so a user can't shadow or delete a built-in. A persona's `llm.lora` routes to a
vLLM-served adapter; `GET /loras` lists the selectable ones (empty on Mac/LM Studio, where a LoRA
is merged into the base model at train time).

**Authoring helper (draft from a description).** Writing a persona from scratch is a blank-page
problem, so an optional generator turns a one-line description into a validated draft. `POST
/personas/draft?user=` with `{"description": "a patient French tutor who only speaks in B1"}` asks
the configured cascade LLM (LM Studio on Mac — fully offline) to fill the persona shape, then
validates it against the `Persona` model and **clamps** anything unservable: the prompt enumerates
the legal `voice.ref` ids and served LoRA names, an invalid choice is repaired to a legal one, a
bad draft gets one repair retry, the id is de-duped against the curated + the user's own, and
`demeanor` is never authored to `rude`. The draft is **returned, not persisted** — the client drops
it into the existing New/Edit form, the user tweaks and confirms, and `POST /personas` does the
write (human in the loop). The same path is on the CLI:

Option A — run as a module (no reinstall)

```bash
cd /Users/oleg/Dev/Common/persona-voice
.venv/bin/python -m personavoice.persona.cli draft "a patient French tutor who only speaks in B1"   # validated YAML → stdout
.venv/bin/python -m personavoice.persona.cli draft "a patient French tutor who only speaks in B1" --out config/personas/{take id from validated YAML}.yaml
```

Option B — register the short command, then use it

```bash
cd /Users/oleg/Dev/Common/persona-voice
.venv/bin/pip install -e . --no-deps        # regenerates the console scripts
.venv/bin/personavoice-persona draft "a patient French tutor who only speaks in B1"
```

### Voice cloning (zero-shot)

Clone a voice from a short sample and make a persona speak in it. Cloning needs a cloning TTS
backend — `chatterbox` on CUDA (Kokoro is preset-only):

```bash
# Install the cloning backend (one-time):
pip install -e '.[clone]'            # Chatterbox
# Then set the CUDA TTS adapter to chatterbox in config/backends/cuda.yaml.

# Clone from a wav (or --record 10 from the mic) and assign it to a persona:
personavoice-clone --sample me.wav --name my_voice --assign companion
personavoice-clone --record 10 --name my_voice --assign companion --say "Hello!" --play

personavoice-clone --list             # cloned voices + their assignments
personavoice-clone --unassign companion
```

A clone is a stored reference sample plus its transcript (auto-filled by the cascade's STT).
It lands under the clones directory
(`PERSONAVOICE_CLONES_DIR`, default `<models>/clones`) with a `clones.json` manifest, so it
**survives restarts** and the live agent picks it up at startup. Assigning a clone is
non-destructive: it overlays the voice registry at resolve time and is only honored on a backend
that can clone. Switching back to Kokoro simply restores the persona's preset voice.

**Enroll from a client.** Besides the CLI, the token server exposes the library over HTTP:
`GET /voices` lists every selectable voice (presets + clones + fine-tunes) with an `available`
flag per the active backend, `POST /voices/clone` enrolls one from an uploaded wav (requires
`authorized=1`, since cloning a real person's voice is sensitive), and `DELETE /voices/clone/{name}`
removes it. A picked voice is then chosen per session via the **voice** session option above.
Because clones/fine-tunes are only **audible on a cloning backend**, the catalog still lists them
on Kokoro but marks them `available:false` with a reason. A "bring your own voice" session
should run `chatterbox`.

**Bundled voices.** The picker isn't empty on first run: `assets/seed_voices/` ships
`Feminine.wav` + `Masculine.wav`, wired as **presets** in `config/voices.yaml` (their `sample`
field). On a cloning backend (Chatterbox on CUDA) the backend zero-shot-clones the sample, so
they're speakable out of the box with no enrollment step. They don't map onto Kokoro, so they
don't appear on the Mac backend.

To enroll **your own** wavs as clones, `personavoice-seed-voices`
(`python scripts/seed_voices.py`) enrolls every wav in `assets/seed_voices/` as a clone named
after its file stem. It goes through the same path a client upload uses and is **idempotent**
(already-present clones are skipped; `--force` re-enrolls). On a preset-only backend the seeds
still enroll but list `available:false`.

### Persona fine-tuning (LoRA)

Train a per-persona LoRA when prompting is not enough. One CLI, `personavoice-train`, drives the
whole loop (the heavy trainers are shelled out to, so the repo installs and tests without a GPU):

```bash
pip install -e '.[train]'        # Mac: mlx-lm (also in '.[mac]')

# 1. Curate in-character data by self-chat (the persona LLM vs a user simulator).
personavoice-train curate --persona hr_interviewer --num 20 --exchanges 4

# 2. Train (mlx-lm on Mac; LLaMA-Factory QLoRA on CUDA with BACKEND=cuda).
personavoice-train run --persona hr_interviewer --dry-run    # preview config + command
personavoice-train run --persona hr_interviewer              # launch

# 3a. Serve it: hot-swap with vLLM, then set the persona's llm.lora to the adapter name.
#     docker compose -f docker-compose.yml -f docker-compose.lora.yml up -d vllm
# 3b. ...or merge it into a standalone checkpoint.
personavoice-train merge --persona hr_interviewer --adapter models/adapters/hr_interviewer

# 4. Score persona adherence (prompt-only vs LoRA).
personavoice-train eval --persona hr_interviewer --compare [--bare-prompt]
```

The dataset is the cascade's **messages-JSONL** shape, so curated data trains unchanged (and
converts to ShareGPT for LLaMA-Factory). For the format, the CUDA dataset registration, the
Blackwell trainer image, and the hyperparameters, see
[training/persona_lora/README.md](training/persona_lora/README.md).

A strong instruct base with the **full** persona prompt is already near its ceiling on the
proxies, so the LoRA shows up as parity. With `--bare-prompt` (which drops the derived
spoken-clean and turn-style directives) the LoRA measurably wins — it *internalizes* the persona's
behavior beyond prompting.

### Memory across sessions

The assistant can remember a user across calls. Each turn it injects a compact **memory block**:
an LLM-distilled profile (durable facts plus a rolling summary) and the most relevant prior turns
for what was just said. It is **consent-gated** — nothing is stored or recalled until a user opts
in — and keyed per user (`{"user": "..."}` room/job metadata, or the LiveKit participant
identity). The core memory (store, profile, keyword recall, distill) needs **no extra**.

A user can set their own consent from the client: the app's **Settings → Privacy** toggle posts
to the token server (`GET`/`POST /consent?user=<id>`), writing the same `consent.json` as the
`personavoice-memory --grant`/`--revoke` CLI below — the CLI stays the operator backstop.

```bash
# Try it offline (no LiveKit). --user opts that id in and keys their memory:
personavoice-stream-demo --wav intro.wav --persona companion --user sam   # records + distills
personavoice-stream-demo --wav later.wav --persona companion --user sam   # recalls prior facts

# Manage and inspect memory (the privacy surface):
personavoice-memory --list                       # users, consent, turn/session counts
personavoice-memory --show sam                    # profile (facts + summary) + recent turns
personavoice-memory --grant sam [--training]      # opt in (recording / training use)
personavoice-memory --consolidate sam             # (re)distill the profile via the LLM
personavoice-memory --export sam --out sam.json   # portable dump   (privacy)
personavoice-memory --delete sam                  # wipe everything (privacy)
personavoice-memory --distill sam --out sam.jsonl # transcripts -> a persona-LoRA dataset (opt-in)
```

**Encryption at rest** is optional: set `PERSONAVOICE_MEMORY_KEY` (generate one with
`pip install -e '.[memory]'` then `personavoice-memory --gen-key`) and stored conversations are
Fernet-encrypted; leaving it unset stores plaintext (the dev default, flagged by `--check`).
**Semantic recall** is an optional upgrade (`pip install -e '.[memory-embeddings]'`); the default
keyword retriever needs nothing. Per-user `--export` and `--delete` satisfy the data
portability/wipe controls, and transcripts can also **distill** into the persona-LoRA dataset
(strictly opt-in via `--training`).

### Voice fine-tuning (high fidelity)

When a zero-shot clone is not faithful enough, **fine-tune** a high-fidelity voice for a target
speaker. Where a clone is reference conditioning, a fine-tune adapts the TTS model itself to the
speaker on a small dataset. One CLI, `personavoice-voice-train`, drives the loop. It is
**CUDA-only** (the heavy trainers live in the CUDA training image, so the repo installs and tests
without a GPU):

```bash
pip install -e '.[voice-eval]'        # speaker-similarity A/B (Resemblyzer)

# 1. Build the target-speaker dataset (metadata.csv of audio|text; auto-transcribe with the STT).
personavoice-voice-train dataset --voice my_voice --audio-dir clips/ --probe-durations

# 2. Fine-tune with Chatterbox. Preview, then launch on the GPU.
personavoice-voice-train run --voice my_voice --dry-run
personavoice-voice-train run --voice my_voice --config training/voice/configs/my_voice.chatterbox.yaml

# 3. A/B vs the zero-shot clone — speaker similarity to held-out real target clips.
personavoice-voice-train eval --voice my_voice --clone my_clone --target-dir held_out/ --margin 0.02 --register

# 4. Fold the winner into the registry and assign it (it outranks a clone for that persona).
personavoice-voice-train register --voice my_voice --checkpoint models/finetuned/my_voice --assign companion
personavoice-voice-train list
```

A fine-tuned voice lands under `<models>/finetuned` with a `finetuned.json` manifest. Once
assigned, it takes precedence in the registry — **fine-tuned ▶ clone ▶ preset** — and the cloning
adapter loads the trained checkpoint via `VoiceRef.model_path`. For the dataset format and
trainer config, see
[training/voice/README.md](training/voice/README.md).

Only fine-tune voices you are authorized to use.

### Run on CUDA (production)

The `cuda` backend mirrors the Mac cascade: `faster_whisper`/`parakeet` (STT), `vllm` (LLM),
`chatterbox` (TTS). The LLM is served by a separate **vLLM** process so the app image
stays light; the STT and TTS models run in the persona-voice container.

```bash
# Bring up vLLM (Qwen2.5-7B-AWQ) + the persona-voice server on the GPU box.
docker compose up --build
#   vllm:         http://localhost:8000/v1   (OpenAI-compatible)
#   personavoice: validates the cuda config against vLLM
```

Both services share one GPU. The compose file documents the VRAM budget (about 10–12 GB: vLLM
4-bit ~6–7 GB + STT ~2 GB + Chatterbox ~2–3 GB, within 16 GB) and caps vLLM's
`--gpu-memory-utilization` so STT/TTS fit.

Without Docker, run the pieces directly: `vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --quantization
awq`, then `BACKEND=cuda personavoice-demo --wav question.wav` (needs the `cuda` extra plus
`chatterbox-tts`).

### Benchmarking latency

`scripts/bench_latency.py` runs the turn-based pipeline N times and reports per-stage
min/median/mean/max — use it to compare machines and to catch regressions:

```bash
python scripts/bench_latency.py --backend mac  --wav question.wav --runs 5
python scripts/bench_latency.py --backend cuda --wav question.wav --runs 5 --json cuda.json
```

For the streaming "time to first audio", add `--stream`. You can sweep the TTS chunk sizes
(`--first-chunk-chars` / `--max-chunk-chars`) and gate a run against the latency budget with
`--budget-ms` (it exits non-zero if the median misses the budget):

```bash
# Sweep the first-chunk size and gate against the 900 ms time-to-first-audio budget.
python scripts/bench_latency.py --backend cuda --wav q.wav --stream \
    --first-chunk-chars 60 --budget-ms 900
```

The most effective latency lever is the early **first-chunk** clause break: it lets the persona
start speaking before its opening sentence ends.

### Eval, observability, and security

**Automated eval and a dashboard.** Score STT word-error-rate, then gate the headline metrics
green/red. `run_eval.py` exits non-zero when any metric is red, so it works as a CI gate:

```bash
python scripts/eval_stt.py --backend cuda --manifest clips.jsonl --json wer.json
python scripts/run_eval.py --latency-json lat.json --wer-json wer.json \
    --metrics persona.json --mos-ratings mos.json   # persona.json = {"persona_adherence": 0.92}
```

**Observability.** Set `PERSONAVOICE_LOG_FORMAT=json` for structured logs. The live agent logs a
per-turn metrics record (STT / first-token / first-audio / total, plus barge-in and errors). A
failed turn is logged and skipped — it never crashes the worker.

The same per-turn record is also exported for **Prometheus**: the token server exposes
`GET /metrics` (alongside `/healthz`, unauthenticated and un-throttled) with
`personavoice_turns_total{persona,outcome}` and latency histograms
(`personavoice_{stt,first_token,first_audio,turn}_seconds`). The exporter is the optional
`metrics` extra (`pip install -e '.[metrics]'`, already in the CUDA server image); without it
`/metrics` just serves an empty body. Because the agent worker (which records turns) and the token
server (which serves `/metrics`) are usually separate processes, set the standard
`PROMETHEUS_MULTIPROC_DIR` to a shared, writable directory on both so their metrics aggregate;
otherwise each process exports only its own.

**Concurrency / admission control.** A single GPU holds one conversation: a second concurrent
caller runs in a second process that loads its own STT+TTS copy and OOMs a 16 GB card. The agent
worker therefore admits **one session at a time** by default — set `PERSONAVOICE_MAX_SESSIONS` to
change the cap, but scale out by running more workers (LiveKit balances across them) rather than
raising it on one box. The worker reports its live load to LiveKit so the server stops dispatching
once it's full, and is the backstop for an over-capacity job that still slips through. The count is
visible on `GET /healthz` (`sessions_active` / `sessions_max`) and `GET /metrics`
(`personavoice_sessions_active`, `personavoice_sessions_max`, `personavoice_sessions_rejected_total`).

Rather than strand a turned-away caller in a silent room, an over-capacity job is briefly admitted
to play a **"busy" clip** (a synthesized telephone busy tone by default, or your own WAV via
`PERSONAVOICE_BUSY_CLIP`) plus a `{"event":"busy"}` data message, then disconnected — no models are
loaded on that path. For a nicer UX you can also enable the **token-server early gate** with
`PERSONAVOICE_ADMISSION_503=1`: `/token` then returns `503 + Retry-After` once the worker reports
full, so the client backs off before connecting. It reads the worker's live count, so it needs a
shared `PROMETHEUS_MULTIPROC_DIR` on both the worker and the token server, and is an optimization on
top of the worker's load gate (it has a read→mint→connect race) — it fails open and mints when the
count is unreadable.

**Worker job executor (warm reuse).** On the Mac dev backend the worker runs each conversation in a
**thread** (`PERSONAVOICE_JOB_EXECUTOR=thread`, the Mac default), so the models prewarmed at startup
are reused across calls — a new call right after ending one is warm immediately. LiveKit's off-Windows
default is a **process** per job, which reloads every model each conversation; on a single-session
worker that makes every call pay the cold start (and back-to-back calls stall on the next prewarm).
Set `PERSONAVOICE_JOB_EXECUTOR=process` to opt back into per-job isolation.

**Load test and security.** Hammer the token server and confirm rate limiting kicks in:

```bash
PERSONAVOICE_RATE_LIMIT_RPS=20 personavoice --token-server   # token bucket, 429s over budget
python scripts/loadtest.py --url http://localhost:8080 --requests 500 --concurrency 20 --token "$TOK"
```

In production, set `PERSONAVOICE_REQUIRE_AUTH=1` (the server then refuses to start wide-open), a
strong `PERSONAVOICE_API_TOKEN`, and either `PERSONAVOICE_TLS_CERT`/`_KEY` or a TLS-terminating
proxy. The token comparison is constant-time. See the hardening block in `.env.example` for the
full production checklist.

## Repository layout

```
config/
  backends/{mac,cuda}.yaml   # which adapter + model per stage; the BACKEND switch
  personas/*.yaml            # the four personas (prompt + voice + behavior)
  voices.yaml                # voice registry: logical voice -> per-backend preset
src/personavoice/
  models.py                  # Persona, VoiceRef, VoiceDef, Transcript, Msg, configs
  adapters/                  # stt/ llm/ tts/ — base classes + per-backend impls + factory
  persona/                   # loader, prompt builder, registry
  voice/                     # registry + zero-shot clones + fine-tuned voices
  server/                    # settings, config, --check/--serve/--token-server; tokens.py
  orchestrator/              # turn-based pipeline + chunker/streaming/turn/agent/endpointing/completion/tools
  training/                  # persona LoRA + voice/ fine-tuning: dataset/config/finetune/eval
  memory/                    # per-user store + profile + RAG + distill
  eval/                      # WER + MOS + dashboard gating
  obs/                       # structured logging + per-turn latency metrics + Prometheus exporter
client/                      # Flutter app (iOS + Android), LiveKit SDK
training/persona_lora/       # LoRA configs, seed datasets, workflow docs
training/voice/              # voice fine-tune configs, sample dataset, workflow docs
scripts/
  download_models.py         # pinned model manifest + downloader
  bench_latency.py           # per-stage latency benchmark + budget gate (mac/cuda)
  eval_stt.py                # STT word-error-rate over a clip manifest
  run_eval.py                # aggregate eval artifacts into a green/red dashboard
  loadtest.py                # token-server concurrency load test
docker-compose.yml           # CUDA stack: vLLM + persona-voice server
docker-compose.livekit.yml   # self-hosted LiveKit SFU + token server
Dockerfile                   # CUDA server image (STT + TTS)
tests/
```

## Models and licenses

`python scripts/download_models.py --backend <mac|cuda> --list` prints the pinned manifest.

| Stage | Mac | CUDA | License |
|-------|-----|------|---------|
| STT | `mlx-community/whisper-large-v3-turbo` | `Systran/faster-whisper-large-v3` | MIT |
| LLM | LM Studio / Ollama (any loaded model) | `Qwen/Qwen2.5-7B-Instruct` (vLLM) | model-dependent |
| TTS | `hexgrad/Kokoro-82M` | `ResembleAI/chatterbox` | MIT |

Licenses were verified against each model card (2026-06).

> Licenses drift — **re-verify before any redistribution.**

## Development

```bash
pip install -e '.[dev]'
pre-commit install      # ruff lint + format on commit
ruff check . && ruff format --check .
mypy src
pytest
```

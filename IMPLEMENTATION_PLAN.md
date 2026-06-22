# Persona-Voice — Implementation Plan (Next Features)

A consolidated, detailed plan for the next round of functionality. It merges two sets of
ideas:

1. **Capability / production gaps** surfaced from a code read (tool calling, session reports,
   safety, dynamic emotion, semantic endpointing, metrics, concurrency, web client, persona
   authoring, memory introspection).
2. **Session-personalization features** requested directly: **pick a voice before a
   conversation** (built on **multi-voice cloning**), **CEFR level** for language learners, and
   **person type / demeanor** (kind / natural / rude).

Most of this is still spec, not code. File paths and function names refer to the current tree so
each item is grounded and actionable. See the **Status** section below for the client/server
scaffold that has since landed and what it changes about the remaining work.

---

## Status (as of 2026-06-23)

A first **client scaffold + a thin slice of the `/personas` enrichment** have landed (commits
`30a16b3`…`5473c8d`). This is the "thin client" layer the headline path put *last* — it now exists,
ahead of the server-side session-options work it was meant to host. None of the actual
session-options behavior (voice override / CEFR / demeanor) or voice cloning is implemented yet.

**Landed — client (Flutter):**
- Redesigned **persona-picker home screen** (`client/lib/screens/home_screen.dart`): a shelf of
  persona cards (avatar, name, `description`, voice blurb, default / last-used tags, connecting
  spinner); last-used persona floats to the top; connection chip + pull-to-refresh.
- **Settings screen** split out (`client/lib/screens/settings_screen.dart`): server URL, API token,
  display name, test-connection, default mic mode, theme.
- **`AppPreferences`** model (`client/lib/models/app_preferences.dart`): default mic mode, theme,
  last-called persona — persisted via `shared_preferences`. Owned at the app root
  (`client/lib/main.dart`); light/dark theming.
- `Persona` model gains `description` + `voice` (`client/lib/models/persona.dart`); call screen
  applies the preferred mic mode before connecting. New tests: `app_preferences_test`,
  `persona_test`, updated `token_client_test`.

**Landed — server (small slice of Feature A/B groundwork):**
- `Persona.description` field (`src/personavoice/models.py`); all four `config/personas/*.yaml`
  carry a one-line `description`.
- `GET /personas` now returns `{id, name, description, voice}` (was `{id, name}`); the voice blurb
  is resolved through the new `VoiceRegistry.describe(ref)` (`src/personavoice/voice/registry.py`),
  and `TokenService` takes an optional `VoiceRegistry` (`src/personavoice/server/token_server.py`).
  Token-server tests updated.

**What this changes about the plan:**
- The headline trio's **client UI (§15 step 4 / Feature J)** is now partly built — the picker and
  settings surfaces exist. The *remaining* client work is the **voice / CEFR / demeanor selectors**
  on top of this scaffold, plus sending the chosen options in the `/token` body.
- `GET /personas` is **no longer "unchanged"** (see §2.5 threading table) — it's already the richer
  shape. CEFR / demeanor live in the `/token` request, not in `/personas`.
- `VoiceRegistry.describe()` exists; the catalog work in **Feature B (§3.2)** still needs
  `VoiceOption` / `catalog()` / `resolve_choice()`.

Still entirely unimplemented: SessionOptions data model + directives (A), voice override plumbing
(A/B), voice catalog + clone enrollment (B), and Features C–L.

---

## 0. Guiding principles (unchanged from the project's design)

- **Cascade, not one model.** New behavior is added at the stage that owns it (prompt vs. voice
  resolution vs. TTS), never by coupling stages.
- **Backends behind an interface.** Anything new must keep the `BACKEND=mac|cuda` switch honest
  and degrade gracefully where a backend lacks a capability (e.g. clones on a preset-only TTS).
- **Thin client, fat server.** New user controls ride the existing token-metadata / data-message
  rail; the client only *chooses*, the server *applies*.
- **Testable offline.** Every server-side change ships with `pytest` coverage that runs without
  GPUs or model weights, following the existing fake-adapter patterns.

---

## 1. Priority & sequencing

| # | Feature | Theme | Effort | Risk | Depends on |
|---|---------|-------|--------|------|------------|
| A | **Session options** (voice / CEFR / demeanor) | Personalization | M | Low | — |
| B | **Multi-voice cloning + voice library** | Personalization | L | Med | A (voice field) |
| C | **Safety / moderation layer** | Trust | M | Low | — (enables "rude") |
| D | **Tool / function calling** | Capability | L | Med | — |
| E | **Post-session feedback report** | Capability | M | Low | memory/transcript |
| F | **Dynamic emotion / prosody** | Naturalness | M | Med | A (emotion plumbing) |
| G | **Semantic endpointing** | Naturalness | M | Med | — |
| H | **Prometheus `/metrics`** | Ops | S | Low | obs/metrics |
| I | **Concurrency / admission control** | Ops | M | Med | — |
| J | **Web client** | Reach | L | Low | token server |
| K | **Persona authoring helper** | DX | S | Low | persona loader |
| L | **Memory introspection (client)** | Trust | S | Low | memory facade |

**Suggested order:** C (small, unblocks "rude" safely) → A → B → then pick from D/E/F by product
priority. H is a quick win any time. The personalization trio (A+B) is the headline; everything
else is independent and can land in any order.

---

## 2. Feature A — Session options (voice / CEFR / demeanor)

### 2.1 Motivation

Three user-facing knobs, all the **same shape**: a *session-level override* layered on top of the
chosen persona, decided **before** the call and (free bonus) swappable mid-call. They reuse the
persona-selection rail already in place: token metadata → agent → optional `data_received`
message.

### 2.2 Data model — `src/personavoice/models.py`

```python
class CEFRLevel(StrEnum):
    a1 = "A1"; a2 = "A2"; b1 = "B1"; b2 = "B2"; c1 = "C1"; c2 = "C2"

class Demeanor(StrEnum):
    kind = "kind"; natural = "natural"; rude = "rude"

class SessionOptions(BaseModel):
    """Per-conversation overrides on top of a persona. All optional; None = persona default."""
    model_config = ConfigDict(extra="ignore")
    voice: str | None = None        # voice-library id (preset | clone | finetuned)
    cefr: CEFRLevel | None = None
    demeanor: Demeanor | None = None

    @classmethod
    def from_metadata(cls, meta: str | None) -> "SessionOptions": ...
    def merged_over(self, base: "SessionOptions") -> "SessionOptions": ...  # precedence
```

`from_metadata` parses the JSON the client puts in room/job metadata
(`{"persona": "...", "voice": "...", "cefr": "B1", "demeanor": "kind"}`), tolerating missing keys
and unknown values (drop unknowns, log once). This sits next to the existing
`persona_id_from_metadata` / `user_id_from_metadata` helpers in
`src/personavoice/orchestrator/agent.py`.

### 2.3 CEFR + demeanor — `src/personavoice/persona/prompt.py`

Extend the existing directive mechanism (the file already does this for `turn_style`):

```python
_DEMEANOR_DIRECTIVE = {
    Demeanor.kind: "Be especially warm, patient, and encouraging. Soften corrections, "
                   "praise effort, and never make the user feel rushed.",
    Demeanor.natural: None,  # baseline: persona as authored
    Demeanor.rude: "Adopt a blunt, curt, impatient tone — terse replies, little hand-holding. "
                   "Stay within bounds: never use slurs, harassment, threats, or demeaning "
                   "personal attacks; brusque, not abusive.",
}

_CEFR_DIRECTIVE = {
    CEFRLevel.a1: "The user is a beginner (CEFR A1). Use very short, simple sentences and the "
                  "most common words. Speak slowly and rephrase if needed.",
    # A2 / B1 / B2 / C1 ...
    CEFRLevel.c2: "The user is near-native (CEFR C2). Use rich, idiomatic, fully natural "
                  "language with no simplification.",
}

def render_system_prompt(persona: Persona, options: SessionOptions | None = None) -> str:
    parts = [persona.system_prompt.strip()]
    # ... existing turn_style directive ...
    if options and options.demeanor and (d := _DEMEANOR_DIRECTIVE.get(options.demeanor)):
        parts.append(d)
    if options and options.cefr and (c := _CEFR_DIRECTIVE.get(options.cefr)):
        parts.append(c)
    parts.append("You are speaking out loud ...")   # keep the spoken-language nudge LAST
    return "\n\n".join(parts)
```

`build_messages(persona, ..., options=None)` forwards `options` to `render_system_prompt`.

> **Safety note:** the `rude` directive is deliberately bounded. It is the natural first consumer
> of **Feature C** (moderation): even with a bounded prompt, the output guard is what *guarantees*
> "brusque, not abusive."

### 2.4 Voice override — `src/personavoice/orchestrator/pipeline.py`

`voice_ref_for` currently resolves from `persona.voice.ref` via `resolve_for_persona`. Add an
explicit-choice path that ignores persona assignment:

```python
def voice_ref_for(persona, backend, voices=None, *, voice_choice: str | None = None) -> VoiceRef:
    if voices is not None and voice_choice:
        ref = voices.resolve_choice(
            voice_choice, backend.tts.name,
            supports_cloning=getattr(backend.tts, "supports_cloning", False),
            default_emotion=persona.voice.emotion,
        )
        if ref is not None:
            return ref
        logger.warning("voice %r not available on %s; using persona default",
                       voice_choice, backend.tts.name)
    # ... existing resolve_for_persona / passthrough ...
```

`resolve_choice` is new in the voice registry (see **Feature B §3.3**).

### 2.5 Threading (precise list)

| File | Change |
|------|--------|
| `models.py` | `CEFRLevel`, `Demeanor`, `SessionOptions` + helpers |
| `persona/prompt.py` | `render_system_prompt(persona, options)`, `build_messages(..., options)` |
| `orchestrator/pipeline.py` | `Pipeline.options` field; `voice_ref_for(..., voice_choice)`; pass `options` to `build_messages` |
| `orchestrator/streaming.py` | `StreamingPipeline.options` field; same wiring in `stream_response` |
| `orchestrator/agent.py` | `PersonaAgent.options` + `set_options()`; `entrypoint` parses `SessionOptions` from metadata; extend `_on_data` to accept option changes (a `session_from_metadata` next to `persona_id_from_metadata`) |
| `server/token_server.py` | `issue(..., voice, cefr, demeanor)` validates + embeds in metadata JSON. `personas()` already enriched (`description` + `voice` blurb, see Status); leave as-is |
| `orchestrator/stream_demo.py`, `demo.py` | `--voice` / `--cefr` / `--demeanor` flags so it is testable offline |

### 2.6 Tests

- Prompt directives present/absent for each CEFR level and demeanor; `natural` is a no-op;
  spoken-language nudge stays last.
- `SessionOptions.from_metadata` for valid / partial / garbage / unknown-enum metadata.
- `voice_ref_for(voice_choice=...)` precedence and graceful fallback when unavailable.
- Token `issue` rejects an unknown voice/cefr/demeanor and echoes valid ones into metadata.

### 2.7 Open decisions

- **Settings persistence.** Recommend: client stores CEFR + demeanor locally and sends them in the
  `/token` body each session (stateless, no consent dependency). Optional upgrade: persist them in
  the per-user **memory profile** when the user has opted in, so they roam across devices.
- **CEFR scope.** Available on all personas but only injected when set; surfaced prominently for
  `language_teacher`. No per-persona allow-list needed.

---

## 3. Feature B — Multi-voice cloning + voice library

> This is the substrate for "pick a voice before a conversation." The store already holds **many**
> named clones (`ClonesStore._voices: dict[name -> ClonedVoice]` in
> `src/personavoice/voice/clone.py`). The gaps are: (1) building that library **from the client**
> (cloning is CLI-only today), (2) **listing** the full voice catalog, and (3) **selecting any
> library voice per session** rather than one clone assigned per persona.

### 3.1 What exists vs. what's missing

| Capability | Today | Needed |
|------------|-------|--------|
| Store many clones | ✅ `ClonesStore` (clones.json) | — |
| Many fine-tuned voices | ✅ `FinetunedVoicesStore` | — |
| Clone from CLI | ✅ `personavoice-clone` | — |
| **Clone from client** | ❌ | **Enrollment endpoint** |
| **List full catalog** | ❌ (only `--list` CLI / `--check`) | **`GET /voices`** |
| **Resolve a chosen voice (not persona-assigned)** | ❌ (`resolve_for_persona` only) | **`resolve_choice`** |
| Backend-capability gating in the picker | partial (`has_preset`) | **`catalog()` with `available`** |

### 3.2 Voice catalog — `src/personavoice/voice/registry.py`

```python
class VoiceOption(BaseModel):
    id: str
    name: str
    kind: str            # "preset" | "clone" | "finetuned"
    emotion: str | None = None
    available: bool      # speakable on the active TTS backend?
    reason: str | None = None   # e.g. "clone requires a cloning backend (f5_mlx/chatterbox)"

class VoiceRegistry:
    def catalog(self, tts_name: str, *, supports_cloning: bool) -> list[VoiceOption]:
        """Unified, selectable list: presets (available iff has_preset), clones + finetuned
        (available iff supports_cloning). Stable order: finetuned, clones, presets."""

    def resolve_choice(self, voice_id, tts_name, *, supports_cloning, default_emotion=None
                       ) -> VoiceRef | None:
        """Resolve an explicitly chosen voice id, independent of any persona assignment:
        finetuned store → clones store → preset (resolve). Returns None when the choice is
        unknown or not speakable on this backend (caller falls back to persona default)."""
```

This reuses `ClonesStore.voice_ref(name, tts_name, emotion=...)` and
`FinetunedVoicesStore.voice_ref(...)`, which already build the right `VoiceRef`. The registry also
already has `describe(ref)` (landed for the `/personas` voice blurb, see Status) — `VoiceOption` can
reuse it for the human-readable name in the catalog.

### 3.3 Clone enrollment — server side (`src/personavoice/server/`)

New routes on the token/HTTP server (auth-gated by the existing `PERSONAVOICE_API_TOKEN`):

| Route | Body / params | Action |
|-------|---------------|--------|
| `GET /voices` | `?backend=` (optional) | return `catalog(...)` for the active backend |
| `POST /voices/clone` | multipart: `audio` (wav ~10 s) + `name` + optional `text` | `validate_sample` → store under clones dir → if a reference-text backend (F5), auto-transcribe `ref_text` via STT → `ClonesStore.record` + `save` → return the new `VoiceOption` |
| `DELETE /voices/clone/{name}` | — | remove from store + save |

Notes / constraints:
- **Validation:** reuse `validate_sample` (duration / format checks already in `clone.py`); cap
  upload size and clamp the number of clones per deployment (abuse surface).
- **Reference text:** F5 needs a transcript of the sample; the server already has an STT in the
  active backend — transcribe on enroll and store as `ClonedVoice.ref_text` (mirrors what the
  cascade does on the CLI path).
- **Persistence:** clones land in `PERSONAVOICE_CLONES_DIR` with the `clones.json` manifest, so the
  live agent picks them up at startup (already true) and across restarts.

### 3.4 Capability truth to surface

Clones/fine-tunes are only **audible on a cloning backend** (`f5_mlx` on Mac, `chatterbox` on
CUDA). On Kokoro/Orpheus the catalog still lists them but with `available=false` + `reason`. The
product implication: a "bring your own voice" session should run a cloning backend. Document this
in the README voice section and have `--check` already-style warnings extend to "N clones present
but active TTS cannot speak them."

### 3.5 Threading & tests

- `voice/registry.py`: `VoiceOption`, `catalog`, `resolve_choice`.
- `server/token_server.py`: three routes; multipart parsing (stdlib `cgi`/manual — keep
  zero-dependency per the existing token server style); auth via existing `check_auth`.
- Tests: `catalog` availability flags per backend; `resolve_choice` precedence + None paths;
  enrollment happy-path with a fake STT + fake sample; rejection of bad samples / oversize /
  over-quota; `DELETE` round-trip.

### 3.6 Open decisions

- **Enrollment auth & quotas.** Who may add a clone, and how many? Recommend: require auth, cap
  count + sample size, and (if multi-user) namespace clones per user id.
- **Consent / likeness.** Cloning a real person's voice is sensitive — add an explicit
  "I'm authorized to use this voice" acknowledgement on the client enroll flow, and keep the
  existing licensing caveats (F5 = CC-BY-NC; Chatterbox = MIT) visible.

---

## 4. Feature C — Safety / moderation layer

### 4.1 Motivation

No input/output moderation exists today (`grep` for `moderat|guardrail|safety` finds only an
incidental comment). A user-facing companion — and especially the new **`rude`** demeanor — needs
a guard. This is a prerequisite for any real-user exposure.

### 4.2 Design

A thin guard wrapping the LLM turn in `orchestrator/streaming.py` (and `pipeline.py`):

- **Input moderation:** classify the user transcript; on a flagged category, branch to a safe
  response instead of the normal turn.
- **Output moderation:** check the assembled reply before TTS; if it breaches bounds (the `rude`
  cap, self-harm, harassment), replace/soften.
- **Crisis path:** a dedicated branch for self-harm / acute distress that returns a calm,
  resource-pointing reply and (optionally) disables `rude` for the rest of the session.

Implementation options: a small local classifier, prompt-based self-check via the existing LLM, or
a hosted moderation call. Keep it behind an interface (`Moderator` protocol) with a no-op default,
mirroring how adapters are pluggable.

### 4.3 Files / tests

- New `src/personavoice/safety/` (protocol + a default rule/keyword guard + optional LLM guard).
- Hook points in `streaming.py` / `pipeline.py` around `stream_chat` / `chat`.
- Tests: flagged input short-circuits; bounded output for `rude`; crisis branch fires; no-op guard
  leaves behavior unchanged.

---

## 5. Feature D — Tool / function calling

### 5.1 Motivation

The brain is a closed conversationalist (`grep` for `tool_call|tools=` is empty). Tools let the
teacher pull a dictionary/conjugation, the interviewer pull a question bank or the candidate's
resume, the companion check time/weather.

### 5.2 Design

- Extend the LLM adapter protocol (`adapters/llm/base.py`, the OpenAI-compat path in
  `_openai_compat.py`, and `vllm.py`) to pass tool schemas and surface `tool_calls`.
- A tool-execution loop in the orchestrator turn: model → tool call → execute → feed result →
  continue, with a max-iteration cap and a timeout (latency-sensitive — keep tools fast or run
  them speculatively).
- A small registry of safe, side-effect-free tools first (dictionary, time, retrieval over a
  per-persona knowledge file); gate any side-effecting tool.
- Streaming caveat: defer TTS for a turn that ends in a tool call until the follow-up text streams.

### 5.3 Files / tests

- `adapters/llm/*`, `orchestrator/streaming.py` + a new `orchestrator/tools.py` registry.
- Tests with a fake LLM that emits a tool call then a final answer; verify the loop, the cap, and
  that a no-tool persona is unaffected.

---

## 6. Feature E — Post-session feedback report

### 6.1 Motivation

Eval scores the *system*; nothing scores the *user*. For `pm_interviewer` / `hr_interviewer` /
`language_teacher` the obvious product feature is an end-of-call report.

### 6.2 Design

- A distill-style pass (reuse the `memory/distill.py` machinery) over the session transcript that
  emits a structured rubric: for interviews — STAR structure, filler-word count, pace, clarity;
  for the teacher — CEFR-calibrated grammar/vocab feedback and corrections.
- Triggered on session end (participant leaves in `agent.py`) or on demand; returned to the client
  as a data message / fetched via a new `GET /session/{id}/report` route.
- Persisted with the session (consent-gated, same store as memory).

### 6.3 Files / tests

- `src/personavoice/eval/report.py` (rubric prompts + parsing), hook in `agent.entrypoint`
  teardown, route in `token_server.py`.
- Tests with a canned transcript + fake LLM producing a rubric JSON; schema validation.

---

## 7. Feature F — Dynamic emotion / prosody

### 7.1 Motivation

Emotion is **static** today — resolved once from the voice registry (`registry.py` `resolve`,
`VoiceSettings.emotion` default `"neutral"`). Orpheus has emotion tags and Chatterbox has
exaggeration control that go unused per-utterance.

### 7.2 Design

- Let the LLM emit a lightweight per-reply emotion hint (a leading tag stripped before TTS, or a
  structured side-channel), or infer it cheaply from the reply.
- Thread it into `VoiceRef.emotion` at synth time (the field already exists) so
  `tts/orpheus.py` / `tts/chatterbox.py` render it; Kokoro ignores it gracefully.

### 7.3 Files / tests

- `orchestrator/streaming.py` (extract hint), `voice_ref_for` (apply per-utterance emotion),
  TTS adapters (map hint → backend control).
- Tests: hint extraction + stripping; emotion reaches the synth call; preset-only backend ignores
  it.

---

## 8. Feature G — Semantic endpointing

### 8.1 Motivation

Endpointing is pure Silero VAD silence (`orchestrator/endpointing.py`). It cannot tell a
mid-thought pause ("I think… *pause* …it's fine") from a finished turn — a source of both dead air
and mid-thought barge-ins.

### 8.2 Design

- A cheap "is this utterance complete?" gate (punctuation/heuristic or a tiny classifier) layered
  on the VAD timeout in `agent.py`'s turn detection; extend the turn just past a VAD trigger when
  completion confidence is low.
- Optional: backchannels ("mhm") and telling the LLM when it was cut off mid-sentence so it can
  recover gracefully.

### 8.3 Files / tests

- New `orchestrator/completion.py` (pure, offline-testable like `endpointing.py`), wired in the
  VAD handling in `agent.py`.
- Tests over transcript fragments: complete vs. trailing-conjunction vs. filler.

---

## 9. Feature H — Prometheus `/metrics` (quick win)

Per-turn metrics are only **logged** (`obs/metrics.py` `TurnMetrics.log`). Add a Prometheus
exporter (counters/histograms for STT / first-token / first-audio / total, barge-ins, errors) and
a `/metrics` route on the token/HTTP server alongside `/healthz`. Optional dependency, no-op when
the client lib is absent. Small, isolated, high ops value.

---

## 10. Feature I — Concurrency / admission control

The VRAM budget assumes **one** conversation. Nothing caps simultaneous callers on a 16 GB card.
Add: a max-concurrent-sessions gate in the agent worker (reject/queue with a "busy" reply when
full), a documented capacity number per backend, and a `/healthz` field exposing current load.
Pairs with Feature H for visibility.

---

## 11. Feature J — Web client

Only the Flutter mobile app exists (plus the generic Agents Playground). The Flutter app now has a
real **persona picker + Settings surface** (see Status) — the natural place to add the voice / CEFR /
demeanor selectors before standing up a separate web client. A purpose-built browser client
(LiveKit JS SDK) is still worthwhile to remove the install barrier and is the natural home for the
**voice picker** (Feature B) and the **session report** (Feature E); it reuses the token server and
`/personas` + `/voices` routes. Treat the web client as additive, not a prerequisite for the
session-options UI.

---

## 12. Feature K — Persona authoring helper

Personas are hand-written YAML validated by `--check`. Add a generator that turns a plain-English
description into a valid persona file (prompt + voice + behavior knobs) and validates it against
the `Persona` model + voice registry before writing. Small DX win that lowers the barrier to new
personas.

---

## 13. Feature L — Memory introspection (client)

A "what do you remember about me?" surface over the existing per-user store
(`memory/profile.py`, `memory/facade`), exposed as a data message / route and shown in the client.
Strengthens the consent story already built (`personavoice-memory --show/--export/--delete`).

---

## 14. Cross-cutting checklist (applies to every server-side item)

- `pytest` green offline (fake adapters, no GPU); `ruff check` + `ruff format --check`; `mypy src`.
- `.env.example` updated for any new env var; README section updated.
- `python -m personavoice.server --check` extended where a new capability can be misconfigured
  (e.g. clones present but active TTS can't speak them; auth required for `/voices/clone`).
- Backward compatible: every new field is optional and defaults to today's behavior.

---

## 15. Headline path (the requested trio, end to end)

1. **C** — land the moderation seam (no-op default) so `rude` is safe by construction.
2. **A** — `SessionOptions` + CEFR/demeanor directives + voice-override plumbing + token-server
   validation + offline flags on the demos. Fully testable without the client.
3. **B** — voice **catalog** + `resolve_choice` + the **clone enrollment** endpoints, so the user
   has a real library to pick from. Document the cloning-backend requirement.
4. **Flutter (scaffold landed) / Web** — the persona picker + Settings surface already exist
   (see Status). Remaining: add **voice / CEFR / demeanor selectors** on top, wire them into the `/token`
   body, and consume `/voices` once Feature B lands. A separate web client is optional/additive.

Steps 1–3 are server-only and unit-testable; the client UI is the last, thinnest layer — and its
scaffold (picker + settings + preferences) is already in place.

# Persona-Voice — Implementation Plan (Next Features)

A consolidated, detailed plan for the next round of functionality. It merges two sets of
ideas:

1. **Capability / production gaps** surfaced from a code read (tool calling, safety, dynamic
   emotion, semantic endpointing, metrics, concurrency, web client, persona authoring, memory
   introspection).
2. **Session-personalization features** requested directly: **pick a voice before a
   conversation** (built on **multi-voice cloning**), **CEFR level** for language learners, and
   **person type / demeanor** (kind / natural / rude).

Most of this is still spec, not code. File paths and function names refer to the current tree so
each item is grounded and actionable. See the **Status** section below for the client/server
scaffold that has since landed and what it changes about the remaining work.

**Convention** Mark a functionality as `[Done]` when it's completed, or `[Partial]` when it's started
but not finished. Do not refer code and the implementation plan with Feature B, Feature C, Feature D, etc.
Because this implementation plan will be removed, and we'll have dead references.

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
  account id, display name, test-connection, default mic mode, theme.
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
- The headline trio's **client UI (§14 step 4 / Feature J)** is now partly built — the picker and
  settings surfaces exist. The *remaining* client work is the **voice / CEFR / demeanor selectors**
  on top of this scaffold, plus sending the chosen options in the `/token` body.
- `GET /personas` is **no longer "unchanged"** (see §2.5 threading table) — it's already the richer
  shape. CEFR / demeanor live in the `/token` request, not in `/personas`.
- `VoiceRegistry.describe()` exists; the catalog work in **Feature B (§3.2)** still needs
  `VoiceOption` / `catalog()` / `resolve_choice()`.

**Update (server-side headline trio landed).** Features **C → A → B are implemented server-side
and unit-tested** (full `pytest` green, `ruff`/`mypy` clean):
- **C** — `safety/` moderation seam: `Moderator` protocol + no-op default + `KeywordModerator`
  (crisis/abuse), wired into both pipelines; `PERSONAVOICE_MODERATION` toggle.
- **A** — `SessionOptions`/`CEFRLevel`/`Demeanor`, CEFR+demeanor prompt directives, voice-override
  resolution, options threaded through pipelines + agent (incl. mid-call `set_options` /
  `resolve_session_options`), token-server validation/echo, and `--voice/--cefr/--demeanor` demo
  flags.
- **B** — `VoiceOption` + `catalog()`/`resolve_choice()`/`choice_ids()`, and the
  `GET /voices` / `POST /voices/clone` / `DELETE /voices/clone/{name}` routes (raw-wav upload,
  quotas, capability gating).

Still unimplemented: the **client selectors/picker** (§14 step 4 / Feature J) on top of these
routes, and Features **D–L**.

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

> **NOW (do these first).** The requested client features. Three are *client-only* — the server
> work they ride on (Features **A/B/C**) is already done and unit-tested; one (**N3**) adds the
> only new server surface. Detailed specs are in the **NOW** section directly below this table.

| # | Feature | Builds on | Server work | Status |
|---|---------|-----------|-------------|--------|
| **N1** | **Per-persona session selectors** — voice / CEFR / demeanor, chosen per persona, applied *before* a call | A `[Done]` | none | `[Done]` (client; device run pending) |
| **N2** | **Voice Library page** — list voices + clone by file upload or device mic | B `[Done]` | none | `[Done]` (client; device run pending) |
| **N3** | **Custom personas** — multi-user **JSON store**, create / edit / delete + LoRA pick | K | **new routes** | `[Done]` (server + agent + client form done & tested; device run pending) |
| **N4** | **Seed voices** — two ready voices out of the box (`assets/seed_voices/`) | B `[Done]` | seed script | `[Done]` (script + bundled `Feminine`/`Masculine` wavs) |

**Order: N1 → N2 → N4 → N3.** N1+N2 are built (Flutter client; `flutter analyze` clean, unit tests
green — no on-device run yet). **N4 done** (seed script + bundled Feminine/Masculine voices). **N3 done**:
server (`UserPersonaStore`, `POST/PUT/DELETE/GET /personas`, full-body `GET /personas/{id}`,
`GET /loras`, `user`-scoped `/token`, agent resolution, persona `session_defaults`) **and** the
client New/Edit-persona form — `flutter analyze` clean, 86 client + 633 server tests green; live
device + LiveKit call run still pending.

**Backlog (capabilities / ops / DX — independent, land any time).** Features **A/B/C** are the
*server* substrate the NOW block builds on (already done & tested); **D** (tool calling),
**F** (dynamic emotion), and **G** (semantic endpointing) have since landed; **L** (memory
introspection) is now done; H–K are unchanged.

| # | Feature | Theme | Effort | Risk | Depends on |
|---|---------|-------|--------|------|------------|
| A | **Session options** (voice / CEFR / demeanor) `[Done]` (server) | Personalization | M | Low | — |
| B | **Multi-voice cloning + voice library** `[Done]` (server) | Personalization | L | Med | A (voice field) |
| C | **Safety / moderation layer** `[Done]` | Trust | M | Low | — (enables "rude") |
| D | **Tool / function calling** `[Done]` | Capability | L | Med | — |
| F | **Dynamic emotion / prosody** `[Done]` | Naturalness | M | Med | A (emotion plumbing) |
| G | **Semantic endpointing** `[Done]` | Naturalness | M | Med | — |
| H | **Prometheus `/metrics`** `[Done]` | Ops | S | Low | obs/metrics |
| I | **Concurrency / admission control** `[Partial]` (steps 1–5 done; server-FIFO queue optional) | Ops | M | Med | H (visibility, soft) |
| J | **Web client** | Reach | L | Low | token server |
| K | **Persona authoring helper** `[Done]` (add-on to **N3**) | DX | S | Low | persona loader |
| L | **Memory introspection (client)** `[Done]` | Trust | S | Low | memory facade |

---

## 1.5 NOW — requested client features (implement first)

These four turn the done server work into user-facing controls. Locked decisions: custom
personas live in a **JSON store**; the deployment is **multi-user** (clones *and* personas are
namespaced per `user_id`, taken from the `user` token-metadata key — the participant identity is
only a fallback when it's absent); **LoRA stays vLLM-only**
(see the backend note in **N3**).

### N1 — Per-persona session selectors (client) — *client-only* `[Done]`

> **Built.** `models/session_options.dart` (+ `voice_option.dart`), `TokenClient.requestToken`
> sends voice/cefr/demeanor, a "Customize" sheet per persona card
> (`screens/persona_options_sheet.dart`), overrides persisted in `AppPreferences.personaOptions`,
> and `VoiceSession` re-sends them in the persona data message (mid-call). `flutter analyze` clean,
> unit tests green; on-device run still pending.

**Server:** done (Feature A). `/token` accepts + validates `voice/cefr/demeanor`
(`server/token_server.py`), the agent overlays them on the persona, and they're swappable
mid-call via the data message. Nothing new server-side.

**Client work (`client/lib`):**
- `TokenClient.requestToken(persona, {voice, cefr, demeanor, room})` — add the three optional
  fields to the POST body (today persona-only, `services/token_client.dart:71`).
- A small `SessionOptions` Dart model + `fetchVoices()` (`GET /voices`) on `TokenClient`.
- A **"Customize" sheet per persona card** (`screens/home_screen.dart`): voice picker (from the
  catalog — grey out `available=false` and show `reason`), CEFR dropdown (Default / A1–C2),
  demeanor (Default / kind / natural / rude). **Default = unset = the persona as authored** (the
  4 curated personas are unchanged unless the user opts in).
- Persist last-used overrides **per persona** in `AppPreferences` (`shared_preferences`).
- `VoiceSession._sendPersona` (`services/voice_session.dart:289`) also publishes the chosen
  options in the data message, so mid-call persona switches keep them.

**Tests:** token_client encodes options; `SessionOptions` defaults; preferences round-trip.
**Caveat:** a chosen voice is only *audible* on a cloning backend — the catalog `available` flag
carries this through to the picker.

### N2 — Voice Library page (client) — *client-only* `[Done]`

> **Built.** `screens/voice_library_screen.dart` lists `GET /voices` grouped by kind with
> availability/reason badges; enroll via file upload (`file_picker`) or device mic (`record` +
> `path_provider`, `services/voice_recorder.dart`) behind the "authorized" consent box; delete
> clones. New client deps (file_picker / record / path_provider) — **next iOS build needs
> `pod install`**. Entry points: home AppBar + Settings. Analyzer clean, unit tests green; on-device
> run pending.

**Server:** done (Feature B). `GET /voices`, `POST /voices/clone?name=&text=&authorized=` (raw
wav as the body), `DELETE /voices/clone/{name}`. The shared WAV codec mixes to mono + resamples
(`audio.py:50`), so a device-mic recording or an uploaded wav needs **no client resampling**;
`validate_sample` accepts **2–60 s**.

**Client work:**
- New `VoiceLibraryScreen`: list `GET /voices` grouped by kind (finetuned / clones / presets),
  with availability + `reason` badges.
- **Upload:** `file_picker` → wav bytes → `TokenClient.cloneVoice(name, bytes, {text,
  authorized})` POSTing the raw body.
- **Mic:** record ~10 s (reuse the existing audio-session / record plumbing), encode wav, same
  POST. **Required consent checkbox → `authorized=true`** (plan §3.6 / IMPLEMENTATION_PLAN ask).
- **Delete:** `DELETE /voices/clone/{name}` behind a confirm.
- Reachable from Settings and from the N1 voice picker ("+ Add a voice").

**Tests:** `cloneVoice` builds the right request (raw body + query params); list parses
kinds/availability; delete round-trip (fake `http.Client`).
**Multi-user:** list/clone/delete are scoped to the caller's `user_id` (see N3).

### N3 — Custom personas: multi-user JSON store (server + client) — *supersedes Feature K* `[Done]`

> **Done — server + agent + client.** `persona/store.py` (`UserPersonaStore`, per-user JSON at
> `PERSONAVOICE_USER_PERSONAS`), `persona/lora.py` (`served_loras` / `LoraOption`), token-server
> routes `GET /personas?user=` (curated + custom, `custom` flag, curated win on clash),
> `POST /personas?user=`, full-body `GET /personas/{id}?user=` (prefills the edit form — the list
> returns only a summary), `PUT`/`DELETE /personas/{id}?user=`, and `GET /loras`. `/token` now
> takes a `user` and embeds it in the metadata so a custom persona **resolves at call time** (and
> accepts an owned custom-persona id, which isn't in the curated registry) — without it a custom
> persona could be authored but not called. Drafts validate through the `Persona` model (extra
> fields rejected; voice ref + base model default to the curated default; slug ids never shadow
> curated; per-user quota `PERSONAVOICE_MAX_USER_PERSONAS`; a set `llm.lora` is checked against the
> served adapters). The agent resolves a custom persona by `user_id` (curated-first), hot-reloads
> it for a mid-call switch, and applies a persona's `session_defaults` under any explicit session
> option. **Client:** `PersonaDraft` model (round-trips advanced fields it doesn't surface),
> `TokenClient` CRUD + `/loras`, a `PersonaFormScreen` (name, system prompt, base voice, turn
> style, memory, session-default CEFR/demeanor, LoRA dropdown — empty on Mac with the reason), and
> a home-shelf "New persona" FAB + per-card edit/delete on custom personas. The scoping user is
> the normalized **account id** (`ConnectionSettings.userId`), or `"default"` when unset
> (`ConnectionSettings.effectiveUser`), used for both the CRUD routes and the `/token` body so
> authoring and calling hit the same bucket. The `/token` body also carries a stable, opaque
> participant `identity` (`sub`) and a cosmetic display `name`, both independent of the account id
> — see the field split in `client/lib/models/connection_settings.dart` / `server/tokens.py`.

This was the only item that needed **new server endpoints**.

**Server (`src/personavoice/persona/` + `server/token_server.py`):**
- A `UserPersonaStore` persisting to a writable JSON file (`PERSONAVOICE_USER_PERSONAS` →
  `user_personas.json`), shape `{user_id: {persona_id: Persona}}`. Mirror `ClonesStore`'s
  load / record / remove / save pattern (`voice/clone.py`).
- Routes (auth-gated via existing `check_auth`, scoped to the caller's `user_id`):
  - `GET /personas` — merge **curated** (`config/personas/*.yaml`, read-only) + this user's
    stored personas. Curated win on id clash and are non-deletable.
  - `POST /personas` — body = a persona draft (name, system_prompt, voice ref + emotion,
    turn_style, memory toggle, default `cefr/demeanor/voice`, optional `llm.lora`). Validate with
    the `Persona` model (already `extra="forbid"`), assign a namespaced id, `record` + `save`,
    return the `Persona`.
  - `PUT /personas/{id}` / `DELETE /personas/{id}` — the user's own personas only; never curated.
- Resolution: the agent's persona lookup consults the user store (by `user_id`) **before** the
  curated `PersonaRegistry`; keep `reload()` semantics (`persona/registry.py:19`).

**LoRA selection (vLLM / CUDA only):**
- `GET /loras` → the served adapter names the active backend exposes (vLLM `--lora-modules`),
  each with `available` + `reason`, like the voice catalog. On Mac / LM Studio
  (`supports_lora=False`) the list is empty / "merged at train time."
- A persona's `llm.lora` already routes by basename on vLLM (`adapters/llm/_openai_compat.py:33`)
  — no new routing, just the picker + validation that the chosen name is actually served.

> **Backend note — vLLM on Mac.** vLLM has **no Apple-Silicon / Metal GPU backend** (CUDA-first;
> only experimental x86-CPU / ROCm / TPU), so it can't serve a 7B at voice latency on the M4. We
> keep **LM Studio on Mac**. On Mac, "apply a LoRA" = select a **merged** base model (the mlx-lm
> train→merge path from M7); **served / hot-swap LoRA stays the CUDA/4080 path**. The client LoRA
> picker reflects backend capability (empty on Mac), exactly like the voice catalog.

**Client work:** a "New / Edit persona" form (name, system prompt, base voice, turn_style, memory,
default cefr/demeanor/voice, LoRA dropdown from `/loras`); custom personas appear on the home
shelf beside the 4 curated, with edit/delete (curated are read-only).

**Tests:** store CRUD + per-user isolation; `POST` validation rejects bad drafts / extra fields;
curated personas non-shadowable and non-deletable; `/loras` availability per backend.
**Open decisions:** id namespacing (e.g. `u/{user_id}/{slug}`); per-user persona quota; whether
user-authored system prompts get a safety pass (ties to Feature C).

### N4 — Seed voices (two out-of-the-box clones) — *seed script* `[Done]`

> **Done.** `voice/seed.py` (`personavoice-seed-voices` / `scripts/seed_voices.py`) enrolls every
> wav in `assets/seed_voices/` as a clone named after the file stem, through the same
> `VoiceCloner` → `ClonesStore` path a client upload uses — **idempotent** (skip-if-present;
> `--force` re-enrolls). The bundled `Feminine.wav` + `Masculine.wav` ship as presets in
> `config/voices.yaml`, so the seeder skips them as duplicate clone IDs. On a preset-only backend
> a non-preset seed still records (catalog-visible as `available=false`).
> `PERSONAVOICE_SEED_VOICES_DIR` overrides the dir. Tested: idempotency, discovery, both
> per-backend enrollers, bad-sample rejection.

Ship two ready clones so N1/N2 demo with real voices on first run.

- Source: `assets/seed_voices/{Feminine,Masculine}.wav` (44.1 kHz stereo — fine; the codec mixes
  to mono + resamples).
- A `make seed-voices` / `scripts/seed_voices.py` that enrolls them through the **same path** a
  client upload uses (`VoiceCloner` → `ClonesStore.record`), names clones after the wav stem.
  **Idempotent** (skip if present).
- Runs against the **active cloning backend** (chatterbox); on a preset-only backend they
  enroll but list `available=false`. Multi-user: seed under a shared/global namespace visible to
  all users.

**Tests:** seed is idempotent; enrolled clones appear in `GET /voices`; a bad/short sample is
rejected.

---

## 2. Feature A — Session options (voice / CEFR / demeanor) `[Done]` (server)

> **Status:** Server-side complete and unit-tested. `SessionOptions` + `CEFRLevel`/`Demeanor`
> (`models.py`), the CEFR/demeanor directives (`persona/prompt.py`), the voice-override path
> (`voice_ref_for(..., voice_choice=...)`), options threaded through `Pipeline`/`StreamingPipeline`/
> `PersonaAgent` (with `set_options()` + a `session_from_metadata`/`resolve_session_options`
> helper and mid-call data-message changes), token-server validation/echo, and `--voice/--cefr/
> --demeanor` flags on both demos all landed. The remaining piece is the **client selectors**
> (§14 step 4 / Feature J).

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

## 3. Feature B — Multi-voice cloning + voice library `[Done]` (server)

> **Status:** Server substrate complete and unit-tested. `VoiceOption` + `VoiceRegistry.catalog()`
> / `resolve_choice()` / `choice_ids()` (`voice/registry.py`), and the HTTP routes `GET /voices`,
> `POST /voices/clone` (raw-wav body + `name`/`text`/`authorized`; quota + size caps), and
> `DELETE /voices/clone/{name}` (`server/token_server.py`, with `ClonesStore.remove`) all landed.
> Enrollment is multipart-free (Python 3.13 dropped `cgi`): the wav rides as the raw request
> body. The **client picker UI** is the remaining piece (§14 step 4 / Feature J).

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
    reason: str | None = None   # e.g. "clone requires a cloning backend"

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
| `POST /voices/clone` | `audio` (wav ~10 s, raw body) + `name` | `validate_sample` → store under clones dir → `ClonesStore.record` + `save` → return the new `VoiceOption` |
| `DELETE /voices/clone/{name}` | — | remove from store + save |

Notes / constraints:
- **Validation:** reuse `validate_sample` (duration / format checks already in `clone.py`); cap
  upload size and clamp the number of clones per deployment (abuse surface).
- **Persistence:** clones land in `PERSONAVOICE_CLONES_DIR` with the `clones.json` manifest, so the
  live agent picks them up at startup (already true) and across restarts.

### 3.4 Capability truth to surface

Clones/fine-tunes are only **audible on a cloning backend** (`chatterbox` on CUDA). On Kokoro
the catalog still lists them but with `available=false` + `reason`. The
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
  "I'm authorized to use this voice" acknowledgement on the client enroll flow.

---

## 4. Feature C — Safety / moderation layer `[Done]`

> **Status:** The seam landed. `safety/` ships the `Moderator` protocol + `ModerationResult`, a
> `NoopModerator` default (wired everywhere, behavior unchanged) and a dependency-free
> `KeywordModerator` (crisis-on-input short-circuit + bounded `rude`/abuse/threat output),
> selectable via `PERSONAVOICE_MODERATION`. Input moderation short-circuits both pipelines;
> output moderation runs in the turn-based `Pipeline`. **Follow-up:** full pre-TTS output
> moderation on the *streaming* path (buffering the whole reply would defeat streaming) — for now
> the bounded `rude` prompt + the input guard cover streaming.

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

## 5. Feature D — Tool / function calling `[Done]`

> **Status:** Implemented and unit-tested. New `orchestrator/tools.py` (`ToolSpec` + JSON-schema
> contract, `ToolRegistry` with per-persona `select()`, and the bundled side-effect-free
> `get_current_time`); a structural `Tool` protocol on the adapter (`adapters/llm/base.py`) keeps
> the dependency one-way (orchestrator → adapters). The OpenAI-compatible path
> (`_openai_compat.py`, so vLLM **and** LM Studio) runs the streamed model → tool → result loop:
> tool-call SSE fragments are reassembled (`_ToolCallBuffer`), executed (bad name / bad args /
> timeout / handler error all become a safe result string fed back to the model), and the
> follow-up reply streams for TTS — **nothing is voiced during the tool round-trips**. A
> `PERSONAVOICE_TOOL_MAX_ITERS` cap forces a final tool-free answer; `PERSONAVOICE_TOOL_TIMEOUT`
> bounds each call. Both pipelines + the agent + the demos thread a `ToolRegistry`; a persona opts
> in via a new `Persona.tools` list (the `companion` ships `get_current_time`). Backends that can't
> route tools (Ollama, mlx-lm) degrade gracefully via the base no-op. `server --check` lists the
> registry and warns on unknown / unroutable tool references. The persona-knowledge-retrieval tool
> was left for later (needs per-persona knowledge files); only safe offline tools ship.

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

## 6. Feature F — Dynamic emotion / prosody `[Done]`

> **Status:** Implemented and unit-tested, opt-in via `PERSONAVOICE_DYNAMIC_EMOTION` (default off,
> so behavior is unchanged). New pure module `emotion.py` (top-level, next to `audio.py`): a closed
> emotion vocabulary + aliases, the `EMOTION_DIRECTIVE` prompt text, `canonical_emotion`,
> `split_emotion_hint` (full-reply parse for the turn-based `Pipeline`) and `split_leading_emotion`
> (streaming parse that buffers only the leading token window). When on, `render_system_prompt`
> appends the tag directive; both pipelines strip the leading `[emotion]` tag off the reply (before
> TTS, history, transcript, and the output guard) and apply it to that turn's `VoiceRef.emotion`
> via `model_copy`. Chatterbox's `_EMOTION_EXAGGERATION` map was extended to cover the whole
> vocabulary (a test guards the coupling); Kokoro ignores `emotion`, so it degrades to a no-op on a
> preset-only backend. The static per-persona `emotion` baseline is untouched (used when no tag is
> emitted). `.env.example` + README updated. The optional "infer the emotion from the reply"
> alternative was not taken — the LLM-emitted tag is lower-latency and needs no extra inference.

### 6.1 Motivation

Emotion is **static** today — resolved once from the voice registry (`registry.py` `resolve`,
`VoiceSettings.emotion` default `"neutral"`). Chatterbox has
exaggeration control that go unused per-utterance.

### 6.2 Design

- Let the LLM emit a lightweight per-reply emotion hint (a leading tag stripped before TTS, or a
  structured side-channel), or infer it cheaply from the reply.
- Thread it into `VoiceRef.emotion` at synth time (the field already exists) so
  `tts/chatterbox.py` renders it; Kokoro ignores it gracefully.

### 6.3 Files / tests

- `orchestrator/streaming.py` (extract hint), `voice_ref_for` (apply per-utterance emotion),
  TTS adapters (map hint → backend control).
- Tests: hint extraction + stripping; emotion reaches the synth call; preset-only backend ignores
  it.

---

## 7. Feature G — Semantic endpointing `[Done]`

> **Status:** Implemented and unit-tested, opt-in via `PERSONAVOICE_SEMANTIC_ENDPOINTING` (default
> off, so pure-VAD behavior is unchanged). New pure module `orchestrator/completion.py`:
> `assess_completion`/`is_complete` classify a VAD-utterance transcript as a finished turn or a
> mid-thought pause from conservative trailing-word cues (dangling conjunction / preposition /
> determiner, or a hesitation filler), plus `join_fragments` and the
> `PERSONAVOICE_SEMANTIC_ENDPOINTING` / `PERSONAVOICE_ENDPOINTING_GRACE_MS` env helpers. When on,
> `PersonaAgent` (`agent.py`) holds an unfinished utterance and merges the next one onto it instead
> of answering it, with a grace timer (`call_later`, default 1500 ms) that flushes the held text if
> no continuation arrives; a resumed utterance (VAD START_OF_SPEECH) cancels the pending flush so
> the fragments merge. The classifier is deliberately conservative — a false "incomplete" only
> delays the reply by the grace window, while a false "complete" would let the user be barged in
> on — so it holds only on clear continuation cues and judges ambiguous enders ("for", modals,
> object pronouns) complete. The optional backchannel / "you were cut off" hint was not taken
> (kept to the core hold-and-merge gate).

### 7.1 Motivation

Endpointing is pure Silero VAD silence (`orchestrator/endpointing.py`). It cannot tell a
mid-thought pause ("I think… *pause* …it's fine") from a finished turn — a source of both dead air
and mid-thought barge-ins.

### 7.2 Design

- A cheap "is this utterance complete?" gate (punctuation/heuristic or a tiny classifier) layered
  on the VAD timeout in `agent.py`'s turn detection; extend the turn just past a VAD trigger when
  completion confidence is low.
- Optional: backchannels ("mhm") and telling the LLM when it was cut off mid-sentence so it can
  recover gracefully.

### 7.3 Files / tests

- New `orchestrator/completion.py` (pure, offline-testable like `endpointing.py`), wired in the
  VAD handling in `agent.py`.
- Tests over transcript fragments: complete vs. trailing-conjunction vs. filler.

---

## 8. Feature H — Prometheus `/metrics` (quick win) `[Done]`

> **Status:** Implemented, unit-tested, and verified live end-to-end (worker → shared dir →
> token-server `/metrics` → browser). New `obs/prometheus.py` exposes the optional exporter:
> `personavoice_turns_total{persona,outcome}` (outcome = ok|interrupted|error, so it doubles as
> the barge-in/error counters) plus latency histograms
> `personavoice_{stt,first_token,first_audio,turn}_seconds{persona}`. `record_turn()` feeds them
> from the same per-turn `TurnMetrics` the agent already logs (best-effort — never crashes a
> turn); `render_metrics()` backs a `GET /metrics` route on the token server alongside `/healthz`
> (unauthenticated, un-rate-limited). `prometheus-client` is an optional dep (the `metrics` extra,
> baked into the server image) — absent, `/metrics` serves an empty body. The agent worker and the
> token server are separate processes, so they aggregate via the standard
> `PROMETHEUS_MULTIPROC_DIR` shared dir: wired into `docker-compose.livekit.yml` (mount + env) and
> the `run-mac.sh` / `run-cuda.ps1` launchers. `--check` surfaces exporter availability and warns
> on a missing multiproc dir. README + `.env.example` updated.

Per-turn metrics are only **logged** (`obs/metrics.py` `TurnMetrics.log`). Add a Prometheus
exporter (counters/histograms for STT / first-token / first-audio / total, barge-ins, errors) and
a `/metrics` route on the token/HTTP server alongside `/healthz`. Optional dependency, no-op when
the client lib is absent. Small, isolated, high ops value.

---

## 9. Feature I — Concurrency / admission control `[Partial]`

> **Status (steps 3–4):** Step 3 (busy clip) + step 4 (token-server 503 gate) implemented +
> unit-tested offline (`tests/test_admission_control.py`, `tests/test_token_server.py`; full suite
> green, ruff + mypy clean). Step 3: over-cap jobs are no longer bare-rejected — `_request_fnc`
> counts the rejection and *accepts* with a busy marker (`busy_accept_metadata`), and `entrypoint`
> short-circuits to `_serve_busy_clip`, which publishes a track, plays a **pre-rendered** clip
> (`orchestrator/busy.py`: a synthesized stdlib telephone busy tone, or a `PERSONAVOICE_BUSY_CLIP`
> WAV), sends a `{"event":"busy"}` data message, and `ctx.shutdown()`s — **no backend/model load on
> that path**. Step 4: opt-in `PERSONAVOICE_ADMISSION_503` makes `TokenService.issue` return `503 +
> Retry-After` (+ `retry_after` body) once the shared gauge reports `active >= max_sessions()`;
> fails open (mints) when the count is unknown, so it never blocks on its own. `--check` flags a
> set-but-unreadable busy clip and the 503 gate enabled without `PROMETHEUS_MULTIPROC_DIR`;
> `.env.example` + README updated. *Live two-caller run (busy clip audible + 503 on the 4080) still
> pending a GPU/LiveKit server.* Step 5 (queueing) remains deferred.

> **Status:** Steps 1–2 implemented + unit-tested offline, and **verified live on the Mac stack**
> (admission control confirmed in worker logs: `threshold 0.5`, load `1.0`↔`0.0`, "at full
> capacity, marking as unavailable"). The agent worker caps concurrency via a custom `load_fnc`
> (reports `active_jobs / capacity`, so LiveKit stops dispatching once full — the **race-safe OOM
> rail**, since the framework reserves the slot before `request_fnc` runs) plus a `request_fnc`
> reject backstop; `load_threshold` is per capacity (`(cap-0.5)/cap`, always `< 1`, which the
> framework requires). Capacity is `config.max_sessions()` (`PERSONAVOICE_MAX_SESSIONS`, default 1),
> shared by worker/token-server/`--check`. `/healthz` reports `sessions_max` from that config
> (authoritative) and `sessions_active` from the worker's live gauge; `/metrics` carries
> `personavoice_sessions_{active,max}` + `_sessions_rejected_total`. **Gotcha fixed:** LiveKit wipes
> `PROMETHEUS_MULTIPROC_DIR` at worker startup, so the (unlabeled) session metrics are created
> **lazily on first write** — eager creation at import wrote files LiveKit then unlinked, leaving
> the worker writing to a dangling inode (`sessions_*` stuck at 0); regression-tested. `--check`
> prints the cap and warns when > 1. `.env.example`, README, compose env updated. Steps 3 (busy
> clip) + 4 (token-server 503 gate) are now also done (see the steps-3–4 status above); step 5
> (queueing) remains deferred. *Live two-caller OOM check on the 4080 still pending a GPU run.*

The VRAM budget assumes **one** conversation, but nothing caps simultaneous callers — and the
failure is harder than "degraded." On non-Windows, the LiveKit worker dispatches each job into its
**own subprocess**, and the process-global `_WARMED` cache (`orchestrator/agent.py`) only dedupes
models *within* a process. So a second concurrent caller spawns a second process that loads a
second copy of STT+TTS weights → **OOM on a 16 GB card**. This is the safety rail that keeps the
box alive, not just politeness.

> **Trap:** `num_idle_processes=1` (`agent.py`) does *not* cap concurrency. It only caps the warm
> *idle* pool; LiveKit still spawns additional job processes on demand. Today there is zero
> admission control — the first time two people call at once on CUDA, the worker OOMs.

**Capacity is ~1 session per worker** on both backends (CUDA: vLLM batches the LLM in its own
container, but the worker's STT+TTS don't share across job processes; Mac: MLX/Apple audio aren't
built for parallel sessions — serialize). So the scaling story is: **cap each worker at its safe
number (default 1) and scale out by running more workers — LiveKit's dispatcher already
load-balances across them.** Admission control's job is narrow: protect a single worker from
over-commit and emit a graceful "busy" when *all* workers are full. Capacity is configurable via
`PERSONAVOICE_MAX_SESSIONS` (default 1), documented per backend in `.env.example`.

**Reject, don't queue.** Real-time voice queuing (a caller in silence waiting for a slot) is worse
than "busy, try again." Start reject-only; defer queueing (step 5).

Build in layers, simplest/most load-bearing first. All of it is **offline-testable** (fake
adapters, no GPU): the counter, the reject path, the busy-clip branch, and the gauge are pure
logic.

1. **`load_fnc` + `load_threshold` cap, `request_fnc` backstop — the OOM rail `[Done]`.** Both run
   in the **main worker process**. The chosen design leans on the framework rather than a manual
   counter: `load_fnc` reports `len(worker.active_jobs)/capacity`, and LiveKit calls it right before
   each availability check *and* reserves the slot before `request_fnc` runs — so reporting load here
   is race-free and self-decrementing (no `add_shutdown_callback` bookkeeping). `load_threshold` is
   `(cap-0.5)/cap` (the framework rejects `>= 1` in prod). `request_fnc` then rejects + counts any
   job that still slips through (e.g. explicit dispatch). See `_worker_load_fnc` / `_request_fnc`.

2. **Active-session gauge + `/healthz` fields `[Done]`.** Reuses Feature H's existing
   `PROMETHEUS_MULTIPROC_DIR` channel — a multiproc `Gauge(multiprocess_mode="livesum")` for
   `personavoice_sessions_active` is the cross-process live count the token server reads via
   `read_sessions()`. Added `personavoice_sessions_rejected_total`. `/healthz` reports
   `sessions_max` from **config** (static, authoritative, always present) and `sessions_active`
   from the gauge (omitted when no worker is reporting). The metrics are built **lazily on first
   write** so they survive LiveKit's startup wipe of the multiproc dir (see Status); because
   LiveKit clears that dir on every start, the stale-file-after-restart concern is moot.

3. **Pre-rendered "busy" clip on the over-cap path `[Done]`.** A bare `reject()` leaves the caller
   in a silent room. When over cap, `_request_fnc` counts the rejection and *accepts* with a busy
   marker; `entrypoint` short-circuits to `_serve_busy_clip`, which plays a **pre-rendered** clip
   (`orchestrator/busy.py` — a synthesized stdlib busy tone or a `PERSONAVOICE_BUSY_CLIP` WAV, no
   LLM/TTS inference, no model load — zero GPU), sends a `{"event":"busy"}` data message, and
   disconnects, so the user hears why.

4. **Token-server `503` early gate `[Done]` (opt-in).** With `PERSONAVOICE_ADMISSION_503=1` the
   token server reads the shared gauge and returns `503 + Retry-After` (+ `retry_after` body)
   instead of minting when full, so the client never connects. Nicer UX, but it has a TOCTOU race
   (read → mint → connect) — an *optimization* on top of step 1, never a replacement — so it's off
   by default and **fails open** (mints) when the count is unreadable.

5. **Queueing — client-side, lightweight `[Done]`.** Rather than an in-call queue (a caller
   silent in a LiveKit room, which we deliberately avoided), the **client** does a *pre-connection*
   wait: on a `503` it opens a Queue page that re-asks `/token` on the server's `Retry-After`
   interval and drops the user straight into the call the moment a slot frees — no LiveKit session
   is held while waiting (`client/lib/services/queue_controller.dart`, `screens/queue_screen.dart`).
   This needs the step-4 gate **on** so a full worker returns `503`. No server state, so there's no
   real position/fairness: two simultaneous waiters race for the freed slot (the loser hits the
   worker's busy clip and can re-queue). A server-side FIFO with true positions remains a future
   option if demand warrants it.

Steps 1–2 are the real content (the OOM rail + visibility); 3–4 are UX polish; 5 is the
client-side pre-connection queue (a server-side FIFO with positions is the remaining upgrade).

**Acceptance:** two concurrent connects never OOM (the second is rejected, not crashed); the
rejected caller hears the busy clip (step 3); `/healthz` and `/metrics` report live session count
and a rejection counter; capacity is documented per backend and overridable via
`PERSONAVOICE_MAX_SESSIONS`.

---

## 10. Feature J — Web client

Only the Flutter mobile app exists (plus the generic Agents Playground). The Flutter app now has a
real **persona picker + Settings surface** (see Status) — the natural place to add the voice / CEFR /
demeanor selectors before standing up a separate web client. A purpose-built browser client
(LiveKit JS SDK) is still worthwhile to remove the install barrier and is the natural home for the
**voice picker** (Feature B); it reuses the token server and
`/personas` + `/voices` routes. Treat the web client as additive, not a prerequisite for the
session-options UI.

---

## 11. Feature K — Persona authoring helper `[Done]`

> **Folded into N3** (§1.5) — multi-user **JSON persona store** with `POST/PUT/DELETE /personas`
> and client authoring UI. The note below is now an *optional add-on* to N3, not separate work.
>
> **Status:** Done & unit-tested (offline). Core drafter `persona/author.py`
> (`draft_persona(llm, description, *, voices, loras, existing_ids)`): prompt enumerates the legal
> `voice.ref`/`lora` choices + numeric ranges → strict JSON → `Persona.model_validate` with one
> repair retry → clamp voice/lora to servable, slugify+dedupe the id, never author `rude`. Surface:
> `POST /personas/draft?user=` (returns a draft, does **not** persist — drops into the N3 edit form)
> and the `personavoice-persona draft "…"` CLI (prints validated YAML). The LLM is injected
> (`_ChatLLM`), defaulting to the configured cascade backend (offline LM Studio on Mac); a
> Claude-API drafter is a drop-in. Live LLM drafting on a real backend still rides N3's pending
> device/LiveKit verification.

Personas are hand-written YAML validated by `--check`. An optional **generator** turns a
plain-English description into a valid persona *draft* (system prompt + voice + behavior knobs +
session defaults), validated against the `Persona` model + voice registry before it's written to
the user store. Small DX win on top of N3's authoring routes — it lowers the blank-page cost; it
adds no new capability, which is why it stays parked behind N3's device/LiveKit verification.

**Substrate already exists** (why this is S/Low — it's a prompt + validate/repair wrapper):

- `Persona` model with `extra="forbid"` (`src/personavoice/models.py`) — rejects hallucinated
  fields for free.
- Validation path — `Persona.model_validate(raw)` (`persona/loader.py`).
- Voice registry + catalog with `available`/`reason` (`persona/registry.py`) — the legal
  `voice.ref` values.
- Served-LoRA picker — `served_loras(...)` (`persona/lora.py`) — the legal `llm.lora` values.
- User store — `UserPersonaStore.record(user_id, persona)` (`persona/store.py`).
- A configured cascade LLM backend to do the generation.

**Core drafter — new `persona/author.py`:**

```
draft_persona(description, *, voices, loras, existing_ids) -> Persona
```

- Prompt the LLM with the `Persona` schema and the **enumerated** legal `voice.ref` ids and
  `lora` names, plus numeric ranges (`temperature`, `follow_up_probability ∈ [0,1]`, …).
  Constraining the choice set up front is what keeps drafts valid instead of inventing voices.
- Ask for structured JSON, then `Persona.model_validate` it. `extra="forbid"` catches stray
  fields; separately re-resolve `voice.ref` against the registry and `lora` against
  `served_loras` (drop/repair anything not servable on this backend).
- **One repair retry**: on `ValidationError`, feed the error text back to the model; if it still
  fails, return the error rather than a bad persona.
- Derive `id` from `name` (slugify); ensure no clash with curated personas or the user's
  `existing_ids`. Leave `demeanor`/`rude` unset by default — authoring shouldn't opt into the
  rude path.

**Surface — draft-into-form, not auto-persist (recommended):**

- **Server:** `POST /personas/draft` (auth'd, user-scoped) returns a Persona draft **without
  persisting**. The client drops it into the existing N3 New/Edit form pre-filled; the user tweaks
  and confirms; the existing `POST /personas` does the write. Human stays in the loop, reuses all
  of N3.
- **CLI:** a `personavoice-persona draft "a patient French tutor who only speaks in B1"` verb that
  prints validated YAML to stdout / a file — mirrors the `--check` validation path.

**Model choice:** default to the already-configured cascade LLM so it stays fully offline (LM
Studio). Drafting is one-shot and not latency-sensitive, so make the backend pluggable and
optionally allow a Claude API path (e.g. `claude-opus-4-8` / `claude-sonnet-4-6`) for noticeably
better prompt-writing — offline-by-default.

**Tests (offline, per §13):** fake LLM adapter returning canned JSON → assert a valid `Persona`;
bad-field case → repair loop fixes or fails cleanly; invalid `voice.ref`/`lora` → clamped to a
legal value; id-clash case → no curated/own shadowing. Plus `ruff` + `mypy`.

---

## 12. Feature L — Memory introspection (client) `[Done]`

A "what do you remember about me?" surface over the existing per-user store
(`memory/store.py`, `memory/profile.py`), so the consent toggle the client already has is paired
with a way to *see and erase* what consent produced. It's the read/erase half of the privacy story
the `personavoice-memory --show/--export/--delete` CLI gives operators, brought to the end user.

Everything it needs already exists server-side (`MemoryStore.load_profile_raw` / `read_turns` /
`delete_user`, `UserProfile`), so this is wiring, not new mechanism — hence the **S / Low** rating.

**Server — two routes on the token server, mirroring `/consent`** (`server/token_server.py`):

- `GET /memory?user=[&limit=]` → `TokenService.get_memory`: the distilled profile (rolling
  `summary` + timestamped `facts`) plus the last `limit` raw turns (default 20), and the current
  `granted` flag so the UI can caption the screen. Trimmed analog of `MemoryStore.export_user`.
- `DELETE /memory?user=` → `TokenService.delete_memory`: the HTTP analog of `--delete`. Idempotent
  — wiping a user with nothing stored returns `{"deleted": false}` rather than erroring, so the
  client's "Forget me" button is safe to press twice.

**Decision — read is *not* gated on current consent.** Unlike `recall` (which returns "" without
consent), `get_memory` shows whatever is *stored* regardless of the live consent flag, because
revoking consent **keeps** existing data (it only gates future recording/recall). Hiding retained
data while it still sits on disk would be the privacy anti-pattern; surfacing it — captioned by
`granted` — is what makes the paired Delete meaningful. This matches `export_user`, which also
dumps regardless of consent. Same client-asserted `?user=` trust model as every route here (a
caller with the shared API token can act as any id; documented on `set_consent`).

**Why a route, not a data message.** Introspection is a request/response *pull* from a settings
screen, needed when no call is active, and can be large (40 facts + recent turns). The
data-message channel (`orchestrator/agent.py`) is for mid-call persona/option swaps — the wrong
shape. `/consent` is the precedent: same `?user=` scoping, same `_require_memory()` backing.

**Client (`client/lib`):** `TokenClient.fetchMemory` / `deleteMemory` + a `MemorySnapshot` model,
and a read-only **"What I remember"** screen (summary + facts + recent turns, with Forget/refresh)
reached from the existing **Privacy** section in `settings_screen.dart`, next to the consent toggle.

**Scope boundary.** Show + wipe-all only. Editing individual facts from the client is deliberately
out of scope: it would turn a read surface into a write surface with its own consistency questions
against the background consolidation in `ConversationMemory` — a possible follow-up, not part of L.

**Tests (offline, per §13):** service-level (seeded store → facts/summary/turns; empty user →
empty payload + `granted:false`; delete idempotency; disabled-without-store raises) and HTTP-level
(round-trip GET/DELETE, auth required, bad `limit` → 400), mirroring the `/consent` suite. Plus
`ruff` + `mypy`. Client unit test for `MemorySnapshot.fromJson`.

---

## 13. Cross-cutting checklist (applies to every server-side item)

- `pytest` green offline (fake adapters, no GPU); `ruff check` + `ruff format --check`; `mypy src`.
- `.env.example` updated for any new env var; README section updated.
- `python -m personavoice.server --check` extended where a new capability can be misconfigured
  (e.g. clones present but active TTS can't speak them; auth required for `/voices/clone`).
- Backward compatible: every new field is optional and defaults to today's behavior.

---

## 14. Headline path (the requested trio, end to end)

1. **C** `[Done]` — moderation seam (no-op default) so `rude` is safe by construction.
2. **A** `[Done]` — `SessionOptions` + CEFR/demeanor directives + voice-override plumbing +
   token-server validation + offline flags on the demos. Fully testable without the client.
3. **B** `[Done]` — voice **catalog** + `resolve_choice` + the **clone enrollment** endpoints, so
   the user has a real library to pick from. Cloning-backend requirement documented.
4. **Flutter client** `[Partial]` — the persona picker + Settings surface exist (see Status). The
   remaining client work is now the **NOW block (§1.5): N1** (per-persona voice/CEFR/demeanor
   selectors → `/token` body), **N2** (Voice Library page consuming `/voices` + clone enrollment),
   **N4** (seed voices), then **N3** (multi-user custom personas + LoRA pick — the only new server
   surface). A separate web client stays optional/additive.

Steps 1–3 are server-only and unit-testable; the client UI is the last, thinnest layer — its
scaffold (picker + settings + preferences) is already in place, and **§1.5 (NOW)** is the detailed
spec for finishing it.

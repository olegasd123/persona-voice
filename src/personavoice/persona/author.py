"""Persona authoring helper: draft a valid `Persona` from plain English.

`POST /personas?user=` already lets a user author a persona, but it's a blank-page problem:
the client has to fill in a system prompt, a voice ref, behavior knobs and session defaults
by hand. This module is the *optional* generator on top of N3's authoring routes — it turns a
one-line description ("a patient French tutor who only speaks in B1") into a validated persona
*draft* the client drops into the existing New/Edit form. The human stays in the loop: the
draft is never persisted here; the user tweaks and confirms, and the existing `POST /personas`
does the write.

It is a thin, *mostly pure* prompt + validate/repair wrapper, mirroring `memory/profile.py`:

    description ──▶ build_author_prompt ──▶ LLM ──▶ parse_persona_draft ──▶ Persona.model_validate
                                                          (+ one repair retry on a bad draft)

Everything that keeps a draft valid is local, not trusted to the model:

- The prompt **enumerates** the legal `voice.ref` ids and served `lora` names plus the numeric
  ranges, so the model picks from a constrained set instead of inventing a voice.
- `Persona`'s `extra="forbid"` rejects hallucinated top-level fields for free; on a
  `ValidationError` we feed the error text back for **one** repair retry, then fail cleanly.
- After validation we *re-resolve* `voice.ref` against the registry and `lora` against the
  served adapters and clamp anything not selectable/servable on this backend.
- The `id` is derived from `name` (slugified) and de-duped against the curated + the user's own
  ids, so a draft can't shadow a built-in. `demeanor` is never authored to `rude` — authoring
  shouldn't opt into the rude path.

The LLM is **injected** (the `_ChatLLM` slice below), so it's offline-testable with a fake and
pluggable: the server/CLI default to the already-configured cascade backend (LM Studio on Mac),
but a Claude-API drafter is a drop-in for noticeably better prompt-writing.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from ..models import (
    CEFRLevel,
    Demeanor,
    LLMSettings,
    Msg,
    Persona,
    Role,
    VoiceSettings,
)

# Sampling for the one-shot draft: low temperature for stable structured JSON, a generous token
# budget so a long system_prompt isn't truncated mid-object. Drafting isn't latency-sensitive.
_DRAFT_TEMPERATURE = 0.4
_DRAFT_MAX_TOKENS = 1500

_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")


class PersonaDraftError(ValueError):
    """The model's reply could not be turned into a valid persona (after the repair retry)."""


class _ChatLLM(Protocol):
    """The slice of an LLM adapter the drafter needs (one full, non-streamed reply).

    `persona` is typed as `Persona` (not `object`) so a real `LLMAdapter` is a structural match;
    we always hand it the throwaway `_drafting_persona()` that carries the sampling params.
    """

    async def chat(self, messages: list[Msg], persona: Persona) -> str: ...


# --------------------------------------------------------------------------------------
# Id derivation
# --------------------------------------------------------------------------------------


def slugify(name: str) -> str:
    """A bare-identifier persona id from a name ('My Tutor' → 'my-tutor'; '' → 'persona')."""
    slug = _SLUG_STRIP_RE.sub("-", name.strip().lower()).strip("-")
    return slug or "persona"


def unique_persona_id(name: str, existing_ids: Collection[str]) -> str:
    """Slug of `name`, suffixed with `-2`, `-3`, … until it clashes with no id in `existing_ids`.

    `existing_ids` is the curated persona ids plus the user's own, so a draft never shadows a
    built-in or one of the user's existing personas.
    """
    base = slugify(name)
    existing = set(existing_ids)
    candidate, n = base, 2
    while candidate in existing:
        candidate, n = f"{base}-{n}", n + 1
    return candidate


# --------------------------------------------------------------------------------------
# Prompt (pure)
# --------------------------------------------------------------------------------------


def _enumerate(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "(none available)"


def build_author_prompt(
    description: str, *, voices: Sequence[str], loras: Sequence[str]
) -> list[Msg]:
    """The message list asking the LLM to draft a persona from `description` (pure).

    Shows the **exact nested JSON shape** (with the legal `voice.ref` ids and served `lora` names
    enumerated inline) rather than dotted field paths, because a local model will otherwise copy a
    path like `voice.ref` as a literal flat key — which `Persona`'s `extra="forbid"` rejects.
    """
    cefr_values = ", ".join(level.value for level in CEFRLevel)
    # The `llm` object: include the `lora` slot only when adapters are actually served.
    if loras:
        llm_field = (
            '  "llm": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 280, '
            f'"lora": "<one of: {_enumerate(loras)}, or drop the key>"}},\n'
        )
        lora_note = ""
    else:
        llm_field = '  "llm": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 280},\n'
        lora_note = "No LoRA adapters are served on this backend, so omit the lora field. "
    voice_field = (
        f'  "voice": {{"ref": "<one of: {_enumerate(voices)}>", "emotion": "warm"}},\n'
        if voices
        else '  (omit "voice": no voice library is configured)\n'
    )
    system = (
        "You design personas for a voice assistant. Turn the user's plain-English description into "
        "ONE persona as a single JSON object — no prose, no markdown fences, JSON only.\n\n"
        "Use this EXACT nested shape. Use nested objects; do NOT flatten keys — write "
        '{"voice": {"ref": "..."}}, never {"voice.ref": "..."}. Omit any field to take its '
        "default:\n"
        "{\n"
        '  "name": "<short display name, e.g. Patient French Tutor>",\n'
        '  "description": "<one-line blurb shown in a picker>",\n'
        '  "system_prompt": "<instructions addressed to the persona as \\"You are …\\"; capture '
        'its character, tone and rules; vivid and specific>",\n'
        f"{llm_field}"
        f"{voice_field}"
        '  "behavior": {"turn_style": "<concise|balanced|verbose>", '
        '"follow_up_probability": 0.5},\n'
        '  "memory": {"enabled": false},\n'
        f'  "session_defaults": {{"cefr": "<one of {cefr_values}; omit unless a fixed level '
        'fits, e.g. a B1 tutor>"}}\n'
        "}\n\n"
        "Numeric ranges: temperature 0.0-1.2, top_p 0.0-1.0, max_tokens 120-400, "
        "follow_up_probability 0.0-1.0. "
        f"{lora_note}"
        "Do NOT set a demeanor. Do NOT invent voices, LoRA names, tools, or any field not shown "
        "above. Reply with ONLY the JSON object."
    )
    user = (
        f"Description:\n{description.strip()}\n\n"
        "Draft the persona now. Return only the JSON object."
    )
    return [Msg(role=Role.system, content=system), Msg(role=Role.user, content=user)]


def _repair_instruction(error: str) -> str:
    """A follow-up turn that hands the validation error back for one corrected attempt."""
    return (
        f"That draft was invalid: {error}\n"
        "Return a corrected JSON object that fixes this. Use only the fields and allowed values "
        "described above. Reply with ONLY the JSON object."
    )


# --------------------------------------------------------------------------------------
# Parsing (pure)
# --------------------------------------------------------------------------------------


def parse_persona_draft(text: str) -> dict[str, Any] | None:
    """Best-effort: parse the first JSON object in `text` (tolerates ```json fences / prose)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(cleaned[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _format_validation_error(exc: ValidationError) -> str:
    """A compact, model-feedable summary of the first pydantic error (loc + message)."""
    errors = exc.errors()
    if not errors:
        return str(exc)
    first = errors[0]
    loc = ".".join(str(p) for p in first.get("loc", ())) or "persona"
    return f"{loc}: {first.get('msg')}"


def unflatten_dotted_keys(data: dict[str, Any]) -> dict[str, Any]:
    """Fold dotted top-level keys into nested objects: `{"voice.ref": x}` → `{"voice": {"ref": x}}`.

    Local models sometimes emit the field *paths* from the prompt as flat keys; `Persona`'s
    `extra="forbid"` would reject them (`voice.ref: Extra inputs are not permitted`). Splitting a
    single-dot key back into a nested dict — merged with any real nested object for the same parent
    — recovers a valid shape. No persona field name contains a dot, so this only ever rescues a
    flattened draft; deeper paths and plain keys pass through untouched.
    """
    out: dict[str, Any] = {}
    for key, value in data.items():
        parent, sep, child = key.partition(".")
        if sep and child and "." not in child:
            bucket = out.get(parent)
            if not isinstance(bucket, dict):
                bucket = {}
                out[parent] = bucket
            bucket.setdefault(child, value)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**value, **out[key]}  # merge, letting folded dotted keys win
        else:
            out[key] = value
    return out


# --------------------------------------------------------------------------------------
# Clamp to legal / servable choices (pure)
# --------------------------------------------------------------------------------------


def _coerce_voice_ref(ref: str, *, voices: Sequence[str], default_voice: str | None) -> str:
    """Keep `ref` if it's a known voice; otherwise fall back to the default / first legal one.

    With no voice library configured (`voices` empty) the ref passes through unchanged — the
    registry tolerates an unknown ref and the adapter then uses its own default voice.
    """
    legal = list(voices)
    if not legal or ref in legal:
        return ref
    if default_voice and default_voice in legal:
        return default_voice
    return legal[0]


def _coerce_lora(lora: str | None, *, loras: Sequence[str]) -> str | None:
    """Keep a `lora` only when its basename is one of the served adapters; else drop it.

    Mirrors the request-time routing (`lora_request_model`), which selects a served adapter by
    basename. On a non-LoRA backend `loras` is empty, so any drafted LoRA is dropped.
    """
    if not lora:
        return None
    return lora if Path(lora).name in set(loras) else None


def _clamp_choices(
    persona: Persona, *, voices: Sequence[str], loras: Sequence[str], default_voice: str | None
) -> Persona:
    """Re-resolve a validated draft's voice/lora/demeanor against what this backend can do."""
    updates: dict[str, Any] = {}
    new_ref = _coerce_voice_ref(persona.voice.ref, voices=voices, default_voice=default_voice)
    if new_ref != persona.voice.ref:
        updates["voice"] = persona.voice.model_copy(update={"ref": new_ref})
    new_lora = _coerce_lora(persona.llm.lora, loras=loras)
    if new_lora != persona.llm.lora:
        updates["llm"] = persona.llm.model_copy(update={"lora": new_lora})
    # Never let authoring opt into the rude path; a kind/natural default is fine.
    if persona.session_defaults.demeanor == Demeanor.rude:
        updates["session_defaults"] = persona.session_defaults.model_copy(update={"demeanor": None})
    return persona.model_copy(update=updates) if updates else persona


# --------------------------------------------------------------------------------------
# Drafter
# --------------------------------------------------------------------------------------


def _drafting_persona() -> Persona:
    """A throwaway persona that only carries the drafting sampling params for the adapter."""
    return Persona(
        id="persona-drafter",
        name="Persona Drafter",
        system_prompt="(persona authoring)",
        llm=LLMSettings(
            base_model="drafter",
            temperature=_DRAFT_TEMPERATURE,
            max_tokens=_DRAFT_MAX_TOKENS,
        ),
        voice=VoiceSettings(ref="voices/_drafter"),
    )


def _persona_from_reply(
    text: str,
    *,
    voices: Sequence[str],
    loras: Sequence[str],
    existing_ids: Collection[str],
    default_voice: str | None,
    default_base_model: str | None,
) -> Persona:
    """Build a clamped `Persona` from one model reply, raising `PersonaDraftError` on a bad draft.

    Fills the two required fields a minimal draft may omit (`voice.ref`, `llm.base_model`) from
    the curated defaults, assigns a unique id, validates, then clamps voice/lora/demeanor.
    """
    obj = parse_persona_draft(text)
    if obj is None:
        raise PersonaDraftError("the model did not return a JSON object")
    # Rescue a model that flattened the nested shape into dotted keys before anything else looks
    # at the fields (so `voice.ref` becomes `voice: {ref: ...}` rather than a forbidden extra).
    obj = unflatten_dotted_keys(obj)
    name = obj.get("name")
    if not isinstance(name, str) or not name.strip():
        raise PersonaDraftError("the draft is missing a 'name'")

    # The id is server-assigned (a model-sent id/user is ignored).
    data = {k: v for k, v in obj.items() if k not in ("id", "user")}
    data["id"] = unique_persona_id(name, existing_ids)

    voice = dict(data["voice"]) if isinstance(data.get("voice"), dict) else {}
    if not voice.get("ref"):
        voice["ref"] = default_voice or (voices[0] if voices else "voices/companion_soft")
    data["voice"] = voice
    llm = dict(data["llm"]) if isinstance(data.get("llm"), dict) else {}
    if not llm.get("base_model"):
        llm["base_model"] = default_base_model or "qwen2.5-7b-instruct"
    data["llm"] = llm

    try:
        persona = Persona.model_validate(data)
    except ValidationError as exc:
        raise PersonaDraftError(_format_validation_error(exc)) from exc
    return _clamp_choices(persona, voices=voices, loras=loras, default_voice=default_voice)


async def draft_persona(
    llm: _ChatLLM,
    description: str,
    *,
    voices: Sequence[str],
    loras: Sequence[str],
    existing_ids: Collection[str],
    default_voice: str | None = None,
    default_base_model: str | None = None,
    repair: bool = True,
) -> Persona:
    """Draft a validated `Persona` from a plain-English `description` via the injected `llm`.

    `voices` are the legal `voice.ref` ids (e.g. ``voices/companion_soft``); `loras` the served
    adapter names; `existing_ids` the curated + the user's own ids the draft must not shadow.
    `default_voice` / `default_base_model` fill the two required fields a minimal draft may omit
    (typically from the curated default persona).

    On a `ValidationError` the error text is fed back for **one** repair retry; a second failure
    raises `PersonaDraftError` rather than returning a bad persona.
    """
    description = (description or "").strip()
    if not description:
        raise PersonaDraftError("a persona description is required")
    voices = list(voices)
    loras = list(loras)

    prompt = build_author_prompt(description, voices=voices, loras=loras)
    raw = await llm.chat(prompt, _drafting_persona())
    try:
        return _persona_from_reply(
            raw,
            voices=voices,
            loras=loras,
            existing_ids=existing_ids,
            default_voice=default_voice,
            default_base_model=default_base_model,
        )
    except PersonaDraftError as exc:
        if not repair:
            raise
        repair_prompt = [
            *prompt,
            Msg(role=Role.assistant, content=raw),
            Msg(role=Role.user, content=_repair_instruction(str(exc))),
        ]
        raw = await llm.chat(repair_prompt, _drafting_persona())
        # A second failure propagates a clean PersonaDraftError (no bad persona is returned).
        return _persona_from_reply(
            raw,
            voices=voices,
            loras=loras,
            existing_ids=existing_ids,
            default_voice=default_voice,
            default_base_model=default_base_model,
        )

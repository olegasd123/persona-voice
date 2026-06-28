"""Persona authoring helper: prompt building, parsing, id derivation, drafting + repair/clamp."""

from __future__ import annotations

import json

import pytest

from personavoice.models import CEFRLevel, Demeanor, Persona
from personavoice.persona.author import (
    PersonaDraftError,
    build_author_prompt,
    draft_persona,
    parse_persona_draft,
    slugify,
    unique_persona_id,
)

VOICES = ["voices/companion_soft", "voices/teacher_clear"]
LORAS = ["hr_interviewer", "pm"]


class _ReplyLLM:
    """LLM stub: returns queued replies from `.chat`, one per call (last one repeats).

    A single string is treated as a one-element queue, so most tests just pass the canned JSON.
    The repair-loop tests pass `[bad, good]` to drive the first-fails-then-fixes path.
    """

    def __init__(self, replies: str | list[str]) -> None:
        self._replies = [replies] if isinstance(replies, str) else list(replies)
        self.calls = 0

    async def chat(self, messages: object, persona: object) -> str:
        reply = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        return reply


def _draft_json(**overrides: object) -> str:
    body: dict[str, object] = {
        "name": "French Tutor",
        "description": "A gentle French tutor.",
        "system_prompt": "You are a patient French tutor.",
        "voice": {"ref": "voices/teacher_clear", "emotion": "warm"},
    }
    body.update(overrides)
    return json.dumps(body)


async def _draft(llm: _ReplyLLM, *, existing: set[str] | None = None, **kw: object) -> Persona:
    return await draft_persona(
        llm,
        "a patient French tutor",
        voices=VOICES,
        loras=LORAS,
        existing_ids=existing or set(),
        default_voice="voices/companion_soft",
        default_base_model="qwen2.5-7b-instruct",
        **kw,
    )


# --------------------------------------------------------------------------------------
# slugify / unique_persona_id
# --------------------------------------------------------------------------------------


def test_slugify_basic() -> None:
    assert slugify("My French Tutor!") == "my-french-tutor"
    assert slugify("   ") == "persona"


def test_unique_persona_id_avoids_clashes() -> None:
    assert unique_persona_id("Companion", {"companion"}) == "companion-2"
    assert unique_persona_id("Companion", {"companion", "companion-2"}) == "companion-3"
    assert unique_persona_id("Fresh", {"companion"}) == "fresh"


# --------------------------------------------------------------------------------------
# build_author_prompt
# --------------------------------------------------------------------------------------


def test_prompt_enumerates_voices_and_loras() -> None:
    msgs = build_author_prompt("a tutor", voices=VOICES, loras=LORAS)
    system = msgs[0].content
    assert "voices/companion_soft" in system and "voices/teacher_clear" in system
    assert "hr_interviewer" in system and "pm" in system
    assert "a tutor" in msgs[1].content


def test_prompt_handles_empty_voice_and_lora_sets() -> None:
    system = build_author_prompt("a tutor", voices=[], loras=[]).pop(0).content
    assert "no voice library is configured" in system
    assert "no LoRA adapters are served" in system


# --------------------------------------------------------------------------------------
# parse_persona_draft
# --------------------------------------------------------------------------------------


def test_parse_fenced_and_embedded_json() -> None:
    assert parse_persona_draft('```json\n{"name": "X"}\n```') == {"name": "X"}
    assert parse_persona_draft('Sure:\n{"name": "Y"}\nDone.') == {"name": "Y"}
    assert parse_persona_draft("no json here") is None


# --------------------------------------------------------------------------------------
# draft_persona — happy path
# --------------------------------------------------------------------------------------


async def test_draft_canned_json_yields_valid_persona() -> None:
    persona = await _draft(_ReplyLLM(_draft_json()))
    assert isinstance(persona, Persona)
    assert persona.id == "french-tutor"  # slug of the name
    assert persona.system_prompt == "You are a patient French tutor."
    assert persona.voice.ref == "voices/teacher_clear"


async def test_draft_fills_required_defaults_from_minimal_reply() -> None:
    # A minimal reply (name + system_prompt) still validates: voice.ref + base_model are filled.
    reply = json.dumps({"name": "Minimal", "system_prompt": "You are minimal."})
    persona = await _draft(_ReplyLLM(reply))
    assert persona.voice.ref == "voices/companion_soft"  # default_voice
    assert persona.llm.base_model == "qwen2.5-7b-instruct"  # default_base_model


async def test_draft_sets_session_default_cefr() -> None:
    reply = _draft_json(session_defaults={"cefr": "b1"})
    persona = await _draft(_ReplyLLM(reply))
    assert persona.session_defaults.cefr == CEFRLevel.b1


# --------------------------------------------------------------------------------------
# draft_persona — clamping
# --------------------------------------------------------------------------------------


async def test_draft_clamps_unknown_voice_ref() -> None:
    persona = await _draft(_ReplyLLM(_draft_json(voice={"ref": "voices/made_up"})))
    assert persona.voice.ref == "voices/companion_soft"  # clamped to the default legal voice


async def test_draft_drops_unservable_lora() -> None:
    reply = _draft_json(llm={"base_model": "m", "lora": "adapters/ghost"})
    persona = await _draft(_ReplyLLM(reply))
    assert persona.llm.lora is None  # not among the served adapters → dropped


async def test_draft_keeps_servable_lora_by_basename() -> None:
    reply = _draft_json(llm={"base_model": "m", "lora": "adapters/hr_interviewer"})
    persona = await _draft(_ReplyLLM(reply))
    assert persona.llm.lora == "adapters/hr_interviewer"  # basename is served → kept


async def test_draft_never_authors_rude_demeanor() -> None:
    reply = _draft_json(session_defaults={"demeanor": "rude"})
    persona = await _draft(_ReplyLLM(reply))
    assert persona.session_defaults.demeanor is None


async def test_draft_keeps_kind_demeanor() -> None:
    reply = _draft_json(session_defaults={"demeanor": "kind"})
    persona = await _draft(_ReplyLLM(reply))
    assert persona.session_defaults.demeanor == Demeanor.kind


async def test_draft_voice_ref_passes_through_without_registry() -> None:
    # No voice library configured → an arbitrary ref is left as-is (the adapter defaults).
    persona = await draft_persona(
        _ReplyLLM(_draft_json(voice={"ref": "voices/whatever"})),
        "a tutor",
        voices=[],
        loras=[],
        existing_ids=set(),
    )
    assert persona.voice.ref == "voices/whatever"


# --------------------------------------------------------------------------------------
# draft_persona — id clash
# --------------------------------------------------------------------------------------


async def test_draft_id_never_shadows_existing() -> None:
    persona = await _draft(
        _ReplyLLM(_draft_json(name="Companion")), existing={"companion", "french-tutor"}
    )
    assert persona.id == "companion-2"  # de-duped against curated + the user's own


# --------------------------------------------------------------------------------------
# draft_persona — repair loop
# --------------------------------------------------------------------------------------


async def test_draft_repairs_a_bad_field() -> None:
    bad = _draft_json(bogus_field=1)  # extra top-level field → Persona's extra="forbid" rejects it
    good = _draft_json()
    llm = _ReplyLLM([bad, good])
    persona = await _draft(llm)
    assert llm.calls == 2  # one draft + one repair
    assert persona.id == "french-tutor"


async def test_draft_fails_cleanly_when_repair_also_bad() -> None:
    bad = _draft_json(bogus_field=1)
    llm = _ReplyLLM([bad, bad])
    with pytest.raises(PersonaDraftError):
        await _draft(llm)
    assert llm.calls == 2  # didn't loop forever


async def test_draft_no_repair_raises_on_first_bad() -> None:
    llm = _ReplyLLM([_draft_json(bogus_field=1), _draft_json()])
    with pytest.raises(PersonaDraftError):
        await _draft(llm, repair=False)
    assert llm.calls == 1  # repair disabled → no retry


async def test_draft_rejects_missing_name() -> None:
    reply = json.dumps({"system_prompt": "You are nameless."})
    with pytest.raises(PersonaDraftError, match="name"):
        await _draft(_ReplyLLM([reply, reply]))


async def test_draft_rejects_non_json_reply() -> None:
    with pytest.raises(PersonaDraftError):
        await _draft(_ReplyLLM(["sorry, I can't help", "still no json"]))


async def test_draft_requires_description() -> None:
    with pytest.raises(PersonaDraftError, match="description"):
        await draft_persona(_ReplyLLM("{}"), "   ", voices=VOICES, loras=LORAS, existing_ids=set())

"""Persona loading, validation, prompt building, and the registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from personavoice.models import CEFRLevel, Demeanor, Role, SessionOptions
from personavoice.persona import (
    PersonaRegistry,
    build_messages,
    load_persona,
    render_system_prompt,
)
from personavoice.persona.loader import PersonaError

PERSONA_IDS = ["pm_interviewer", "hr_interviewer", "language_teacher", "companion"]


def test_all_shipped_personas_load(config_dir: Path) -> None:
    registry = PersonaRegistry(config_dir / "personas")
    assert set(registry.ids()) == set(PERSONA_IDS)
    for persona in registry.all():
        assert persona.system_prompt.strip()
        assert persona.llm.base_model


def test_system_prompt_reflects_turn_style(config_dir: Path) -> None:
    pm = load_persona(config_dir / "personas" / "pm_interviewer.yaml")
    prompt = render_system_prompt(pm)
    assert "short and to the point" in prompt  # concise directive
    assert "without markdown" in prompt  # spoken-output guardrail


def test_build_messages_shape(config_dir: Path) -> None:
    persona = load_persona(config_dir / "personas" / "companion.yaml")
    messages = build_messages(persona, history=None, user_input="hi there")
    assert messages[0].role == Role.system
    assert messages[-1].role == Role.user
    assert messages[-1].content == "hi there"


def test_demeanor_natural_is_a_noop(config_dir: Path) -> None:
    persona = load_persona(config_dir / "personas" / "companion.yaml")
    base = render_system_prompt(persona)
    natural = render_system_prompt(persona, SessionOptions(demeanor=Demeanor.natural))
    assert natural == base  # the authored persona, unchanged


def test_demeanor_rude_directive_is_bounded(config_dir: Path) -> None:
    persona = load_persona(config_dir / "personas" / "companion.yaml")
    prompt = render_system_prompt(persona, SessionOptions(demeanor=Demeanor.rude))
    assert "brusque" in prompt
    assert "not abusive" in prompt  # the bound the moderation layer then enforces


def test_demeanor_kind_directive_present(config_dir: Path) -> None:
    persona = load_persona(config_dir / "personas" / "companion.yaml")
    prompt = render_system_prompt(persona, SessionOptions(demeanor=Demeanor.kind))
    assert "warm" in prompt and "patient" in prompt


def test_cefr_directive_injected_per_level(config_dir: Path) -> None:
    persona = load_persona(config_dir / "personas" / "language_teacher.yaml")
    for level in CEFRLevel:
        prompt = render_system_prompt(persona, SessionOptions(cefr=level))
        assert level.value in prompt  # "A1".."C2" named in its directive


def test_options_absent_leaves_prompt_unchanged(config_dir: Path) -> None:
    persona = load_persona(config_dir / "personas" / "companion.yaml")
    assert render_system_prompt(persona, SessionOptions()) == render_system_prompt(persona)


def test_spoken_language_nudge_stays_last(config_dir: Path) -> None:
    persona = load_persona(config_dir / "personas" / "language_teacher.yaml")
    prompt = render_system_prompt(
        persona, SessionOptions(demeanor=Demeanor.rude, cefr=CEFRLevel.a1)
    )
    # The spoken-output guardrail must remain the final directive, after the overrides.
    assert prompt.rstrip().endswith("without markdown, lists, or emoji.")


def test_build_messages_threads_options(config_dir: Path) -> None:
    persona = load_persona(config_dir / "personas" / "companion.yaml")
    messages = build_messages(
        persona, user_input="hi", options=SessionOptions(demeanor=Demeanor.rude)
    )
    assert "brusque" in messages[0].content


def test_unknown_persona_raises(config_dir: Path) -> None:
    registry = PersonaRegistry(config_dir / "personas")
    with pytest.raises(PersonaError):
        registry.get("does_not_exist")


def test_id_must_match_filename(tmp_path: Path) -> None:
    (tmp_path / "wrong.yaml").write_text(
        "id: right\nname: X\nsystem_prompt: hi\nllm: {base_model: m}\nvoice: {ref: voices/x}\n"
    )
    from personavoice.persona.loader import load_personas

    with pytest.raises(PersonaError, match="does not match filename"):
        load_personas(tmp_path)

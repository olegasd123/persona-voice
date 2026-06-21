"""Persona loading, validation, prompt building, and the registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from personavoice.models import Role
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

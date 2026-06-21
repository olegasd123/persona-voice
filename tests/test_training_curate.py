"""Dialogue curation: self-chat structure, role-flipping, simulator prompt."""

from __future__ import annotations

from collections.abc import AsyncIterator

from personavoice.adapters.llm.base import LLMAdapter
from personavoice.models import Msg, Persona, Role
from personavoice.training.curate import (
    DialogueGenerator,
    _swap_roles,
    generate_dataset,
    user_simulator_prompt,
)
from personavoice.training.dataset import validate_example

from .fakes import FakeLLM, make_persona


class RoleEchoLLM(LLMAdapter):
    """Echoes which role's system prompt it saw, so tests can tell the two roles apart."""

    name = "role_echo"

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[Msg]] = []

    async def stream_chat(self, messages: list[Msg], persona: Persona) -> AsyncIterator[str]:
        self.calls.append(list(messages))
        system = messages[0].content if messages else ""
        yield "USER" if "role-playing" in system else "PERSONA"


def test_user_simulator_prompt_describes_other_side() -> None:
    persona = make_persona("hr_interviewer")
    prompt = user_simulator_prompt(persona, scenario="You are nervous.")
    assert "role-playing" in prompt
    assert "the human user, not the assistant" in prompt
    assert "You are nervous." in prompt


def test_swap_roles_flips_user_and_assistant() -> None:
    swapped = _swap_roles([Msg(role=Role.user, content="a"), Msg(role=Role.assistant, content="b")])
    assert [m.role for m in swapped] == [Role.assistant, Role.user]


async def test_generate_produces_valid_alternating_example() -> None:
    gen = DialogueGenerator(FakeLLM("ok reply"), make_persona())
    ex = await gen.generate(num_exchanges=3)
    validate_example(ex)  # system + strictly alternating, ends on assistant
    assert ex.messages[0].role is Role.system
    # system + 3 user/assistant exchanges
    assert len(ex.messages) == 1 + 3 * 2


async def test_generate_honors_opening() -> None:
    gen = DialogueGenerator(FakeLLM("ok"), make_persona())
    ex = await gen.generate(num_exchanges=1, opening="My fixed first line.")
    assert ex.messages[1].role is Role.user
    assert ex.messages[1].content == "My fixed first line."


async def test_generate_uses_distinct_roles_for_user_and_persona() -> None:
    gen = DialogueGenerator(RoleEchoLLM(), make_persona())
    ex = await gen.generate(num_exchanges=2)
    # user turns come from the simulator (sees the role-play system), persona turns don't
    user_turns = [m.content for m in ex.messages if m.role is Role.user]
    persona_turns = [m.content for m in ex.messages if m.role is Role.assistant]
    assert set(user_turns) == {"USER"}
    assert set(persona_turns) == {"PERSONA"}


async def test_generate_stops_on_empty_turn() -> None:
    gen = DialogueGenerator(FakeLLM(""), make_persona())
    ex = await gen.generate(num_exchanges=3)
    # the very first (empty) user turn aborts -> only the system message remains
    assert [m.role for m in ex.messages] == [Role.system]


async def test_generate_dataset_count_and_validity() -> None:
    examples = await generate_dataset(
        FakeLLM("a reply"), make_persona(), num_dialogues=5, num_exchanges=2
    )
    assert len(examples) == 5
    for ex in examples:
        validate_example(ex)

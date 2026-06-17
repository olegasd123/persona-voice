"""Curate in-character training dialogues by self-chat (M7).

The cheapest way to bootstrap a persona dataset is to let the persona talk to a *simulated*
user. Two roles share one LLM:

  - the **persona** answers (its real system prompt + behavior directives, via `build_messages`);
  - a **user simulator** plays a realistic conversation partner for that persona (a candidate for
    an interviewer, a learner for the teacher, ...). To make the model emit the *user's* next
    line we flip the transcript's roles, so from the simulator's view it is the assistant.

`DialogueGenerator.generate` produces a `DialogueExample` (system + alternating turns ending on
the persona) ready for `dataset.write_dataset`. The role-flipping and prompt building are pure
and unit-tested; the only impurity is the injected `LLMAdapter`, so a fake drives the tests.

These synthetic dialogues are a *seed*: curate, then hand-review/edit before training. They are
not a substitute for real, in-character data.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from ..adapters.llm.base import LLMAdapter
from ..models import Msg, Persona, Role
from ..persona.prompt import build_messages, render_system_prompt
from .dataset import DialogueExample

logger = logging.getLogger("personavoice.training.curate")

# A few neutral openers so generated dialogues don't all start identically. The simulator is
# free to ignore them; they just seed variety when no explicit opening is supplied.
DEFAULT_SCENARIOS = (
    "You are starting the conversation.",
    "You have a specific situation in mind you want to talk through.",
    "You are a little hesitant and need to be drawn out.",
    "You are upbeat and talkative.",
)


def user_simulator_prompt(persona: Persona, *, scenario: str | None = None) -> str:
    """System prompt that makes the LLM role-play a realistic partner for `persona`."""
    parts = [
        f"You are role-playing a person talking to {persona.name}.",
        f"For context, that persona is described as: {persona.system_prompt.strip()}",
        "Play the OTHER side of this conversation — the human user, not the assistant. "
        "Speak naturally in the first person, one short conversational turn at a time. "
        "Stay in character, never break role, and do not narrate or use stage directions. "
        "Respond only with what the person would say out loud.",
    ]
    if scenario:
        parts.append(scenario)
    return "\n\n".join(parts)


def _swap_roles(history: Sequence[Msg]) -> list[Msg]:
    """Flip user<->assistant so the persona's turns become inputs to the simulator."""
    swap = {Role.user: Role.assistant, Role.assistant: Role.user}
    return [Msg(role=swap.get(m.role, m.role), content=m.content) for m in history]


class DialogueGenerator:
    """Drive one persona<->simulator conversation to a `DialogueExample`."""

    def __init__(self, llm: LLMAdapter, persona: Persona) -> None:
        self._llm = llm
        self._persona = persona

    async def _persona_turn(self, history: list[Msg]) -> str:
        messages = build_messages(self._persona, history=history)
        return (await self._llm.chat(messages, self._persona)).strip()

    async def _user_turn(self, history: list[Msg], scenario: str | None) -> str:
        sim_system = Msg(
            role=Role.system, content=user_simulator_prompt(self._persona, scenario=scenario)
        )
        messages = [sim_system, *_swap_roles(history)]
        return (await self._llm.chat(messages, self._persona)).strip()

    async def generate(
        self,
        *,
        num_exchanges: int = 4,
        opening: str | None = None,
        scenario: str | None = None,
    ) -> DialogueExample:
        """Generate one dialogue: `system` + `num_exchanges` user→assistant exchanges.

        `opening` fixes the user's first line (else the simulator invents one); `scenario`
        biases the simulated user. Empty turns abort early so a flaky model can't produce a
        malformed example.
        """
        if num_exchanges < 1:
            raise ValueError("num_exchanges must be >= 1")

        history: list[Msg] = []
        for i in range(num_exchanges):
            if i == 0 and opening:
                user_text = opening.strip()
            else:
                user_text = await self._user_turn(history, scenario)
            if not user_text:
                logger.warning("empty user turn at exchange %d; stopping early", i)
                break
            history.append(Msg(role=Role.user, content=user_text))

            assistant_text = await self._persona_turn(history)
            if not assistant_text:
                logger.warning("empty persona turn at exchange %d; dropping last user turn", i)
                history.pop()
                break
            history.append(Msg(role=Role.assistant, content=assistant_text))

        system = Msg(role=Role.system, content=render_system_prompt(self._persona))
        return DialogueExample(messages=[system, *history])


async def generate_dataset(
    llm: LLMAdapter,
    persona: Persona,
    *,
    num_dialogues: int = 10,
    num_exchanges: int = 4,
    scenarios: Sequence[str] | None = None,
) -> list[DialogueExample]:
    """Generate `num_dialogues` dialogues, cycling `scenarios` for variety."""
    gen = DialogueGenerator(llm, persona)
    scenario_pool = list(scenarios) if scenarios is not None else list(DEFAULT_SCENARIOS)
    out: list[DialogueExample] = []
    for i in range(num_dialogues):
        scenario = scenario_pool[i % len(scenario_pool)] if scenario_pool else None
        example = await gen.generate(num_exchanges=num_exchanges, scenario=scenario)
        # Skip degenerate dialogues (a turn failed mid-way) rather than emit unusable data.
        if len(example.messages) >= 3:
            out.append(example)
        else:
            logger.warning("dialogue %d produced too few turns; skipped", i)
    return out

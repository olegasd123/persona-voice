"""Build the LLM message list for a persona.

`render_system_prompt` augments the authored `system_prompt` with behavior directives
(turn style) so a single YAML knob actually changes how the model talks. `build_messages`
assembles system + prior history + the new user turn.
"""

from __future__ import annotations

from ..models import Msg, Persona, Role, TurnStyle

_TURN_STYLE_DIRECTIVE = {
    TurnStyle.concise: "Keep replies short and to the point — usually one to three sentences.",
    TurnStyle.balanced: "Keep replies conversational and appropriately sized for the moment.",
    TurnStyle.verbose: "Give thorough, detailed replies when the topic warrants it.",
}


def render_system_prompt(persona: Persona) -> str:
    """The authored system prompt plus derived behavior directives."""
    parts = [persona.system_prompt.strip()]

    directive = _TURN_STYLE_DIRECTIVE.get(persona.behavior.turn_style)
    if directive:
        parts.append(directive)

    # This is a spoken assistant: nudge the model away from markdown/formatting that
    # would read poorly through TTS.
    parts.append(
        "You are speaking out loud in a live voice conversation. "
        "Respond in natural spoken language without markdown, lists, or emoji."
    )
    return "\n\n".join(parts)


def build_messages(
    persona: Persona,
    history: list[Msg] | None = None,
    user_input: str | None = None,
) -> list[Msg]:
    """Assemble system + history (+ optional new user turn) into the message list."""
    messages: list[Msg] = [Msg(role=Role.system, content=render_system_prompt(persona))]
    if history:
        messages.extend(history)
    if user_input is not None:
        messages.append(Msg(role=Role.user, content=user_input))
    return messages

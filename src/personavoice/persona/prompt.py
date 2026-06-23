"""Build the LLM message list for a persona.

`render_system_prompt` augments the authored `system_prompt` with behavior directives
(turn style) so a single YAML knob actually changes how the model talks. `build_messages`
assembles system + prior history + the new user turn.
"""

from __future__ import annotations

from ..models import CEFRLevel, Demeanor, Msg, Persona, Role, SessionOptions, TurnStyle

_TURN_STYLE_DIRECTIVE = {
    TurnStyle.concise: "Keep replies short and to the point — usually one to three sentences.",
    TurnStyle.balanced: "Keep replies conversational and appropriately sized for the moment.",
    TurnStyle.verbose: "Give thorough, detailed replies when the topic warrants it.",
}

# Demeanor is a session-level overlay on the persona's authored tone. `natural` is a no-op
# (the persona as written). `rude` is deliberately *bounded*: brusque, never abusive — the
# moderation layer (Feature C) is what guarantees that bound holds on the output side.
_DEMEANOR_DIRECTIVE = {
    Demeanor.kind: (
        "Be especially warm, patient, and encouraging. Soften corrections, praise effort, "
        "and never make the user feel rushed."
    ),
    Demeanor.natural: None,  # baseline: persona as authored
    Demeanor.rude: (
        "Adopt a blunt, curt, impatient tone — terse replies, little hand-holding. Stay "
        "within bounds: never use slurs, harassment, threats, or demeaning personal attacks; "
        "be brusque, not abusive."
    ),
}

# CEFR level calibrates the language difficulty for learners. Injected only when set;
# surfaced prominently for the language_teacher persona but available to all.
_CEFR_DIRECTIVE = {
    CEFRLevel.a1: (
        "The user is a beginner (CEFR A1). Use very short, simple sentences and the most "
        "common words. Speak slowly and rephrase if needed."
    ),
    CEFRLevel.a2: (
        "The user is an elementary speaker (CEFR A2). Use short, simple sentences and "
        "everyday vocabulary; keep grammar basic and explain less common words."
    ),
    CEFRLevel.b1: (
        "The user is an intermediate speaker (CEFR B1). Use clear, straightforward language "
        "on familiar topics; introduce some new vocabulary but avoid heavy idiom."
    ),
    CEFRLevel.b2: (
        "The user is an upper-intermediate speaker (CEFR B2). Speak naturally at a normal "
        "pace; you can use a range of vocabulary and some idiom, explaining only when asked."
    ),
    CEFRLevel.c1: (
        "The user is an advanced speaker (CEFR C1). Use rich, varied, idiomatic language "
        "with little simplification."
    ),
    CEFRLevel.c2: (
        "The user is near-native (CEFR C2). Use fully natural, idiomatic language with no "
        "simplification."
    ),
}


def render_system_prompt(persona: Persona, options: SessionOptions | None = None) -> str:
    """The authored system prompt plus derived behavior and per-session directives.

    `options`, if given, layers session-level overrides (demeanor, CEFR) on top of the
    persona. The spoken-language nudge always comes last so it stays closest to generation.
    """
    parts = [persona.system_prompt.strip()]

    directive = _TURN_STYLE_DIRECTIVE.get(persona.behavior.turn_style)
    if directive:
        parts.append(directive)

    if options is not None:
        if options.demeanor and (d := _DEMEANOR_DIRECTIVE.get(options.demeanor)):
            parts.append(d)
        if options.cefr and (c := _CEFR_DIRECTIVE.get(options.cefr)):
            parts.append(c)

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
    *,
    memory_context: str | None = None,
    options: SessionOptions | None = None,
) -> list[Msg]:
    """Assemble system + history (+ optional new user turn) into the message list.

    `memory_context`, if given, is appended to the system prompt as a second system
    message so the persona recalls who it's talking to across sessions. It rides in its own
    message (not folded into the persona prompt) so it can vary per turn without rebuilding
    the persona's base prompt.

    `options` carries per-session overrides (demeanor, CEFR) into the rendered system prompt.
    """
    messages: list[Msg] = [Msg(role=Role.system, content=render_system_prompt(persona, options))]
    if memory_context and memory_context.strip():
        messages.append(Msg(role=Role.system, content=memory_context.strip()))
    if history:
        messages.extend(history)
    if user_input is not None:
        messages.append(Msg(role=Role.user, content=user_input))
    return messages

"""Distill stored conversations into persona-LoRA training data (the memory → LoRA bridge).

This turns a user's real transcripts into the in-character dataset the persona-LoRA
trainer already consumes, so logs can periodically refresh a LoRA (a manual trigger for
now). The module is a thin, *pure* converter on top of
`training/dataset.py`: it groups stored turns into per-session conversations, prepends the
persona's system prompt, coerces them into well-formed training examples, and hands back
`DialogueExample`s ready for `write_dataset` / `train.py`.

It is **strictly opt-in**: `distill_user` refuses unless the user granted the separate
`allow_training` consent (recording consent alone is not enough). Manual trigger only —
nothing here runs a trainer; it produces the dataset a human then feeds to `personavoice-train`.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..models import Msg, Role
from ..training.dataset import DatasetError, DialogueExample, validate_example
from .store import MemoryStore, MemoryStoreError, MemoryTurn


def _coerce_session(turns: list[MemoryTurn]) -> list[Msg]:
    """Coerce one session's raw turns into a clean alternating user→…→assistant body.

    Real transcripts aren't guaranteed to alternate (barge-in fragments, a dangling user
    turn with no reply). We merge consecutive same-role turns, drop a leading assistant turn
    and a trailing user turn, so the result satisfies `validate_example`.
    """
    msgs: list[Msg] = []
    for t in turns:
        content = t.content.strip()
        if not content:
            continue
        if msgs and msgs[-1].role == t.role:
            msgs[-1] = Msg(role=t.role, content=f"{msgs[-1].content} {content}")
        else:
            msgs.append(Msg(role=t.role, content=content))
    while msgs and msgs[0].role is not Role.user:
        msgs.pop(0)
    while msgs and msgs[-1].role is not Role.assistant:
        msgs.pop()
    return msgs


def _group_sessions(turns: list[MemoryTurn]) -> list[list[MemoryTurn]]:
    """Group turns by session id, preserving first-seen session order and turn order."""
    groups: dict[str, list[MemoryTurn]] = {}
    for t in turns:
        groups.setdefault(t.session_id, []).append(t)
    return list(groups.values())


def transcripts_to_examples(
    turns: list[MemoryTurn],
    *,
    system_prompts: Mapping[str, str] | None = None,
    min_exchanges: int = 1,
) -> list[DialogueExample]:
    """Build training examples from stored turns, one per session (pure).

    `system_prompts` maps a persona id to the system prompt to prepend (typically the
    persona's `render_system_prompt`); a session whose persona is absent gets none. Sessions
    with fewer than `min_exchanges` complete user→assistant exchanges are dropped.
    """
    examples: list[DialogueExample] = []
    for session in _group_sessions(turns):
        body = _coerce_session(session)
        if len(body) < 2 * max(1, min_exchanges):
            continue
        persona_id = session[0].persona_id
        prompt = (system_prompts or {}).get(persona_id)
        messages = [Msg(role=Role.system, content=prompt), *body] if prompt else body
        example = DialogueExample(messages=messages)
        try:
            validate_example(example)
        except DatasetError:
            continue  # skip a session we couldn't coerce into a valid example
        examples.append(example)
    return examples


def distill_user(
    store: MemoryStore,
    user_id: str,
    *,
    system_prompts: Mapping[str, str] | None = None,
    persona_id: str | None = None,
    min_exchanges: int = 1,
) -> list[DialogueExample]:
    """Distill a user's transcripts into training examples — **requires training opt-in**.

    Raises `MemoryStoreError` unless the user granted `allow_training` consent.
    """
    consent = store.get_consent(user_id)
    if not consent.allow_training:
        raise MemoryStoreError(
            f"user {user_id!r} has not opted in to training use of their data "
            "(set allow_training); refusing to distill"
        )
    turns = store.read_turns(user_id, persona_id=persona_id)
    return transcripts_to_examples(
        turns, system_prompts=system_prompts, min_exchanges=min_exchanges
    )

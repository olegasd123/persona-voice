"""Distill transcripts -> persona-LoRA training data (memory -> LoRA bridge)."""

from __future__ import annotations

from pathlib import Path

import pytest

from personavoice.memory import MemoryStore, MemoryStoreError, MemoryTurn, transcripts_to_examples
from personavoice.memory.distill import distill_user
from personavoice.models import Role
from personavoice.training.dataset import validate_example


def _turn(
    content: str, role: Role, *, session: str = "s1", persona: str = "companion"
) -> MemoryTurn:
    return MemoryTurn(session_id=session, persona_id=persona, role=role, content=content)


def test_basic_session_becomes_valid_example() -> None:
    turns = [
        _turn("hello there", Role.user),
        _turn("hi, how are you?", Role.assistant),
        _turn("good thanks", Role.user),
        _turn("glad to hear it", Role.assistant),
    ]
    examples = transcripts_to_examples(turns, system_prompts={"companion": "You are warm."})
    assert len(examples) == 1
    ex = examples[0]
    validate_example(ex)  # no raise
    assert ex.messages[0].role is Role.system
    assert ex.messages[0].content == "You are warm."
    assert ex.messages[-1].role is Role.assistant


def test_drops_leading_assistant_and_trailing_user() -> None:
    turns = [
        _turn("(opening line from assistant)", Role.assistant),
        _turn("hi", Role.user),
        _turn("hello", Role.assistant),
        _turn("a dangling question", Role.user),
    ]
    examples = transcripts_to_examples(turns)
    assert len(examples) == 1
    roles = [m.role for m in examples[0].messages]
    assert roles == [Role.user, Role.assistant]  # leading assistant + trailing user removed


def test_merges_consecutive_same_role() -> None:
    turns = [
        _turn("part one", Role.user),
        _turn("part two", Role.user),
        _turn("a reply", Role.assistant),
    ]
    examples = transcripts_to_examples(turns)
    assert examples[0].messages[0].content == "part one part two"


def test_min_exchanges_filter() -> None:
    turns = [_turn("hi", Role.user), _turn("hello", Role.assistant)]
    assert transcripts_to_examples(turns, min_exchanges=1)  # one exchange ok
    assert transcripts_to_examples(turns, min_exchanges=2) == []  # needs two


def test_one_example_per_session() -> None:
    turns = [
        _turn("a", Role.user, session="s1"),
        _turn("b", Role.assistant, session="s1"),
        _turn("c", Role.user, session="s2"),
        _turn("d", Role.assistant, session="s2"),
    ]
    assert len(transcripts_to_examples(turns)) == 2


def test_distill_user_requires_training_optin(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.set_consent("alice", granted=True, allow_training=False)
    store.record_turn("alice", _turn("hi", Role.user))
    store.record_turn("alice", _turn("hello", Role.assistant))
    with pytest.raises(MemoryStoreError, match="opted in to training"):
        distill_user(store, "alice")


def test_distill_user_with_optin(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.set_consent("alice", granted=True, allow_training=True)
    store.record_turn("alice", _turn("hi", Role.user))
    store.record_turn("alice", _turn("hello", Role.assistant))
    examples = distill_user(store, "alice", system_prompts={"companion": "be warm"})
    assert len(examples) == 1
    assert examples[0].messages[0].content == "be warm"

"""Semantic-endpointing completion gate (Feature G).

Pure transcript-fragment classification — no LiveKit, no models — mirroring `test_endpointing`.
The agent-side hold/merge/grace wiring is covered in `test_agent.py`.
"""

from __future__ import annotations

import pytest

from personavoice.orchestrator.completion import (
    TRAILING_PHRASES,
    assess_completion,
    endpointing_grace_s,
    is_complete,
    join_fragments,
    semantic_endpointing_enabled,
)


@pytest.mark.parametrize(
    "text",
    [
        "I think it's fine.",  # terminal period
        "Are you sure?",  # question mark
        "Stop!",  # exclamation
        "yes",  # bare content word
        "the meeting is at noon",  # ends on a content word, no punctuation
        "I gave it to her",  # object pronoun is a valid ender (not held)
        "what are you waiting for",  # stranded preposition we deliberately don't hold
        "",  # nothing to hold
        "   ",  # whitespace only
        "!!!",  # punctuation only
    ],
)
def test_complete_fragments(text: str) -> None:
    assert is_complete(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "I went to the store and",  # trailing conjunction
        "it's broken because",  # trailing subordinator
        "I'm thinking of",  # trailing preposition
        "I want to",  # trailing infinitive "to"
        "can you pass me the",  # trailing article
        "it's your",  # trailing possessive determiner
        "it was, um",  # trailing filler
        "um",  # bare filler
        "uh",
        "I think...",  # trailing-off ellipsis (STT's "..."), NOT a finished sentence
        "I think…",  # unicode ellipsis form
        "Well, I suppose so…",  # trails off
    ],
)
def test_incomplete_fragments(text: str) -> None:
    assert is_complete(text) is False


def test_trailing_ellipsis_holds_then_merges() -> None:
    # Regression: STT renders a mid-thought pause as "I think..."; it must be held, then merged
    # with the continuation into one finished turn (the live-log failure that motivated the fix).
    assert assess_completion("I think...").reason == "trailing-ellipsis"
    merged = join_fragments("I think...", "I'm fine.")
    assert merged == "I think... I'm fine."
    assert is_complete(merged) is True  # single terminal period ends the merged sentence


def test_single_dot_is_terminal_but_double_is_ellipsis() -> None:
    assert is_complete("I'm fine.") is True  # one dot = finished
    assert is_complete("hold on..") is False  # two+ dots = trailing off


# --- hedging lead-in phrases ("I guess", "I suppose", …) -------------------------------


@pytest.mark.parametrize("phrase", sorted(TRAILING_PHRASES))
def test_every_lead_in_phrase_holds_bare(phrase: str) -> None:
    # Each catalogued phrase, on its own with no terminal punctuation, reads as unfinished.
    assert is_complete(phrase) is False
    assert assess_completion(phrase).reason == "trailing-lead-in"


@pytest.mark.parametrize(
    "text",
    [
        "I guess",  # the requested cases
        "I suppose",
        "Hmm, I mean",  # phrase as the trailing words of a longer utterance
        "Well, the thing is",
        "I guess,",  # a trailing comma still holds
        "I GUESS",  # case-insensitive
        "I'm thinking",  # contraction form
    ],
)
def test_lead_in_phrase_in_context_holds(text: str) -> None:
    assert is_complete(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "I guess.",  # a confident terminal period is trusted as finished
        "I suppose!",
        "What do you think",  # ends "you think", not the "i think" lead-in
        "I don't know",  # deliberately excluded — a common complete answer
        "that's what I think it is",  # phrase is embedded, not trailing
        "the thing is broken",  # "the thing is" only holds when it's the tail
    ],
)
def test_non_lead_in_stays_complete(text: str) -> None:
    assert is_complete(text) is True


def test_requested_phrases_are_catalogued() -> None:
    # The two the user named explicitly, plus a representative spread of the +20 added.
    assert {"i guess", "i suppose", "i think", "i mean", "you know", "the thing is"} <= (
        TRAILING_PHRASES
    )


def test_verdict_reasons() -> None:
    assert assess_completion("").reason == "empty"
    assert assess_completion("done.").reason == "terminal-punctuation"
    assert assess_completion("I went and").reason == "trailing-function-word"
    assert assess_completion("it was um").reason == "trailing-filler"
    assert assess_completion("that sounds good").reason == "default"


def test_classification_is_case_insensitive() -> None:
    assert is_complete("I went to the store AND") is False
    assert is_complete("It's broken Because") is False


def test_trailing_punctuation_after_function_word_still_holds() -> None:
    # A comma is not terminal, so "and," still reads as a dangling conjunction.
    assert is_complete("I went to the store and,") is False
    # But a period after it wins (terminal punctuation short-circuits).
    assert is_complete("I went to the store and.") is True


@pytest.mark.parametrize(
    ("held", "new", "expected"),
    [
        ("I think", "it's fine", "I think it's fine"),
        (None, "hello", "hello"),
        ("", "hello", "hello"),
        ("  I think  ", "  it's fine ", "I think it's fine"),
        ("held", "", "held"),
    ],
)
def test_join_fragments(held: str | None, new: str, expected: str) -> None:
    assert join_fragments(held, new) == expected


def test_semantic_endpointing_enabled_default_off() -> None:
    assert semantic_endpointing_enabled({}) is False
    assert semantic_endpointing_enabled({"PERSONAVOICE_SEMANTIC_ENDPOINTING": ""}) is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_semantic_endpointing_enabled_truthy(value: str) -> None:
    assert semantic_endpointing_enabled({"PERSONAVOICE_SEMANTIC_ENDPOINTING": value}) is True


def test_grace_default_and_override() -> None:
    assert endpointing_grace_s({}) == 1.5  # 1500 ms default
    assert endpointing_grace_s({"PERSONAVOICE_ENDPOINTING_GRACE_MS": "800"}) == 0.8


@pytest.mark.parametrize("value", ["", "soon", "-200"])
def test_grace_invalid_falls_back_to_default(value: str) -> None:
    assert endpointing_grace_s({"PERSONAVOICE_ENDPOINTING_GRACE_MS": value}) == 1.5

"""The token -> sentence chunker that feeds streaming TTS (M3)."""

from __future__ import annotations

from collections.abc import AsyncIterator

from personavoice.orchestrator.chunker import (
    SentenceAggregator,
    chunk_kwargs_from_env,
    stream_sentences,
)


def _feed(text: str, *, chunk: int = 1, **kw: object) -> list[str]:
    """Push `text` through the aggregator `chunk` chars at a time, then flush."""
    agg = SentenceAggregator(**kw)  # type: ignore[arg-type]
    out: list[str] = []
    for i in range(0, len(text), chunk):
        out.extend(agg.push(text[i : i + chunk]))
    tail = agg.flush()
    if tail:
        out.append(tail)
    return out


def test_splits_on_sentence_punctuation() -> None:
    assert _feed("Hello there. How are you? I'm fine!") == [
        "Hello there.",
        "How are you?",
        "I'm fine!",
    ]


def test_boundary_needs_trailing_whitespace() -> None:
    # A decimal must not split even though it contains a period.
    assert _feed("Pi is about 3.14 today. Right?") == ["Pi is about 3.14 today.", "Right?"]


def test_does_not_break_on_common_abbreviations() -> None:
    assert _feed("Dr. Smith arrived. He waited.") == ["Dr. Smith arrived.", "He waited."]


def test_groups_multi_char_punctuation_run() -> None:
    assert _feed("Really?! Yes... okay.") == ["Really?!", "Yes...", "okay."]


def test_newline_forces_a_break() -> None:
    assert _feed("first line\nsecond line") == ["first line", "second line"]


def test_trailing_text_without_punctuation_is_flushed() -> None:
    assert _feed("an unfinished thought") == ["an unfinished thought"]


def test_chunking_is_independent_of_token_boundaries() -> None:
    # The same text split one char at a time vs in big chunks yields identical sentences.
    text = "One. Two! Three? Four."
    assert _feed(text, chunk=1) == _feed(text, chunk=100) == ["One.", "Two!", "Three?", "Four."]


def test_max_chunk_chars_forces_a_break_on_run_ons() -> None:
    long = "word " * 20  # 100 chars, no sentence punctuation
    out = _feed(long, chunk=5, max_chunk_chars=30)
    assert len(out) > 1
    assert all(len(c) <= 30 for c in out)
    # No words are split across chunks.
    assert " ".join(out).split() == long.split()


def test_keeps_closing_quote_with_the_sentence() -> None:
    assert _feed('She said "go." Then left.') == ['She said "go."', "Then left."]


async def test_stream_sentences_async_wrapper() -> None:
    async def tokens() -> AsyncIterator[str]:
        for t in ["Hel", "lo. ", "Wor", "ld!"]:
            yield t

    got = [c async for c in stream_sentences(tokens())]
    assert got == ["Hello.", "World!"]


# --- M10: first-chunk early flush + env tuning -----------------------------------------


def test_first_chunk_breaks_early_on_clause_boundary() -> None:
    # With first_chunk_chars set, the long opening sentence is broken at the first available
    # clause boundary once it's long enough — so audio can start before the sentence ends.
    text = "Well, that is a genuinely great question to ask, and I will answer it fully now."
    out = _feed(text, chunk=1, first_chunk_chars=20)
    assert len(out) >= 2
    assert out[0] == "Well, that is a genuinely great question to ask,"
    # Nothing is lost across the break.
    assert " ".join(out) == text


def test_first_chunk_shortcut_only_affects_the_first_chunk() -> None:
    # After the first chunk, later sentences use normal full-sentence boundaries.
    text = "Sure, I can help with that, no problem. What would you like to do next, exactly?"
    out = _feed(text, chunk=1, first_chunk_chars=15)
    assert out[0] == "Sure, I can help with that,"
    assert out[1] == "no problem."
    assert out[2] == "What would you like to do next, exactly?"


def test_first_chunk_chars_does_not_split_short_first_sentence() -> None:
    # A first sentence that ends before first_chunk_chars still flushes on its period.
    assert _feed("Hi there. How are you?", chunk=1, first_chunk_chars=200) == [
        "Hi there.",
        "How are you?",
    ]


def test_first_chunk_chars_waits_when_no_clause_boundary_yet() -> None:
    # No clause punctuation: it must not break mid-word; falls through to the run-on valve.
    out = _feed("one two three four five six", chunk=1, first_chunk_chars=10)
    assert " ".join(out).split() == ["one", "two", "three", "four", "five", "six"]
    assert all(c == c.strip() for c in out)


def test_chunk_kwargs_from_env_defaults() -> None:
    assert chunk_kwargs_from_env({}) == {"max_chunk_chars": 240, "first_chunk_chars": None}


def test_chunk_kwargs_from_env_parses_overrides() -> None:
    env = {
        "PERSONAVOICE_TTS_MAX_CHUNK_CHARS": "180",
        "PERSONAVOICE_TTS_FIRST_CHUNK_CHARS": "60",
    }
    assert chunk_kwargs_from_env(env) == {"max_chunk_chars": 180, "first_chunk_chars": 60}


def test_chunk_kwargs_from_env_ignores_invalid() -> None:
    env = {"PERSONAVOICE_TTS_MAX_CHUNK_CHARS": "nope", "PERSONAVOICE_TTS_FIRST_CHUNK_CHARS": "0"}
    assert chunk_kwargs_from_env(env) == {"max_chunk_chars": 240, "first_chunk_chars": None}


def test_invalid_first_chunk_chars_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        SentenceAggregator(first_chunk_chars=0)

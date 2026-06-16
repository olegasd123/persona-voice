"""The token -> sentence chunker that feeds streaming TTS (M3)."""

from __future__ import annotations

from collections.abc import AsyncIterator

from personavoice.orchestrator.chunker import SentenceAggregator, stream_sentences


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

"""Per-utterance emotion tags: the pure tag grammar + extractors."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from personavoice.emotion import (
    _ALIASES,
    EMOTIONS,
    canonical_emotion,
    dynamic_emotion_enabled,
    split_emotion_hint,
    split_leading_emotion,
)


async def _aiter(items: list[str]) -> AsyncIterator[str]:
    for it in items:
        yield it


async def _drain(it: AsyncIterator[str]) -> str:
    return "".join([tok async for tok in it])


# --- canonical_emotion -----------------------------------------------------------------


def test_canonical_emotion_recognizes_vocab_aliases_and_rejects_unknown() -> None:
    assert canonical_emotion("excited") == "excited"
    assert canonical_emotion("  EXCITED ") == "excited"  # case/space-insensitive
    assert canonical_emotion("cheerful") == "happy"  # alias folds to canonical
    assert canonical_emotion("nope") is None
    assert canonical_emotion("") is None


# --- split_emotion_hint (full reply) ---------------------------------------------------


def test_split_hint_strips_recognized_tag() -> None:
    emotion, body = split_emotion_hint("[excited] That's wonderful news!")
    assert emotion == "excited"
    assert body == "That's wonderful news!"


def test_split_hint_folds_alias_and_is_case_insensitive() -> None:
    emotion, body = split_emotion_hint("[Cheerful]  hi there")
    assert emotion == "happy"
    assert body == "hi there"


def test_split_hint_no_space_after_tag() -> None:
    emotion, body = split_emotion_hint("[calm]breathe")
    assert emotion == "calm"
    assert body == "breathe"


def test_split_hint_tolerates_leading_whitespace_before_tag() -> None:
    emotion, body = split_emotion_hint("  [sad] oh no")
    assert emotion == "sad"
    assert body == "oh no"


def test_split_hint_leaves_unrecognized_bracket_intact() -> None:
    # A literal bracket that isn't an emotion (citation, stage note) must not be eaten.
    for text in ("[1] per the source", "[pause] then continue", "no tag here"):
        emotion, body = split_emotion_hint(text)
        assert emotion is None
        assert body == text


def test_split_hint_unclosed_bracket_is_literal() -> None:
    emotion, body = split_emotion_hint("[excited and then some text with no close")
    assert emotion is None
    assert body == "[excited and then some text with no close"


# --- split_leading_emotion (streaming) -------------------------------------------------


async def test_streaming_tag_split_across_tokens() -> None:
    emotion, rest = await split_leading_emotion(_aiter(["[", "exc", "ited] ", "Hello ", "there."]))
    assert emotion == "excited"
    assert await _drain(rest) == "Hello there."


async def test_streaming_no_tag_passes_through_untouched() -> None:
    emotion, rest = await split_leading_emotion(_aiter(["Hello ", "there. ", "How are you?"]))
    assert emotion is None
    assert await _drain(rest) == "Hello there. How are you?"


async def test_streaming_unrecognized_bracket_preserved() -> None:
    emotion, rest = await split_leading_emotion(_aiter(["[1] ", "According to..."]))
    assert emotion is None
    assert await _drain(rest) == "[1] According to..."


async def test_streaming_tag_in_single_token() -> None:
    emotion, rest = await split_leading_emotion(_aiter(["[warm] welcome back!"]))
    assert emotion == "warm"
    assert await _drain(rest) == "welcome back!"


async def test_streaming_unclosed_bracket_beyond_window_is_literal() -> None:
    long_run = "[" + "x" * 40
    emotion, rest = await split_leading_emotion(_aiter([long_run, " tail"]))
    assert emotion is None
    assert await _drain(rest) == long_run + " tail"


async def test_streaming_tag_only_then_stream_ends() -> None:
    emotion, rest = await split_leading_emotion(_aiter(["[happy]"]))
    assert emotion == "happy"
    assert await _drain(rest) == ""


async def test_streaming_empty_stream() -> None:
    emotion, rest = await split_leading_emotion(_aiter([]))
    assert emotion is None
    assert await _drain(rest) == ""


# --- env toggle ------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_dynamic_emotion_enabled_truthy(value: str) -> None:
    assert dynamic_emotion_enabled({"PERSONAVOICE_DYNAMIC_EMOTION": value}) is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "nope"])
def test_dynamic_emotion_enabled_falsy(value: str) -> None:
    assert dynamic_emotion_enabled({"PERSONAVOICE_DYNAMIC_EMOTION": value}) is False


def test_dynamic_emotion_disabled_when_unset() -> None:
    assert dynamic_emotion_enabled({}) is False


# --- coupling: every emotion the model may emit must map on the cloning backend --------


def test_every_emotion_has_a_chatterbox_exaggeration() -> None:
    # The dynamic tags are only useful if Chatterbox can render them; guard that the vocabulary
    # (canonical words + the canonical targets of every alias) stays covered by its map.
    from personavoice.adapters.tts.chatterbox import _EMOTION_EXAGGERATION

    for emotion in (*EMOTIONS, *_ALIASES.values()):
        assert emotion in _EMOTION_EXAGGERATION, f"no exaggeration mapping for {emotion!r}"

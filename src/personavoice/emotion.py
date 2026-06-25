"""Per-utterance emotion hints (Feature F — dynamic emotion / prosody).

Emotion is otherwise *static*: a persona's voice resolves one `VoiceRef.emotion` once and every
reply is spoken at that intensity. This module lets the LLM color each reply individually by
prefixing it with a tiny tag — ``[excited] That's wonderful news!`` — which is parsed off and
mapped onto the voice for that turn only, then stripped before TTS (and before it reaches the
transcript or history). On a cloning backend Chatterbox renders it via its ``exaggeration`` knob;
a preset-only backend (Kokoro) just ignores `VoiceRef.emotion`, so the tag degrades to a no-op.

The whole feature is opt-in via ``PERSONAVOICE_DYNAMIC_EMOTION`` (default off): when it's off the
directive is never added to the prompt and nothing is parsed, so behavior is unchanged.

Everything here is pure (no I/O, no ML deps), so the tag grammar is fully unit-testable offline —
the same discipline as `chunker.py` / `endpointing.py`. Two entry points cover the two pipelines:

- `split_emotion_hint(text)` — full-reply parse for the turn-based `Pipeline`.
- `split_leading_emotion(tokens)` — streaming parse that consumes only the leading window of a
  token stream and hands back the remainder, so the streaming pipeline never has to buffer the
  whole reply to learn its emotion.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping

# Canonical emotions the model may tag a reply with. Deliberately small and concrete so the LLM
# picks reliably, and each maps to a TTS prosody level (see `chatterbox._EMOTION_EXAGGERATION`);
# an unmapped word or a preset-only backend falls back to the voice's default intensity.
EMOTIONS: tuple[str, ...] = (
    "neutral",
    "happy",
    "excited",
    "calm",
    "sad",
    "serious",
    "warm",
    "curious",
    "sympathetic",
)

# Natural synonyms the model might reach for, folded onto a canonical word so they still resolve.
_ALIASES: dict[str, str] = {
    "cheerful": "happy",
    "joyful": "happy",
    "glad": "happy",
    "enthusiastic": "excited",
    "thrilled": "excited",
    "relaxed": "calm",
    "gentle": "calm",
    "sorry": "sympathetic",
    "empathetic": "sympathetic",
    "concerned": "sympathetic",
    "friendly": "warm",
    "kind": "warm",
    "inquisitive": "curious",
}

# How far into the reply we'll scan for a closing ``]`` before deciding the opening ``[`` was just
# literal text. Any real tag is far shorter (longest canonical word + brackets < 16 chars).
_MAX_TAG_SCAN = 32

# The prompt directive that teaches the model the tag grammar. Added to the system prompt only
# when dynamic emotion is enabled; the vocabulary mirrors `EMOTIONS`.
EMOTION_DIRECTIVE = (
    "You can color your delivery with emotion. When it fits, begin your reply with a single "
    "emotion tag in square brackets — one of: " + ", ".join(EMOTIONS) + " — then a space, then "
    'your reply. For example: "[excited] That\'s wonderful news!" Put the tag only at the very '
    "start, never elsewhere, and simply omit it (or use [neutral]) when no particular emotion "
    "applies. The tag is stripped before your words are spoken, so it is never read aloud."
)


def canonical_emotion(word: str) -> str | None:
    """Map a raw tag word to its canonical emotion, or None if it isn't one we recognize."""
    key = word.strip().lower()
    if key in EMOTIONS:
        return key
    return _ALIASES.get(key)


def _match_leading_tag(text: str) -> tuple[str, str] | None:
    """If `text` opens with a recognized ``[emotion]`` tag, return (canonical, body_after_tag).

    `text` is expected already left-stripped. Returns None when there's no leading bracket, no
    closing bracket, or the bracketed word isn't a known emotion — in which case the caller leaves
    the text untouched (so a literal ``[1]`` citation or ``[pause]`` stage note is never eaten).
    """
    if not text.startswith("["):
        return None
    close = text.find("]")
    if close == -1:
        return None
    emotion = canonical_emotion(text[1:close])
    if emotion is None:
        return None
    return emotion, text[close + 1 :].lstrip()


def split_emotion_hint(text: str) -> tuple[str | None, str]:
    """Split a leading emotion tag off a full reply string.

    Returns (emotion, body): `emotion` is the canonical word when the reply opens with a known
    ``[emotion]`` tag (else None), and `body` is the reply with that tag removed (unchanged when
    there is no tag). Used by the turn-based pipeline, which has the whole reply in hand.
    """
    match = _match_leading_tag(text.lstrip())
    if match is None:
        return None, text
    return match


async def _chain(head: str, tail: AsyncIterator[str]) -> AsyncIterator[str]:
    """Yield `head` (if non-empty) then everything remaining in `tail`."""
    if head:
        yield head
    async for tok in tail:
        yield tok


async def _empty() -> AsyncIterator[str]:
    """An exhausted token stream (for the case where the source ended within the scan window)."""
    return
    yield  # pragma: no cover - unreachable, makes this an async generator


async def split_leading_emotion(
    tokens: AsyncIterator[str],
) -> tuple[str | None, AsyncIterator[str]]:
    """Pull a leading emotion tag off a *token stream*, buffering only its leading window.

    Returns (emotion, remaining) where `emotion` is the canonical word if the reply opens with a
    known ``[emotion]`` tag (else None) and `remaining` is an async iterator yielding the reply
    text with that tag stripped. Only the few leading tokens needed to decide are buffered; the
    rest of `tokens` is resumed in place (the source generator is suspended, not closed), so this
    adds negligible latency before the first sentence reaches TTS.
    """
    buf = ""
    async for tok in tokens:
        buf += tok
        head = buf.lstrip()
        if head and not head.startswith("["):
            return None, _chain(buf, tokens)  # plainly no tag — pass everything through untouched
        if "]" in head:
            match = _match_leading_tag(head)
            if match is None:
                return None, _chain(buf, tokens)  # bracketed but not an emotion — leave as-is
            emotion, body = match
            return emotion, _chain(body, tokens)
        if len(head) >= _MAX_TAG_SCAN:
            return None, _chain(buf, tokens)  # an unclosed '[' run this long is just literal text
        # else: still ambiguous (e.g. just "[" or "[exc") — pull another token and decide later
    # The stream ended inside the leading window: decide on what we have.
    match = _match_leading_tag(buf.lstrip())
    if match is None:
        return None, _chain(buf, _empty())
    emotion, body = match
    return emotion, _chain(body, _empty())


def dynamic_emotion_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Whether per-utterance emotion is on, from ``PERSONAVOICE_DYNAMIC_EMOTION`` (default off).

    Off by default so existing deployments are unchanged; set it to a truthy value
    (``1``/``true``/``yes``/``on``) to have the model tag replies and the pipelines apply them.
    """
    env = os.environ if env is None else env
    return (env.get("PERSONAVOICE_DYNAMIC_EMOTION") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

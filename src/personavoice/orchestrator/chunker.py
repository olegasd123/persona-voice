"""Token → sentence chunking for streaming TTS (M3).

Streaming the LLM's reply into TTS sentence-by-sentence is what keeps perceived latency
low: the first sentence can be *spoken* while the rest of the reply is still being
generated. This module turns a stream of LLM tokens into a stream of speakable chunks.

`SentenceAggregator` is pure (no I/O, no ML deps) so the boundary logic is fully
unit-testable offline; `stream_sentences` is the thin async-generator wrapper the pipeline
uses. A chunk is flushed when a sentence-ending run (``. ! ? …``) is followed by whitespace
(so a trailing decimal like ``3.14`` or an abbreviation like ``Dr.`` doesn't break early),
or on a newline, or when a run-on passage exceeds ``max_chunk_chars``.

**Latency tuning (M10).** The single biggest perceived-latency lever is how soon the *first*
chunk is voiced. ``first_chunk_chars`` lets the first chunk break early at a clause boundary
(``, ; : —``) once it reaches that length, rather than waiting for a full sentence — trading a
slightly less natural first boundary for faster time-to-first-audio. Later chunks keep using
full sentence boundaries up to ``max_chunk_chars``.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

# Sentence-ending punctuation and the closers (quotes/brackets) that may trail it.
_END = ".!?…"
_CLOSERS = "\"')]}”’»"  # noqa: RUF001 - typographic closing quotes are intentional
# Clause punctuation the first chunk may break on early (to cut time-to-first-audio).
_CLAUSE = ",;:—–"  # noqa: RUF001 - en/em dashes are intentional clause separators
# Common abbreviations whose trailing dot must NOT end a sentence.
_ABBREV = frozenset(
    {
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.",
        "vs.", "etc.", "e.g.", "i.e.", "no.", "fig.", "approx.", "dept.",
    }
)


class SentenceAggregator:
    """Accumulates streamed text and emits complete, speakable chunks.

    Feed tokens with :meth:`push` (returns any chunks completed by that token) and call
    :meth:`flush` once the stream ends to drain the remaining buffer.
    """

    def __init__(self, *, max_chunk_chars: int = 240, first_chunk_chars: int | None = None) -> None:
        if max_chunk_chars < 1:
            raise ValueError("max_chunk_chars must be >= 1")
        if first_chunk_chars is not None and first_chunk_chars < 1:
            raise ValueError("first_chunk_chars must be >= 1")
        self.max_chunk_chars = max_chunk_chars
        self.first_chunk_chars = first_chunk_chars
        self._buf = ""
        self._emitted = 0  # how many chunks have been emitted (for the first-chunk shortcut)

    def push(self, text: str) -> list[str]:
        """Append `text` and return any sentence chunks it completed (possibly empty)."""
        self._buf += text
        out: list[str] = []

        while True:
            idx = self._next_break()
            if idx is None:
                break
            self._take(idx, out)

        # First-chunk shortcut (M10 latency): if nothing has been voiced yet, break the first
        # chunk early at a clause boundary once it's long enough, rather than waiting for a
        # full sentence. This is what cuts time-to-first-audio on a long opening sentence.
        clause = self._first_clause_break()
        if clause is not None:
            self._take(clause, out)

        # Safety valve: force a break on a run-on passage with no punctuation so a single
        # giant sentence can't stall the first audio chunk. Break at the last space below
        # the limit to avoid splitting a word. The first (not-yet-voiced) chunk uses the
        # tighter `first_chunk_chars` cap when set.
        while len(self._buf) > self._cap():
            cap = self._cap()
            cut = self._buf.rfind(" ", 0, cap)
            if cut <= 0:
                cut = cap
            self._take(cut, out)

        return out

    def _take(self, idx: int, out: list[str]) -> None:
        """Slice `[:idx]` off the buffer as the next chunk (if non-empty) and record it."""
        chunk = self._buf[:idx].strip()
        self._buf = self._buf[idx:].lstrip()
        if chunk:
            out.append(chunk)
            self._emitted += 1

    def _cap(self) -> int:
        """Run-on character cap for the chunk currently being built."""
        if self._emitted == 0 and self.first_chunk_chars is not None:
            return self.first_chunk_chars
        return self.max_chunk_chars

    def _first_clause_break(self) -> int | None:
        """Index past the first usable clause boundary for an early first chunk, or None.

        Only fires before the first chunk is voiced, and only for the *first* clause boundary
        whose preceding text is already at least ``first_chunk_chars`` long — so we don't emit
        a tiny fragment like ``"Well,"`` but do break before a long opening sentence ends.
        """
        if self.first_chunk_chars is None or self._emitted > 0:
            return None
        for i in range(len(self._buf) - 1):
            if (
                i + 1 >= self.first_chunk_chars
                and self._buf[i] in _CLAUSE
                and self._buf[i + 1].isspace()
            ):
                return i + 1
        return None

    def flush(self) -> str | None:
        """Return whatever text remains (stripped) and clear the buffer, or None."""
        tail = self._buf.strip()
        self._buf = ""
        return tail or None

    def _next_break(self) -> int | None:
        """Index just past the earliest valid sentence boundary in the buffer, or None.

        Returns None when no boundary is *provably* complete yet — i.e. a trailing
        ``.``/closer with nothing after it could still be a decimal or mid-abbreviation,
        so we wait for more lookahead rather than break early.
        """
        buf = self._buf
        n = len(buf)
        i = 0
        while i < n:
            c = buf[i]
            if c == "\n":
                return i + 1
            if c in _END:
                # Consume the full run of end-chars ("?!", "...") then any closers.
                j = i
                while j < n and buf[j] in _END:
                    j += 1
                k = j
                while k < n and buf[k] in _CLOSERS:
                    k += 1
                if k >= n:
                    return None  # boundary at end of buffer — need more lookahead
                if buf[k].isspace():
                    if self._is_abbrev(i, j):
                        i = j  # e.g. "Dr." — skip this dot, keep scanning
                        continue
                    return k
                # Punctuation not followed by whitespace (e.g. "3.14", "U.S.") — not a break.
                i = j
                continue
            i += 1
        return None

    def _is_abbrev(self, dot: int, end: int) -> bool:
        """True if the single dot at `dot` closes a known abbreviation (e.g. ``Dr.``)."""
        if end != dot + 1 or self._buf[dot] != ".":
            return False
        s = dot
        while s > 0 and (self._buf[s - 1].isalpha() or self._buf[s - 1] == "."):
            s -= 1
        return self._buf[s : dot + 1].lower() in _ABBREV


async def stream_sentences(
    tokens: AsyncIterator[str],
    *,
    max_chunk_chars: int = 240,
    first_chunk_chars: int | None = None,
) -> AsyncIterator[str]:
    """Yield speakable sentence chunks from a stream of LLM tokens."""
    agg = SentenceAggregator(max_chunk_chars=max_chunk_chars, first_chunk_chars=first_chunk_chars)
    async for tok in tokens:
        for chunk in agg.push(tok):
            yield chunk
    tail = agg.flush()
    if tail:
        yield tail


# --------------------------------------------------------------------------------------
# Tuning resolved from the environment (M10)
# --------------------------------------------------------------------------------------


def _int_env(env: dict[str, str], name: str, default: int | None) -> int | None:
    """Parse an optional positive-int env var, falling back to `default` on unset/invalid."""
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 1 else default


def chunk_kwargs_from_env(env: dict[str, str] | None = None) -> dict[str, int | None]:
    """Resolve `stream_sentences` chunk-sizing kwargs from the environment (M10 latency).

    ``PERSONAVOICE_TTS_MAX_CHUNK_CHARS`` caps a run-on sentence; ``…_FIRST_CHUNK_CHARS``
    enables the early first-chunk clause break that cuts time-to-first-audio. Unset/invalid
    values fall back to the defaults (no early first chunk), so behavior is unchanged unless
    an operator opts in.
    """
    env = os.environ if env is None else env
    return {
        "max_chunk_chars": _int_env(env, "PERSONAVOICE_TTS_MAX_CHUNK_CHARS", 240),
        "first_chunk_chars": _int_env(env, "PERSONAVOICE_TTS_FIRST_CHUNK_CHARS", None),
    }

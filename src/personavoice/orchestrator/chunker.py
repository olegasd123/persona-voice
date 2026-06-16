"""Token → sentence chunking for streaming TTS (M3).

Streaming the LLM's reply into TTS sentence-by-sentence is what keeps perceived latency
low: the first sentence can be *spoken* while the rest of the reply is still being
generated. This module turns a stream of LLM tokens into a stream of speakable chunks.

`SentenceAggregator` is pure (no I/O, no ML deps) so the boundary logic is fully
unit-testable offline; `stream_sentences` is the thin async-generator wrapper the pipeline
uses. A chunk is flushed when a sentence-ending run (``. ! ? …``) is followed by whitespace
(so a trailing decimal like ``3.14`` or an abbreviation like ``Dr.`` doesn't break early),
or on a newline, or when a run-on passage exceeds ``max_chunk_chars``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

# Sentence-ending punctuation and the closers (quotes/brackets) that may trail it.
_END = ".!?…"
_CLOSERS = "\"')]}”’»"  # noqa: RUF001 - typographic closing quotes are intentional
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

    def __init__(self, *, max_chunk_chars: int = 240) -> None:
        if max_chunk_chars < 1:
            raise ValueError("max_chunk_chars must be >= 1")
        self.max_chunk_chars = max_chunk_chars
        self._buf = ""

    def push(self, text: str) -> list[str]:
        """Append `text` and return any sentence chunks it completed (possibly empty)."""
        self._buf += text
        out: list[str] = []

        while True:
            idx = self._next_break()
            if idx is None:
                break
            chunk = self._buf[:idx].strip()
            self._buf = self._buf[idx:].lstrip()
            if chunk:
                out.append(chunk)

        # Safety valve: force a break on a run-on passage with no punctuation so a single
        # giant sentence can't stall the first audio chunk. Break at the last space below
        # the limit to avoid splitting a word.
        while len(self._buf) > self.max_chunk_chars:
            cut = self._buf.rfind(" ", 0, self.max_chunk_chars)
            if cut <= 0:
                cut = self.max_chunk_chars
            chunk = self._buf[:cut].strip()
            self._buf = self._buf[cut:].lstrip()
            if chunk:
                out.append(chunk)

        return out

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
    tokens: AsyncIterator[str], *, max_chunk_chars: int = 240
) -> AsyncIterator[str]:
    """Yield speakable sentence chunks from a stream of LLM tokens."""
    agg = SentenceAggregator(max_chunk_chars=max_chunk_chars)
    async for tok in tokens:
        for chunk in agg.push(tok):
            yield chunk
    tail = agg.flush()
    if tail:
        yield tail

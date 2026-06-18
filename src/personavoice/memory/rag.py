"""Retrieval of relevant past turns + assembly of the memory context block.

What actually gets injected into the prompt each turn is a short block built from two
sources: the distilled `UserProfile` (always — summary + durable facts) and the top-k most
*relevant* prior turns retrieved against the current user message. The profile gives
continuity; retrieval surfaces the specific past exchange that bears on what was just said.

Retrieval is pluggable behind `MemoryRetriever`:
  - `KeywordRetriever` — pure token-overlap (cosine over term counts), **zero deps**, the
    default. Good enough to find "the time we talked about my sister's wedding".
  - `EmbeddingRetriever` — semantic similarity via sentence-transformers (lazy, the
    `memory-embeddings` extra). Strictly an upgrade; the system is fully functional without it.

The assembled block is plain spoken-friendly text (no markdown) so it reads cleanly when it
rides in the system prompt of a TTS-bound assistant.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol, runtime_checkable

from .profile import UserProfile
from .store import MemoryTurn

_TOKEN_RE = re.compile(r"[a-z0-9']+")
# Tiny stoplist: words too common to carry retrieval signal. Kept short on purpose — this is
# lexical recall, not search-engine IR.
_STOPWORD_TEXT = (
    "a an and are as at be but by do for from had has have i if in is it its me my no not "
    "of on or our so that the their them they this to up us was we were what when who will "
    "with you your"
)
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


@runtime_checkable
class MemoryRetriever(Protocol):
    """Ranks candidate turns against a query, most-relevant first."""

    def rank(
        self, query: str, turns: list[MemoryTurn], *, k: int
    ) -> list[tuple[float, MemoryTurn]]: ...


class KeywordRetriever:
    """Pure lexical retrieval: cosine similarity over term-frequency vectors."""

    name = "keyword"

    def rank(
        self, query: str, turns: list[MemoryTurn], *, k: int
    ) -> list[tuple[float, MemoryTurn]]:
        q = Counter(_tokens(query))
        if not q or k <= 0:
            return []
        q_norm = math.sqrt(sum(v * v for v in q.values()))
        scored: list[tuple[float, MemoryTurn]] = []
        for turn in turns:
            d = Counter(_tokens(turn.content))
            if not d:
                continue
            dot = sum(q[t] * d[t] for t in q if t in d)
            if dot == 0:
                continue
            d_norm = math.sqrt(sum(v * v for v in d.values()))
            scored.append((dot / (q_norm * d_norm), turn))
        # Stable: highest score first, ties keep transcript order (oldest first).
        scored.sort(key=lambda st: st[0], reverse=True)
        return scored[:k]


class EmbeddingRetriever:
    """Semantic retrieval via sentence-transformers (lazy; the `memory-embeddings` extra)."""

    name = "embedding"

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model
        self._model: object | None = None

    def _ensure_model(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise RuntimeError(
                    "sentence-transformers isn't installed; install the embeddings extra: "
                    "`pip install -e '.[memory-embeddings]'` (or use KeywordRetriever)"
                ) from exc
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def rank(
        self, query: str, turns: list[MemoryTurn], *, k: int
    ) -> list[tuple[float, MemoryTurn]]:
        if not turns or k <= 0 or not query.strip():
            return []
        model = self._ensure_model()
        import numpy as np

        texts = [t.content for t in turns]
        embeddings = model.encode([query, *texts], normalize_embeddings=True)  # type: ignore[attr-defined]
        q_vec = np.asarray(embeddings[0])
        sims = [(float(np.dot(q_vec, np.asarray(embeddings[i + 1]))), turns[i]) for i in range(len(turns))]
        sims.sort(key=lambda st: st[0], reverse=True)
        return sims[:k]


def recall_context(
    profile: UserProfile | None,
    turns: list[MemoryTurn],
    query: str,
    *,
    retriever: MemoryRetriever | None = None,
    k: int = 4,
    min_score: float = 0.0,
) -> str:
    """Assemble the spoken-friendly memory block to inject, or "" if there's nothing to add.

    Combines the profile (summary + facts) with the top-k relevant prior turns retrieved for
    `query`. Turns from the *current* session aren't excluded here — the caller passes the
    history it wants searched (typically prior sessions, since the live history already
    covers the current one).
    """
    sections: list[str] = []

    if profile is not None and not profile.is_empty():
        if profile.summary.strip():
            sections.append(f"What you know about this user:\n{profile.summary.strip()}")
        if profile.facts:
            facts = "\n".join(f"- {f.text}" for f in profile.facts)
            sections.append(f"Remembered facts:\n{facts}")

    retriever = retriever or KeywordRetriever()
    ranked = [(s, t) for s, t in retriever.rank(query, turns, k=k) if s > min_score]
    if ranked:
        lines = [f"- {t.role.value.capitalize()}: {t.content.strip()}" for _s, t in ranked]
        sections.append("Relevant moments from earlier conversations:\n" + "\n".join(lines))

    if not sections:
        return ""
    header = (
        "The following is your memory of this user from past conversations. "
        "Use it naturally; do not recite it verbatim or mention that you have notes."
    )
    return header + "\n\n" + "\n\n".join(sections)

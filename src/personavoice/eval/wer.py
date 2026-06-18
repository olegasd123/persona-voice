"""Word Error Rate for STT quality (M10 automated eval).

WER = (substitutions + deletions + insertions) / reference_words, computed from the
word-level Levenshtein alignment between a reference transcript and the STT hypothesis.
Text is normalized first (lowercase, punctuation stripped, whitespace collapsed) so trivial
formatting differences don't inflate the score.

Everything here is pure (no audio, no models): `scripts/eval_stt.py` runs the backend STT
over a manifest of `{wav, reference}` clips and feeds the hypotheses through `wer`.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

# Keep word-internal apostrophes/hyphens (don't, state-of-the-art); drop other punctuation.
_PUNCT_RE = re.compile(r"[^\w\s'-]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, strip surrounding punctuation, and collapse whitespace for fair matching."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Normalize then split into comparison tokens (words)."""
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


@dataclass(frozen=True)
class WordErrors:
    """Edit-operation counts from aligning a hypothesis to a reference."""

    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def total(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        """WER. An empty reference yields 0.0 if the hypothesis is also empty, else 1.0."""
        if self.reference_words == 0:
            return 0.0 if self.insertions == 0 else 1.0
        return self.total / self.reference_words


def word_errors(reference: str, hypothesis: str) -> WordErrors:
    """Levenshtein-align `hypothesis` to `reference` (word level) and count edits.

    Classic DP with backtracking. Costs: match=0, substitute/insert/delete=1.
    """
    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
    n, m = len(ref), len(hyp)

    # dp[i][j] = min edits to turn ref[:i] into hyp[:j].
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],  # substitution
                    dp[i - 1][j],  # deletion (ref word dropped)
                    dp[i][j - 1],  # insertion (extra hyp word)
                )

    # Backtrack to attribute each edit to S / D / I.
    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return WordErrors(substitutions=subs, deletions=dels, insertions=ins, reference_words=n)


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate for a single (reference, hypothesis) pair."""
    return word_errors(reference, hypothesis).rate


def corpus_wer(pairs: Sequence[tuple[str, str]]) -> float:
    """Aggregate WER across `(reference, hypothesis)` pairs (edits / total reference words).

    Pooled — not the mean of per-utterance rates — so long utterances weigh proportionally,
    which is the standard way WER is reported over a test set.
    """
    total_edits = 0
    total_words = 0
    for reference, hypothesis in pairs:
        e = word_errors(reference, hypothesis)
        total_edits += e.total
        total_words += e.reference_words
    if total_words == 0:
        return 0.0
    return total_edits / total_words

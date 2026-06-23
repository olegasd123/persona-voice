"""A dependency-free rule/keyword guard — the default real `Moderator`.

This is intentionally small and offline: no model, no network. It covers the two cases the
plan calls out as load-bearing for safe real-user exposure:

- **Crisis on input** — phrases signalling self-harm / acute distress short-circuit the turn
  to a calm, resource-pointing reply (and flag `crisis` so the caller can disable `rude`).
- **Bounded output** — the `rude` demeanor is allowed to be brusque but not abusive; an
  assistant reply with demeaning second-person attacks or threats is replaced with a neutral
  line, which is what *guarantees* "brusque, not abusive".

A keyword guard trades recall for zero dependencies; swap in an LLM/hosted `Moderator` for
better coverage. The phrase sets are deliberately conservative to avoid false positives on
ordinary conversation. Slur/hate lists are left to a configured guard rather than shipped
inline; pass extra patterns via `extra_output_patterns` if a deployment needs them.
"""

from __future__ import annotations

import re

from ..models import Demeanor
from .base import ModerationCategory, ModerationResult

# Acute-distress / self-harm signals in the *user's* utterance. Conservative phrasings so a
# casual "this is killing me" doesn't trip it.
_CRISIS_PATTERNS = [
    r"\bkill (?:myself|me)\b",
    r"\b(?:want|going|wanting) to die\b",
    r"\bend (?:my|it all) life\b",
    r"\bend my life\b",
    r"\bsuicid(?:e|al)\b",
    r"\b(?:hurt|harm|cut|cutting) myself\b",
    r"\bself[- ]harm\b",
    r"\bno reason to live\b",
]

# Demeaning second-person attacks / threats in the *assistant's* reply (the `rude` bound).
# Anchored to "you" so the guard targets attacks on the user, not neutral usage.
_HARASSMENT_PATTERNS = [
    r"\byou(?:'re| are)\s+(?:an?\s+)?(?:idiot|stupid|moron|worthless|pathetic|useless|dumb)\b",
    r"\byou\s+(?:idiot|moron|loser)\b",
    r"\bshut up\b",
]
_THREAT_PATTERNS = [
    r"\b(?:i'?ll|i will|going to|gonna)\s+(?:kill|hurt|destroy|beat)\s+you\b",
]

# A calm, resource-pointing reply for the crisis branch. Generic and region-neutral; a real
# deployment should localize the resources.
_CRISIS_REPLY = (
    "I'm really sorry you're feeling this way, and I'm glad you told me. I can't provide "
    "crisis care, but you deserve support from someone who can. If you might act on these "
    "feelings, please contact your local emergency number, or reach a crisis line — in the "
    "US you can call or text 988. Would you like to keep talking for a bit?"
)

# Neutral fallback when the assistant's own reply breaches the bound.
_SOFTENED_REPLY = "Let's keep this respectful. Here's a straight answer without the edge."


class KeywordModerator:
    """Rule-based `Moderator`: crisis detection on input, abuse/threat bound on output."""

    def __init__(self, *, extra_output_patterns: list[str] | None = None) -> None:
        self._crisis = [re.compile(p, re.IGNORECASE) for p in _CRISIS_PATTERNS]
        self._harassment = [re.compile(p, re.IGNORECASE) for p in _HARASSMENT_PATTERNS]
        self._threat = [re.compile(p, re.IGNORECASE) for p in _THREAT_PATTERNS]
        self._extra = [re.compile(p, re.IGNORECASE) for p in (extra_output_patterns or [])]

    async def check_input(self, text: str) -> ModerationResult:
        if any(p.search(text) for p in self._crisis):
            return ModerationResult(
                flagged=True,
                categories=[ModerationCategory.crisis],
                replacement=_CRISIS_REPLY,
                crisis=True,
            )
        return ModerationResult.ok()

    async def check_output(
        self, text: str, *, demeanor: Demeanor | None = None
    ) -> ModerationResult:
        categories: list[ModerationCategory] = []
        if any(p.search(text) for p in self._harassment):
            categories.append(ModerationCategory.harassment)
        if any(p.search(text) for p in self._threat):
            categories.append(ModerationCategory.threat)
        if self._extra and any(p.search(text) for p in self._extra):
            categories.append(ModerationCategory.hate)
        if not categories:
            return ModerationResult.ok()
        return ModerationResult(flagged=True, categories=categories, replacement=_SOFTENED_REPLY)

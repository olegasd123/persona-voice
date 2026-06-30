"""A dependency-free rule/keyword guard — the default real `Moderator`.

This is intentionally small and offline: no model, no network. It covers the two cases the
plan calls out as load-bearing for safe real-user exposure:

- **Crisis on input** — phrases signalling self-harm / acute distress short-circuit the turn
  to a calm, resource-pointing reply (and flag `crisis` so the caller can disable `rude`).
- **Bounded output** — the `rude` demeanor is allowed to be brusque but not abusive; an
  assistant reply with demeaning second-person attacks, threats, or anything that *encourages*
  self-harm is replaced with a safe line, which is what *guarantees* "brusque, not abusive".

A keyword guard trades recall for zero dependencies; swap in an LLM/hosted `Moderator` for
better coverage. The phrase sets are deliberately conservative to avoid false positives on
ordinary conversation. Slur/hate lists are left to a configured guard rather than shipped
inline; pass extra patterns via `extra_output_patterns` if a deployment needs them.
"""

from __future__ import annotations

import re

from ..models import Demeanor
from .base import ModerationCategory, ModerationResult

# Acute-distress / self-harm signals in the *user's* utterance. The bare verb forms stay
# conservative so a casual "this homework is killing me" doesn't trip ("kill me", not "killing
# me"), but the set is broadened to the common passive-ideation phrasings — for the crisis
# branch recall matters more than precision, since a false positive only shows a calm,
# supportive message.
_CRISIS_PATTERNS = [
    r"\bkill (?:myself|me)\b",
    r"\bkilling myself\b",
    r"\b(?:want|wanna|going|wanting) to die\b",
    r"\bdon'?t want to (?:live|be alive|be here|exist)\b",
    r"\bend(?:ing|ed)? (?:my|my own) life\b",
    r"\bend(?:ing|ed)? it all\b",
    r"\btak(?:e|ing) my (?:own )?life\b",
    r"\bsuicid(?:e|al)\b",
    r"\b(?:hurt|harm|cut|cutting|harming|hurting) myself\b",
    r"\bself[- ]harm\b",
    r"\bno reason to live\b",
    r"\bnothing to live for\b",
    r"\bbetter off without me\b",
    r"\bwish i (?:was|were) dead\b",
]

# Assistant output that *encourages* self-harm / suicide — the model must never do this, so it
# is bounded regardless of demeanor and gets the caring, resource-pointing replacement.
_SELF_HARM_OUTPUT_PATTERNS = [
    r"\bkill yourself\b",
    r"\byou should (?:kill yourself|end (?:it all|your life)|die|harm yourself|hurt yourself)\b",
    r"\b(?:just|go ahead and|go)\s+(?:kill yourself|end it all|die)\b",
    r"\byou(?:'?d| would)?\s+(?:be )?better off dead\b",
    r"\bnobody would miss you\b",
    r"\bthe world would be better (?:off )?without you\b",
]

# Demeaning second-person attacks / threats in the *assistant's* reply (the `rude` bound).
# Anchored to "you" so the guard targets attacks on the user, not neutral usage.
_HARASSMENT_PATTERNS = [
    r"\byou(?:'re| are)\s+(?:an?\s+)?"
    r"(?:idiot|stupid|moron|worthless|pathetic|useless|dumb|disgrace|failure|loser|trash|"
    r"garbage|embarrassment|joke|disgusting|repulsive|incompetent)\b",
    r"\byou\s+(?:idiot|moron|loser|imbecile|fool)\b",
    r"\bshut up\b",
    r"\bi\s+(?:hate|despise|loathe) you\b",
    r"\bnobody (?:likes|loves|wants) you\b",
    r"\byou disgust me\b",
    r"\byou should be ashamed\b",
]
_THREAT_PATTERNS = [
    r"\b(?:i'?ll|i will|i'?m going to|im going to|going to|gonna)\s+"
    r"(?:kill|hurt|harm|destroy|beat)\s+you\b",
    r"\byou'?ll regret (?:this|that|it)\b",
    r"\bwatch your back\b",
    r"\bi know where you live\b",
    r"\bmake you pay\b",
]

# A calm, resource-pointing reply for the crisis branch. Generic and region-neutral; a real
# deployment should localize the resources.
_CRISIS_REPLY = (
    "I'm really sorry you're feeling this way, and I'm glad you told me. I can't provide "
    "crisis care, but you deserve support from someone who can. If you might act on these "
    "feelings, please contact your local emergency number, or reach a crisis line — in the "
    "US you can call or text 988. Would you like to keep talking for a bit?"
)

# Neutral fallback when the assistant's own reply breaches the `rude` bound (abuse / threats).
_SOFTENED_REPLY = "Let's keep this respectful. Here's a straight answer without the edge."

# Caring fallback when the assistant's own reply *encourages* self-harm — a graver breach than
# rudeness, so it points to support rather than just de-escalating.
_UNSAFE_OUTPUT_REPLY = (
    "I can't help with that, and I'd never want to. If you're going through something, you "
    "deserve real support — please reach out to someone you trust, or a crisis line; in the "
    "US you can call or text 988."
)


class KeywordModerator:
    """Rule-based `Moderator`: crisis detection on input, abuse/threat bound on output."""

    def __init__(self, *, extra_output_patterns: list[str] | None = None) -> None:
        self._crisis = [re.compile(p, re.IGNORECASE) for p in _CRISIS_PATTERNS]
        self._self_harm = [re.compile(p, re.IGNORECASE) for p in _SELF_HARM_OUTPUT_PATTERNS]
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
        if any(p.search(text) for p in self._self_harm):
            categories.append(ModerationCategory.self_harm)
        if any(p.search(text) for p in self._harassment):
            categories.append(ModerationCategory.harassment)
        if any(p.search(text) for p in self._threat):
            categories.append(ModerationCategory.threat)
        if self._extra and any(p.search(text) for p in self._extra):
            categories.append(ModerationCategory.hate)
        if not categories:
            return ModerationResult.ok()
        # Self-harm encouragement is the graver breach — point to support; everything else
        # (abuse / threats / configured patterns) gets the neutral de-escalation line.
        replacement = (
            _UNSAFE_OUTPUT_REPLY if ModerationCategory.self_harm in categories else _SOFTENED_REPLY
        )
        return ModerationResult(flagged=True, categories=categories, replacement=replacement)

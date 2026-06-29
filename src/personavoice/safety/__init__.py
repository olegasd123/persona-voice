"""Safety / moderation layer.

A thin, pluggable guard around the LLM turn (`Moderator`), with a no-op default so the seam
is present everywhere without changing behavior until a real guard is configured. The shipped
real guard is a dependency-free rule/keyword guard (`KeywordModerator`).

    base.py    — Moderator protocol, ModerationResult, NoopModerator (default)
    keyword.py — KeywordModerator: crisis-on-input + abuse/threat bound on output
"""

from __future__ import annotations

import os

from .base import (
    ModerationCategory,
    ModerationResult,
    Moderator,
    NoopModerator,
)
from .keyword import KeywordModerator

__all__ = [
    "KeywordModerator",
    "ModerationCategory",
    "ModerationResult",
    "Moderator",
    "NoopModerator",
    "build_moderator",
    "moderator_from_env",
]


def build_moderator(kind: str | None) -> Moderator:
    """Construct a moderator by name. Unknown / empty / "none" → the no-op guard."""
    name = (kind or "").strip().lower()
    if name in ("keyword", "rule", "rules"):
        return KeywordModerator()
    return NoopModerator()


def moderator_from_env() -> Moderator:
    """Resolve the moderator from `PERSONAVOICE_MODERATION` (default: off / no-op).

    Off by default so existing deployments are unchanged; set `PERSONAVOICE_MODERATION=keyword`
    to enable the rule guard (recommended once the `rude` demeanor is exposed to real users).
    """
    return build_moderator(os.getenv("PERSONAVOICE_MODERATION"))

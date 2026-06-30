"""Safety / moderation layer.

A thin, pluggable guard around the LLM turn (`Moderator`). The seam is present everywhere; the
dependency-free rule/keyword guard (`KeywordModerator`) is wired in by default and can be turned
off with `PERSONAVOICE_MODERATION=none`. `NoopModerator` is the explicit opt-out.

    base.py    — Moderator protocol, ModerationResult, NoopModerator (the opt-out)
    keyword.py — KeywordModerator: crisis-on-input + self-harm/abuse/threat bound on output
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


# Explicit opt-out values that turn moderation off; anything else yields the rule guard.
_DISABLED = frozenset({"none", "off", "no", "false", "0", "noop", "disabled"})


def build_moderator(kind: str | None) -> Moderator:
    """Construct a moderator by name.

    An explicit opt-out (`none`/`off`/`disabled`/…) yields the no-op guard; everything else —
    unset, empty, the rule names, or an unrecognized value — yields the dependency-free
    `KeywordModerator`. Defaulting *to* the guard (fail-safe on) is deliberate: a typo'd value
    or a forgotten env var leaves the safety bound enabled rather than silently off.
    """
    name = (kind or "").strip().lower()
    if name in _DISABLED:
        return NoopModerator()
    return KeywordModerator()


def moderator_from_env() -> Moderator:
    """Resolve the moderator from `PERSONAVOICE_MODERATION` (default: the rule guard).

    The dependency-free keyword guard is **on by default** — crisis-on-input, a bound on
    self-harm-encouraging output, and the bounded `rude` cap — so real-user exposure is guarded
    out of the box. Set `PERSONAVOICE_MODERATION=none` (or `off`) to disable it.
    """
    return build_moderator(os.getenv("PERSONAVOICE_MODERATION"))

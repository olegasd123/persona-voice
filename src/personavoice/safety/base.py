"""The moderation seam: a pluggable guard around the LLM turn.

A user-facing companion — and especially the bounded `rude` demeanor — needs a guard that
keeps input handling and output bounded without coupling it into the cascade stages.
`Moderator` is the interface; concrete guards (a rule/keyword guard, or later an LLM /
hosted guard) implement it. The default wired into the pipelines is `NoopModerator`, so the
seam is present everywhere but behavior is unchanged until a real guard is configured.

Two hook points:

- **input**  — classify the user's transcript; a flagged category (crisis especially) can
  short-circuit to a safe reply instead of running the normal turn.
- **output** — check the assembled reply before TTS; if it breaches bounds (the `rude` cap,
  harassment, self-harm encouragement) replace it with a safe line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from ..models import Demeanor


class ModerationCategory(StrEnum):
    """Why content was flagged. `crisis` drives a dedicated calm/resource-pointing branch."""

    crisis = "crisis"  # self-harm / acute distress (the user is at risk)
    self_harm = "self_harm"  # content encouraging self-harm
    harassment = "harassment"  # demeaning personal attacks / abuse
    threat = "threat"  # threats of violence
    hate = "hate"  # slurs / hateful content


@dataclass
class ModerationResult:
    """The verdict for one piece of text.

    `flagged` is True when the guard wants the caller to act. `replacement`, when set, is the
    safe text the caller should use *instead* of the original (a calm crisis reply on input,
    or a softened line on output). `crisis` marks the acute-distress branch so the caller can,
    e.g., disable `rude` for the rest of the session.
    """

    flagged: bool = False
    categories: list[ModerationCategory] = field(default_factory=list)
    replacement: str | None = None
    crisis: bool = False

    @classmethod
    def ok(cls) -> ModerationResult:
        """The clean verdict — nothing to do."""
        return cls()


@runtime_checkable
class Moderator(Protocol):
    """A pluggable input/output guard. Async so an LLM/hosted guard can implement it."""

    async def check_input(self, text: str) -> ModerationResult:
        """Classify a user utterance before it drives a turn."""
        ...

    async def check_output(
        self, text: str, *, demeanor: Demeanor | None = None
    ) -> ModerationResult:
        """Check an assembled assistant reply before it is voiced."""
        ...


class NoopModerator:
    """The default: never flags anything, so the cascade behaves exactly as before."""

    async def check_input(self, text: str) -> ModerationResult:
        return ModerationResult.ok()

    async def check_output(
        self, text: str, *, demeanor: Demeanor | None = None
    ) -> ModerationResult:
        return ModerationResult.ok()

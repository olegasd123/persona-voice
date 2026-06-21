"""Per-turn latency metrics for the live agent (observability).

The streaming pipeline already records latency landmarks into a `StreamMetrics`; this turns
one of those (plus a little turn context) into a flat, loggable `TurnMetrics` record so each
live turn's STT / time-to-first-token / time-to-first-audio / total is visible in the logs.
Kept free of any orchestrator import (it only duck-types the `StreamMetrics` fields) so it has
no heavy dependencies and is unit-testable on its own.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Protocol


class _StreamMetricsLike(Protocol):
    first_token: float | None
    first_audio: float | None
    total: float | None
    reply: str


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


@dataclass
class TurnMetrics:
    """One conversational turn's latency + outcome, ready to log as structured context."""

    persona: str
    user_chars: int
    reply_chars: int
    stt_s: float | None = None
    first_token_s: float | None = None
    first_audio_s: float | None = None
    total_s: float | None = None
    interrupted: bool = False
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def log(self, logger: logging.Logger) -> None:
        """Emit the record (INFO normally, WARNING when the turn errored or was barged-in)."""
        level = logging.WARNING if (self.error or self.interrupted) else logging.INFO
        logger.log(level, "turn metrics", extra={"turn": self.as_dict()})


def turn_metrics_from_stream(
    persona: str,
    user_text: str,
    metrics: _StreamMetricsLike,
    *,
    stt_s: float | None = None,
    interrupted: bool = False,
    error: str | None = None,
) -> TurnMetrics:
    """Build a `TurnMetrics` from a finished/cancelled `StreamMetrics` and turn context."""
    return TurnMetrics(
        persona=persona,
        user_chars=len(user_text),
        reply_chars=len(metrics.reply or ""),
        stt_s=_round(stt_s),
        first_token_s=_round(metrics.first_token),
        first_audio_s=_round(metrics.first_audio),
        total_s=_round(metrics.total),
        interrupted=interrupted,
        error=error,
    )

"""Observability: structured logging + per-turn latency metrics.

`logging_setup.configure_logging()` gives every entrypoint a consistent log format (plain
text by default, JSON when `PERSONAVOICE_LOG_FORMAT=json` for machine ingestion).
`metrics.TurnMetrics` is the per-turn record the live agent emits so STT/LLM/TTS latency and
barge-ins are visible in the logs. Both are pure/deterministic and unit-tested; the agent and
CLIs just call them. `prometheus` is the optional Prometheus exporter for the same per-turn
record (no-op when `prometheus_client` isn't installed) — `record_turn` feeds it and
`render_metrics` backs the token server's `/metrics` route.
"""

from __future__ import annotations

from .logging_setup import (
    TRACE,
    JsonFormatter,
    PlainFormatter,
    configure_logging,
    log_level_from_env,
)
from .metrics import TurnMetrics, turn_metrics_from_stream
from .prometheus import (
    metrics_enabled,
    read_sessions,
    record_rejected_session,
    record_turn,
    render_metrics,
    set_sessions,
)

__all__ = [
    "TRACE",
    "JsonFormatter",
    "PlainFormatter",
    "TurnMetrics",
    "configure_logging",
    "log_level_from_env",
    "metrics_enabled",
    "read_sessions",
    "record_rejected_session",
    "record_turn",
    "render_metrics",
    "set_sessions",
    "turn_metrics_from_stream",
]

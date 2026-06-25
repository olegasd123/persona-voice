"""Observability: structured logging + per-turn latency metrics.

`logging_setup.configure_logging()` gives every entrypoint a consistent log format (plain
text by default, JSON when `PERSONAVOICE_LOG_FORMAT=json` for machine ingestion).
`metrics.TurnMetrics` is the per-turn record the live agent emits so STT/LLM/TTS latency and
barge-ins are visible in the logs. Both are pure/deterministic and unit-tested; the agent and
CLIs just call them.
"""

from __future__ import annotations

from .logging_setup import TRACE, JsonFormatter, configure_logging, log_level_from_env
from .metrics import TurnMetrics, turn_metrics_from_stream

__all__ = [
    "TRACE",
    "JsonFormatter",
    "TurnMetrics",
    "configure_logging",
    "log_level_from_env",
    "turn_metrics_from_stream",
]

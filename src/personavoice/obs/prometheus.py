"""Optional Prometheus exporter for per-turn latency + outcome metrics.

The live agent already builds a `TurnMetrics` per turn and *logs* it (`metrics.py`); this turns
the same record into Prometheus counters/histograms so the numbers are scrapeable, not just
greppable. `prometheus_client` is an **optional** dependency: when it isn't installed every
function here is a no-op and `/metrics` serves an empty body, so the core install and the offline
test suite never need the library.

Metric families (all prefixed `personavoice_`):

    personavoice_turns_total{persona,outcome}    counter  (outcome = ok|interrupted|error)
    personavoice_stt_seconds{persona}            histogram
    personavoice_first_token_seconds{persona}    histogram
    personavoice_first_audio_seconds{persona}    histogram
    personavoice_turn_seconds{persona}           histogram

`turns_total` doubles as the barge-in / error counters via its `outcome` label (sum over the
label = total turns; `outcome="interrupted"` = barge-ins; `outcome="error"` = failed turns).

Multi-process note: the agent worker (which records turns) and the token server (which serves
`/metrics`) are usually *separate* processes. Point both at one shared, writable directory via the
standard `PROMETHEUS_MULTIPROC_DIR` env var and `prometheus_client` collates them — each process
writes mmap files there and `render_metrics()` aggregates across all of them. Without it a process
only exports its own metrics (fine when scraping the worker directly, or in tests). The variable
must be exported *before* the process starts (it picks the value class at import time).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .metrics import TurnMetrics

logger = logging.getLogger("personavoice.metrics")

# The standard exposition content type, valid even for an empty body (the no-op path below).
_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

try:
    import prometheus_client as _prom

    AVAILABLE = True
except ImportError:  # optional dependency — every entry point degrades to a no-op
    _prom = None  # type: ignore[assignment]
    AVAILABLE = False


# Latency buckets tuned for a speech loop: sub-second time-to-first-audio is the goal, with a
# coarse tail out to ~30 s to catch a slow/cold turn. Shared by every latency histogram.
_LATENCY_BUCKETS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 30.0)

if AVAILABLE:
    _TURNS = _prom.Counter(
        "personavoice_turns_total",
        "Conversational turns by persona and outcome (ok|interrupted|error).",
        ["persona", "outcome"],
    )
    _STT = _prom.Histogram(
        "personavoice_stt_seconds",
        "Speech-to-text latency for a turn's user utterance.",
        ["persona"],
        buckets=_LATENCY_BUCKETS,
    )
    _FIRST_TOKEN = _prom.Histogram(
        "personavoice_first_token_seconds",
        "Time to the LLM's first token after the user's turn.",
        ["persona"],
        buckets=_LATENCY_BUCKETS,
    )
    _FIRST_AUDIO = _prom.Histogram(
        "personavoice_first_audio_seconds",
        "Time to the first synthesized audio chunk (time-to-first-sound).",
        ["persona"],
        buckets=_LATENCY_BUCKETS,
    )
    _TURN_TOTAL = _prom.Histogram(
        "personavoice_turn_seconds",
        "Total wall-clock time for a turn (user end-of-speech to last audio chunk).",
        ["persona"],
        buckets=_LATENCY_BUCKETS,
    )


def metrics_enabled() -> bool:
    """Whether the Prometheus client is installed (so recording/export do real work)."""
    return AVAILABLE


def record_turn(m: TurnMetrics) -> None:
    """Record one turn's latency landmarks + outcome into the Prometheus collectors.

    A no-op when `prometheus_client` isn't installed. Defensive by design: metrics must never
    crash the conversation loop, so any unexpected failure (e.g. an unwritable
    `PROMETHEUS_MULTIPROC_DIR`) is swallowed with a debug log rather than propagated — this runs
    in the agent's per-turn `finally`.
    """
    if not AVAILABLE:
        return
    try:
        outcome = "error" if m.error else "interrupted" if m.interrupted else "ok"
        _TURNS.labels(persona=m.persona, outcome=outcome).inc()
        if m.stt_s is not None:
            _STT.labels(persona=m.persona).observe(m.stt_s)
        if m.first_token_s is not None:
            _FIRST_TOKEN.labels(persona=m.persona).observe(m.first_token_s)
        if m.first_audio_s is not None:
            _FIRST_AUDIO.labels(persona=m.persona).observe(m.first_audio_s)
        if m.total_s is not None:
            _TURN_TOTAL.labels(persona=m.persona).observe(m.total_s)
    except Exception:  # pragma: no cover - metrics are best-effort, never fatal
        logger.debug("failed to record turn metrics", exc_info=True)


def render_metrics() -> tuple[bytes, str]:
    """Render the current metrics as `(payload, content_type)` for the `/metrics` route.

    Returns an empty body (still a valid, parseable exposition) when the client lib is absent. In
    multi-process mode (`PROMETHEUS_MULTIPROC_DIR` set) it aggregates every process's mmap files
    into a fresh registry; otherwise it serves this process's default registry.
    """
    if not AVAILABLE:
        return b"", _CONTENT_TYPE
    multiproc_dir = (os.environ.get("PROMETHEUS_MULTIPROC_DIR") or "").strip()
    if multiproc_dir:
        from prometheus_client import CollectorRegistry, multiprocess

        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
    else:
        registry = _prom.REGISTRY
    return _prom.generate_latest(registry), _prom.CONTENT_TYPE_LATEST

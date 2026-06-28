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
from typing import TYPE_CHECKING, Any

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
    # Concurrency / admission control (Feature I). The agent worker is the source of truth for
    # the live count (LiveKit `active_jobs`); it mirrors that here so the token server's
    # `/healthz` + `/metrics` can report load across processes. `livesum` aggregates across a
    # fleet of workers (each gauge file summed), so `active`/`max` read as fleet totals. Only the
    # worker's main process writes these, so steady state is one file per worker = exact. Like
    # every multiproc metric, an *unclean* worker restart leaves a stale file that inflates the
    # sum until `PROMETHEUS_MULTIPROC_DIR` is cleared; the gate in `agent.py` never reads this, so
    # only the reported number (not safety) is affected. NB: this never gates — it only reports.
    _SESSIONS_ACTIVE = _prom.Gauge(
        "personavoice_sessions_active",
        "Conversation sessions currently active (live count from the agent worker).",
        multiprocess_mode="livesum",
    )
    _SESSIONS_MAX = _prom.Gauge(
        "personavoice_sessions_max",
        "Configured max concurrent sessions (capacity) the worker admits.",
        multiprocess_mode="livesum",
    )
    _SESSIONS_REJECTED = _prom.Counter(
        "personavoice_sessions_rejected_total",
        "Jobs rejected by admission control because the worker was at capacity.",
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


def set_sessions(active: int, capacity: int) -> None:
    """Publish the worker's live session count + capacity (Feature I, admission control).

    The agent worker holds the authoritative count (LiveKit `active_jobs`); this mirrors it into
    the Prometheus channel so the token server can report load on `/healthz` and `/metrics`
    without coupling to the worker. Best-effort: a no-op when `prometheus_client` is absent and
    never raises (the gate in `agent.py` does not depend on this).
    """
    if not AVAILABLE:
        return
    try:
        _SESSIONS_ACTIVE.set(active)
        _SESSIONS_MAX.set(capacity)
    except Exception:  # pragma: no cover - metrics are best-effort, never fatal
        logger.debug("failed to publish session gauges", exc_info=True)


def record_rejected_session() -> None:
    """Count one admission-control rejection (a job refused because the worker was at capacity)."""
    if not AVAILABLE:
        return
    try:
        _SESSIONS_REJECTED.inc()
    except Exception:  # pragma: no cover - metrics are best-effort, never fatal
        logger.debug("failed to record rejected session", exc_info=True)


def read_sessions() -> tuple[int, int] | None:
    """Current `(active, capacity)` session counts for `/healthz`, or None when unknown.

    Reads back the gauges the worker published, aggregating the per-process mmap files when
    `PROMETHEUS_MULTIPROC_DIR` is set (worker + token server are separate processes). Returns
    None when the exporter is absent or the worker hasn't reported yet (no gauge series), so the
    route can simply omit the fields rather than show a misleading zero.
    """
    if not AVAILABLE:
        return None
    try:
        registry = _collect_registry()
        active = _gauge_total(registry, "personavoice_sessions_active")
        capacity = _gauge_total(registry, "personavoice_sessions_max")
    except Exception:  # pragma: no cover - best-effort, /healthz must still answer
        logger.debug("failed to read session gauges", exc_info=True)
        return None
    if active is None and capacity is None:
        return None
    return int(active or 0), int(capacity or 0)


def _collect_registry() -> Any:
    """The registry `/metrics` renders: the multi-process aggregate when `PROMETHEUS_MULTIPROC_DIR`
    is set (worker + token server are separate processes), else this process's default registry.
    """
    multiproc_dir = (os.environ.get("PROMETHEUS_MULTIPROC_DIR") or "").strip()
    if multiproc_dir:
        from prometheus_client import CollectorRegistry, multiprocess

        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return registry
    return _prom.REGISTRY


def _gauge_total(registry: Any, name: str) -> float | None:
    """Sum the samples of gauge `name` in `registry`, or None when the series isn't present."""
    for metric in registry.collect():
        if metric.name != name:
            continue
        samples = [s.value for s in metric.samples if s.name == name]
        return sum(samples) if samples else None
    return None


def render_metrics() -> tuple[bytes, str]:
    """Render the current metrics as `(payload, content_type)` for the `/metrics` route.

    Returns an empty body (still a valid, parseable exposition) when the client lib is absent. In
    multi-process mode (`PROMETHEUS_MULTIPROC_DIR` set) it aggregates every process's mmap files
    into a fresh registry; otherwise it serves this process's default registry.
    """
    if not AVAILABLE:
        return b"", _CONTENT_TYPE
    return _prom.generate_latest(_collect_registry()), _prom.CONTENT_TYPE_LATEST

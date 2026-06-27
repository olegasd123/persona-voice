"""The optional Prometheus exporter (`obs/prometheus.py`).

Covers both paths: the real exporter (prometheus-client is a dev dep so it's present here) and
the no-op degradation when the library is absent (simulated by flipping the module flag).
"""

from __future__ import annotations

import pytest

from personavoice.obs import TurnMetrics
from personavoice.obs import prometheus as prom


def _scrape() -> str:
    payload, content_type = prom.render_metrics()
    assert content_type.startswith("text/plain")
    return payload.decode("utf-8")


@pytest.mark.skipif(not prom.AVAILABLE, reason="prometheus-client not installed")
def test_record_turn_counts_outcomes_and_latency() -> None:
    prom.record_turn(
        TurnMetrics(
            persona="companion",
            user_chars=5,
            reply_chars=10,
            stt_s=0.2,
            first_token_s=0.3,
            first_audio_s=0.5,
            total_s=1.1,
        )
    )
    prom.record_turn(
        TurnMetrics(persona="companion", user_chars=3, reply_chars=0, interrupted=True)
    )
    prom.record_turn(
        TurnMetrics(persona="hr_interviewer", user_chars=3, reply_chars=0, error="boom")
    )

    text = _scrape()
    # The outcome label discriminates total turns / barge-ins / errors off one counter.
    assert 'personavoice_turns_total{outcome="ok",persona="companion"}' in text
    assert 'personavoice_turns_total{outcome="interrupted",persona="companion"}' in text
    assert 'personavoice_turns_total{outcome="error",persona="hr_interviewer"}' in text
    # Latency histograms emit per-persona buckets once observed.
    assert 'personavoice_stt_seconds_bucket{le="0.25",persona="companion"}' in text
    assert "personavoice_first_token_seconds_bucket" in text
    assert "personavoice_first_audio_seconds_bucket" in text
    assert "personavoice_turn_seconds_bucket" in text


@pytest.mark.skipif(not prom.AVAILABLE, reason="prometheus-client not installed")
def test_record_turn_skips_missing_latency_samples() -> None:
    # A turn that errored before any landmark has no latency values to observe — recording it must
    # still bump the counter without raising on the None fields.
    prom.record_turn(TurnMetrics(persona="solo_metric", user_chars=1, reply_chars=0, error="x"))
    text = _scrape()
    assert 'personavoice_turns_total{outcome="error",persona="solo_metric"}' in text
    # The histogram family is always registered (its TYPE line is present)...
    assert "# TYPE personavoice_turn_seconds histogram" in text
    # ...but with no total_s observed for this persona, it has no series for it.
    assert 'personavoice_turn_seconds_count{persona="solo_metric"}' not in text


def test_no_op_when_library_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate prometheus-client not being installed: record_turn does nothing and render returns
    # a valid (empty) exposition with the standard content type — never raising.
    monkeypatch.setattr(prom, "AVAILABLE", False)
    prom.record_turn(TurnMetrics(persona="companion", user_chars=1, reply_chars=1, total_s=0.5))
    payload, content_type = prom.render_metrics()
    assert payload == b""
    assert content_type == "text/plain; version=0.0.4; charset=utf-8"
    assert prom.metrics_enabled() is False


def test_record_turn_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Metrics are best-effort: a failure inside the collector must be swallowed (it runs in the
    # agent's per-turn finally), not propagated.
    if not prom.AVAILABLE:
        pytest.skip("prometheus-client not installed")

    class _Boom:
        def labels(self, **_kw: object) -> _Boom:
            raise RuntimeError("collector exploded")

    monkeypatch.setattr(prom, "_TURNS", _Boom())
    # Should not raise despite the broken collector.
    prom.record_turn(TurnMetrics(persona="companion", user_chars=1, reply_chars=1))

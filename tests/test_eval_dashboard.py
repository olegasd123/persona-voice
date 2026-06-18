"""Eval dashboard gate logic (M10)."""

from __future__ import annotations

import pytest

from personavoice.eval.dashboard import (
    CEILING,
    FLOOR,
    Gate,
    dashboard_ok,
    evaluate_gates,
    format_dashboard,
)


def test_floor_gate_passes_when_at_or_above() -> None:
    gate = Gate(metric="persona_adherence", threshold=0.8, bound=FLOOR)
    assert gate.passes(0.8) is True
    assert gate.passes(0.95) is True
    assert gate.passes(0.79) is False


def test_ceiling_gate_passes_when_at_or_below() -> None:
    gate = Gate(metric="wer", threshold=0.15, bound=CEILING)
    assert gate.passes(0.15) is True
    assert gate.passes(0.05) is True
    assert gate.passes(0.2) is False


def test_invalid_bound_rejected() -> None:
    with pytest.raises(ValueError):
        Gate(metric="x", threshold=1.0, bound="sideways")


def test_evaluate_gates_all_green() -> None:
    gates = [
        Gate("persona_adherence", 0.8, FLOOR),
        Gate("wer", 0.15, CEILING),
        Gate("e2e_audio_ms", 900, CEILING),
    ]
    metrics = {"persona_adherence": 0.92, "wer": 0.08, "e2e_audio_ms": 580}
    results = evaluate_gates(metrics, gates)
    assert dashboard_ok(results) is True
    assert all(r.present for r in results)


def test_missing_metric_fails_dashboard() -> None:
    gates = [Gate("wer", 0.15, CEILING)]
    results = evaluate_gates({}, gates)
    assert results[0].present is False
    assert results[0].passed is False
    assert dashboard_ok(results) is False


def test_format_dashboard_red_and_green() -> None:
    green = evaluate_gates({"wer": 0.1}, [Gate("wer", 0.15, CEILING)])
    assert "GREEN" in format_dashboard(green)
    red = evaluate_gates({"wer": 0.3}, [Gate("wer", 0.15, CEILING)])
    out = format_dashboard(red)
    assert "RED" in out
    assert "FAIL" in out

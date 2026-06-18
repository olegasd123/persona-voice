"""Eval dashboard: gate headline metrics against thresholds (M10 acceptance).

The M10 acceptance is "sustained multi-turn sessions within latency budget; eval dashboard
green". This module is the *green/red* logic: a small set of `Gate`s (metric + threshold +
which side is good) evaluated against a flat `{metric: value}` dict produced by the eval
runner (`scripts/run_eval.py`). Pure so the pass/fail rules are unit-tested; the runner
collects the numbers (latency budget, STT WER, persona adherence, voice MOS) and renders this.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# A gate is a floor (value must be >= threshold; higher is better, e.g. persona adherence) or
# a ceiling (value must be <= threshold; lower is better, e.g. WER / latency).
FLOOR = "floor"
CEILING = "ceiling"


@dataclass(frozen=True)
class Gate:
    """One pass/fail threshold on a named metric."""

    metric: str
    threshold: float
    bound: str  # FLOOR | CEILING
    label: str = ""

    def __post_init__(self) -> None:
        if self.bound not in (FLOOR, CEILING):
            raise ValueError(f"gate bound must be {FLOOR!r} or {CEILING!r}, got {self.bound!r}")

    def passes(self, value: float) -> bool:
        return value >= self.threshold if self.bound == FLOOR else value <= self.threshold


@dataclass(frozen=True)
class MetricResult:
    """A gate's verdict against the measured value (None = the metric wasn't produced)."""

    metric: str
    value: float | None
    threshold: float
    bound: str
    passed: bool
    label: str = ""

    @property
    def present(self) -> bool:
        return self.value is not None


def evaluate_gates(metrics: Mapping[str, float], gates: Sequence[Gate]) -> list[MetricResult]:
    """Evaluate each gate against `metrics`. A gated metric that's missing fails (can't be
    declared green if it was never measured)."""
    results: list[MetricResult] = []
    for gate in gates:
        value = metrics.get(gate.metric)
        passed = value is not None and gate.passes(value)
        results.append(
            MetricResult(
                metric=gate.metric,
                value=value,
                threshold=gate.threshold,
                bound=gate.bound,
                passed=passed,
                label=gate.label,
            )
        )
    return results


def dashboard_ok(results: Sequence[MetricResult]) -> bool:
    """True when every gated metric passed (the 'dashboard green' check)."""
    return all(r.passed for r in results)


def format_dashboard(results: Sequence[MetricResult]) -> str:
    """Render a fixed-width pass/fail table with a final GREEN/RED line."""
    lines = ["Eval dashboard", f"  {'metric':<24} {'value':>10} {'gate':>16} {'status':>8}"]
    for r in results:
        value = "—" if r.value is None else f"{r.value:.4g}"
        op = ">=" if r.bound == FLOOR else "<="
        gate = f"{op} {r.threshold:.4g}"
        status = "ok" if r.passed else ("MISSING" if not r.present else "FAIL")
        lines.append(f"  {r.metric:<24} {value:>10} {gate:>16} {status:>8}")
    lines.append(f"Result: {'GREEN' if dashboard_ok(results) else 'RED'}")
    return "\n".join(lines)

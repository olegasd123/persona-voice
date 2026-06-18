#!/usr/bin/env python3
"""Aggregate eval artifacts into one dashboard and gate them green/red (M10 acceptance).

This composes the outputs of the individual eval harnesses (it does not re-run heavy
backends) into a single pass/fail dashboard:

    # produce artifacts
    python scripts/bench_latency.py --backend cuda --wav q.wav --stream --json lat.json
    python scripts/eval_stt.py      --backend cuda --manifest m.jsonl       --json wer.json
    # gate them
    python scripts/run_eval.py --latency-json lat.json --wer-json wer.json \
        --mos-ratings mos.json --metrics persona.json

`--metrics` is a flat `{metric: value}` JSON (e.g. `{"persona_adherence": 0.92}` from the
persona eval); `--mos-ratings` is `{voice: [1..5, ...]}`. Exits non-zero when the dashboard
is RED, so it doubles as a CI gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from personavoice.eval.dashboard import (
    CEILING,
    FLOOR,
    Gate,
    dashboard_ok,
    evaluate_gates,
    format_dashboard,
)
from personavoice.eval.mos import MosSummary, summarize_mos_by_voice


def default_gates() -> list[Gate]:
    """Headline gates matching the plan's targets (override with `--gate`)."""
    return [
        Gate("e2e_audio_ms", 900.0, CEILING, "time-to-first-audio budget"),
        Gate("wer", 0.15, CEILING, "STT word-error-rate"),
        Gate("persona_adherence", 0.80, FLOOR, "persona adherence (M7)"),
        Gate("mos", 3.5, FLOOR, "voice MOS spot-check"),
    ]


def parse_gate(spec: str) -> Gate:
    """Parse a `metric:bound:threshold` CLI gate (bound = floor|ceiling)."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"gate must be metric:bound:threshold, got {spec!r}")
    metric, bound, threshold = parts
    if bound not in (FLOOR, CEILING):
        raise ValueError(f"gate bound must be {FLOOR!r} or {CEILING!r}, got {bound!r}")
    try:
        value = float(threshold)
    except ValueError as exc:
        raise ValueError(f"gate threshold must be a number, got {threshold!r}") from exc
    return Gate(metric=metric, threshold=value, bound=bound)


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def latency_ms_from_bench(payload: dict, stage: str = "e2e_audio") -> float | None:
    """Median of a bench_latency `--json` stage, in milliseconds."""
    stats = payload.get("stats", {})
    stage_stats = stats.get(stage)
    if not isinstance(stage_stats, dict) or "median" not in stage_stats:
        return None
    return float(stage_stats["median"]) * 1000.0


def merge_gates(defaults: list[Gate], overrides: list[Gate]) -> list[Gate]:
    """Apply `overrides` on top of `defaults` (same metric replaces; new metric appends)."""
    by_metric = {g.metric: g for g in defaults}
    for gate in overrides:
        by_metric[gate.metric] = gate
    return list(by_metric.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate eval artifacts into a dashboard.")
    parser.add_argument("--metrics", type=Path, default=None, help="flat {metric: value} JSON")
    parser.add_argument("--latency-json", type=Path, default=None, help="bench_latency --json file")
    parser.add_argument(
        "--latency-stage", default="e2e_audio", help="bench stage to gate (default e2e_audio)"
    )
    parser.add_argument("--wer-json", type=Path, default=None, help="eval_stt --json file")
    parser.add_argument("--mos-ratings", type=Path, default=None, help="{voice: [1..5]} JSON")
    parser.add_argument(
        "--gate",
        action="append",
        default=[],
        metavar="METRIC:BOUND:THRESHOLD",
        help="add/override a gate (repeatable), e.g. wer:ceiling:0.1",
    )
    parser.add_argument(
        "--no-default-gates", action="store_true", help="only use --gate gates, not the defaults"
    )
    parser.add_argument("--json", type=Path, default=None, help="write the dashboard as JSON")
    args = parser.parse_args(argv)

    metrics: dict[str, float] = {}
    try:
        if args.metrics is not None:
            raw = _load_json(args.metrics)
            if not isinstance(raw, dict):
                raise ValueError(f"{args.metrics}: expected a JSON object of metric -> value")
            metrics.update({str(k): float(v) for k, v in raw.items()})
        if args.latency_json is not None:
            ms = latency_ms_from_bench(_load_json(args.latency_json), args.latency_stage)  # type: ignore[arg-type]
            if ms is not None:
                metrics["e2e_audio_ms"] = ms
        if args.wer_json is not None:
            payload = _load_json(args.wer_json)
            if isinstance(payload, dict) and "corpus_wer" in payload:
                metrics["wer"] = float(payload["corpus_wer"])
        mos_summaries: list[MosSummary] = []
        if args.mos_ratings is not None:
            raw = _load_json(args.mos_ratings)
            if not isinstance(raw, dict):
                raise ValueError(f"{args.mos_ratings}: expected {{voice: [ratings]}}")
            mos_summaries = summarize_mos_by_voice({str(k): list(v) for k, v in raw.items()})
            all_ratings = [r for v in raw.values() for r in v]
            if all_ratings:
                metrics["mos"] = sum(all_ratings) / len(all_ratings)
        gates = [] if args.no_default_gates else default_gates()
        gates = merge_gates(gates, [parse_gate(g) for g in args.gate])
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    results = evaluate_gates(metrics, gates)
    print(format_dashboard(results))
    for s in mos_summaries:
        print(f"  mos[{s.voice}]: {s.mean:.2f} ± {s.ci95:.2f}  (n={s.n})")

    if args.json is not None:
        payload = {
            "metrics": metrics,
            "green": dashboard_ok(results),
            "gates": [
                {
                    "metric": r.metric,
                    "value": r.value,
                    "threshold": r.threshold,
                    "bound": r.bound,
                    "passed": r.passed,
                }
                for r in results
            ],
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote dashboard JSON -> {args.json}")

    return 0 if dashboard_ok(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Measure per-stage latency (STT / LLM / TTS) for a backend.

    python scripts/bench_latency.py --backend mac  --wav question.wav --runs 5
    python scripts/bench_latency.py --backend cuda --wav question.wav --runs 5 --json cuda.json

Runs the turn-based pipeline `--runs` times (after `--warmup` warmup turns that load model
weights) and reports min / median / mean / max per stage and total. Needs a backend extra
installed and its model servers running (LM Studio / vLLM, etc.).

These are *turn-based, full-stage* wall times — the whole reply is synthesized before TTS
stops. They are NOT the streaming "time to first audio" budget (endpointing → STT finalize
→ LLM TTFT → first TTS chunk) that M3 targets; that needs the streaming pipeline. Use this
to compare stage costs across the Mac and the 4080, and to track regressions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from personavoice.adapters.factory import build_backend
from personavoice.audio import read_wav_file
from personavoice.orchestrator.pipeline import Pipeline
from personavoice.persona.registry import PersonaRegistry
from personavoice.server.config import Settings, load_backend_config

# Stage order for stable reporting (matches Pipeline timings keys).
_STAGES = ("stt", "llm", "tts", "total")


@dataclass
class Stat:
    n: int
    min: float
    median: float
    mean: float
    max: float


def summarize(samples: list[float]) -> Stat:
    """Min/median/mean/max of one stage's per-run timings (seconds)."""
    return Stat(
        n=len(samples),
        min=min(samples),
        median=statistics.median(samples),
        mean=statistics.fmean(samples),
        max=max(samples),
    )


def summarize_runs(
    timings: list[dict[str, float]], stages: tuple[str, ...] = _STAGES
) -> dict[str, Stat]:
    """Aggregate a list of per-run `{stage: seconds}` dicts into per-stage `Stat`s."""
    if not timings:
        return {}
    stats: dict[str, Stat] = {}
    for stage in stages:
        samples = [t[stage] for t in timings if stage in t]
        if samples:
            stats[stage] = summarize(samples)
    return stats


def format_report(stats: dict[str, Stat], *, backend: str, runs: int) -> str:
    """Render a fixed-width latency table."""
    lines = [
        f"Latency — backend={backend}  runs={runs}  (turn-based, full-stage seconds)",
        f"  {'stage':<7} {'min':>8} {'median':>8} {'mean':>8} {'max':>8}",
    ]
    for stage, s in stats.items():
        lines.append(f"  {stage:<7} {s.min:>8.3f} {s.median:>8.3f} {s.mean:>8.3f} {s.max:>8.3f}")
    return "\n".join(lines)


async def _bench(settings: Settings, persona_id: str, audio: bytes, *, warmup: int, runs: int):
    backend = build_backend(load_backend_config(settings))
    persona = PersonaRegistry(settings.personas_dir).get(persona_id)
    pipeline = Pipeline(backend, persona)

    for i in range(warmup):
        print(f"warmup {i + 1}/{warmup} ...", file=sys.stderr)
        await pipeline.run_turn(audio, history=[])

    timings: list[dict[str, float]] = []
    for i in range(runs):
        result = await pipeline.run_turn(audio, history=[])
        timings.append(result.timings)
        t = result.timings
        print(
            f"run {i + 1}/{runs}: stt={t['stt']:.3f} llm={t['llm']:.3f} "
            f"tts={t['tts']:.3f} total={t['total']:.3f}",
            file=sys.stderr,
        )
    return timings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-stage latency benchmark for a backend.")
    parser.add_argument("--backend", choices=("mac", "cuda"), default=None, help="override BACKEND")
    parser.add_argument("--wav", type=Path, required=True, help="spoken-question wav to feed STT")
    parser.add_argument("--persona", default="companion", help="persona id (config/personas)")
    parser.add_argument("--runs", type=int, default=5, help="measured turns")
    parser.add_argument("--warmup", type=int, default=1, help="warmup turns (load weights)")
    parser.add_argument("--json", type=Path, default=None, help="also write the report as JSON")
    args = parser.parse_args(argv)

    if not args.wav.is_file():
        print(f"error: input wav not found: {args.wav}", file=sys.stderr)
        return 2

    try:
        settings = Settings.load(backend=args.backend)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    audio = read_wav_file(args.wav)
    try:
        timings = asyncio.run(
            _bench(settings, args.persona, audio, warmup=args.warmup, runs=args.runs)
        )
    except Exception as exc:  # surface backend/model errors without a traceback wall
        print(f"error: {exc}", file=sys.stderr)
        return 1

    stats = summarize_runs(timings)
    print()
    print(format_report(stats, backend=settings.backend, runs=args.runs))

    if args.json is not None:
        payload = {
            "backend": settings.backend,
            "persona": args.persona,
            "runs": args.runs,
            "warmup": args.warmup,
            "stats": {stage: asdict(s) for stage, s in stats.items()},
            "samples": timings,
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote JSON report -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

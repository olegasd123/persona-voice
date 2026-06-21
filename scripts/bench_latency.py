#!/usr/bin/env python3
"""Measure per-stage latency (STT / LLM / TTS) for a backend.

    python scripts/bench_latency.py --backend mac  --wav question.wav --runs 5
    python scripts/bench_latency.py --backend cuda --wav question.wav --runs 5 --json cuda.json
    python scripts/bench_latency.py --backend cuda --wav question.wav --stream   # streaming first-audio

Runs the pipeline `--runs` times (after `--warmup` warmup turns that load model weights) and
reports min / median / mean / max per stage and total. Needs a backend extra installed and
its model servers running (LM Studio / vLLM, etc.).

Default (turn-based): *full-stage* wall times — the whole reply is synthesized before TTS
stops. Good for comparing stage costs across the Mac and the GPU and tracking regressions.

`--stream`: the streaming "time to first audio" budget (STT finalize → LLM TTFT → first TTS chunk),
measured through the streaming pipeline. `e2e_audio` is the perceived latency — when the
persona starts speaking while the rest of the reply is still being generated.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from personavoice.adapters.factory import build_backend
from personavoice.audio import read_wav_file
from personavoice.orchestrator.chunker import chunk_kwargs_from_env
from personavoice.orchestrator.pipeline import Pipeline
from personavoice.orchestrator.streaming import StreamingPipeline, StreamMetrics
from personavoice.persona.registry import PersonaRegistry
from personavoice.server.config import Settings, load_backend_config, load_voice_registry

# Stage order for stable reporting (matches Pipeline timings keys).
_STAGES = ("stt", "llm", "tts", "total")
# Streaming landmarks (`--stream`): first_token / first_audio are measured from LLM start;
# e2e_audio folds in STT finalize (the real perceived latency — when speech starts); total is
# STT + speaking the whole reply sentence-by-sentence.
_STREAM_STAGES = ("stt", "first_token", "first_audio", "e2e_audio", "total")


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


def format_report(
    stats: dict[str, Stat], *, backend: str, runs: int, streaming: bool = False
) -> str:
    """Render a fixed-width latency table."""
    kind = "streaming, seconds to first audio" if streaming else "turn-based, full-stage seconds"
    lines = [
        f"Latency — backend={backend}  runs={runs}  ({kind})",
        f"  {'stage':<11} {'min':>8} {'median':>8} {'mean':>8} {'max':>8}",
    ]
    for stage, s in stats.items():
        lines.append(f"  {stage:<11} {s.min:>8.3f} {s.median:>8.3f} {s.mean:>8.3f} {s.max:>8.3f}")
    return "\n".join(lines)


def check_budget(
    stats: dict[str, Stat], *, stage: str, budget_s: float
) -> tuple[bool, float | None]:
    """Gate the median of `stage` against `budget_s` (within latency budget).

    Returns `(passed, observed_median)`. A missing stage passes vacuously (`observed=None`)
    so a turn-based run isn't failed for lacking a streaming-only stage.
    """
    s = stats.get(stage)
    if s is None:
        return True, None
    return s.median <= budget_s, s.median


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


async def _bench_stream(
    settings: Settings,
    persona_id: str,
    audio: bytes,
    *,
    warmup: int,
    runs: int,
    chunk_kwargs: dict[str, int | None] | None = None,
) -> list[dict[str, float]]:
    """Measure the streaming path: STT finalize → LLM TTFT → first TTS chunk.

    Unlike `_bench` (turn-based, whole-reply), this reports *time to first audio* — the
    perceived latency when the persona starts speaking while the rest is still generating.
    `chunk_kwargs` overrides the TTS chunk-sizing knobs so a sweep can compare settings.
    """
    backend = build_backend(load_backend_config(settings))
    persona = PersonaRegistry(settings.personas_dir).get(persona_id)
    voices = load_voice_registry(settings)

    async def _turn() -> dict[str, float]:
        t0 = time.perf_counter()
        transcript = await backend.stt.transcribe(audio)
        stt = time.perf_counter() - t0
        metrics = StreamMetrics()
        pipe = StreamingPipeline(backend, persona, voices, chunk_kwargs=chunk_kwargs)
        async for _ in pipe.stream_response(transcript.text, history=[], metrics=metrics):
            pass
        first_audio = metrics.first_audio or 0.0
        return {
            "stt": stt,
            "first_token": metrics.first_token or 0.0,
            "first_audio": first_audio,
            "e2e_audio": stt + first_audio,
            "total": stt + (metrics.total or 0.0),
        }

    for i in range(warmup):
        print(f"warmup {i + 1}/{warmup} ...", file=sys.stderr)
        await _turn()

    timings: list[dict[str, float]] = []
    for i in range(runs):
        t = await _turn()
        timings.append(t)
        print(
            f"run {i + 1}/{runs}: stt={t['stt']:.3f} first_token={t['first_token']:.3f} "
            f"first_audio={t['first_audio']:.3f} e2e_audio={t['e2e_audio']:.3f} "
            f"total={t['total']:.3f}",
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
    parser.add_argument(
        "--stream",
        action="store_true",
        help="measure the streaming time-to-first-audio instead of turn-based full-stage",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=None,
        help="TTS run-on chunk cap (sweep); default from PERSONAVOICE_TTS_MAX_CHUNK_CHARS",
    )
    parser.add_argument(
        "--first-chunk-chars",
        type=int,
        default=None,
        help="early first-chunk clause-break length (--stream only) to cut first-audio",
    )
    parser.add_argument(
        "--budget-ms",
        type=float,
        default=None,
        help="gate the median of --budget-stage against this many ms (exit 1 if exceeded)",
    )
    parser.add_argument(
        "--budget-stage",
        default=None,
        help="stage the budget applies to (default: e2e_audio for --stream, else total)",
    )
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
    stages = _STREAM_STAGES if args.stream else _STAGES
    try:
        if args.stream:
            chunk_kwargs: dict[str, int | None] | None = None
            if args.max_chunk_chars is not None or args.first_chunk_chars is not None:
                chunk_kwargs = chunk_kwargs_from_env()
                if args.max_chunk_chars is not None:
                    chunk_kwargs["max_chunk_chars"] = args.max_chunk_chars
                if args.first_chunk_chars is not None:
                    chunk_kwargs["first_chunk_chars"] = args.first_chunk_chars
            timings = asyncio.run(
                _bench_stream(
                    settings,
                    args.persona,
                    audio,
                    warmup=args.warmup,
                    runs=args.runs,
                    chunk_kwargs=chunk_kwargs,
                )
            )
        else:
            timings = asyncio.run(
                _bench(settings, args.persona, audio, warmup=args.warmup, runs=args.runs)
            )
    except Exception as exc:  # surface backend/model errors without a traceback wall
        print(f"error: {exc}", file=sys.stderr)
        return 1

    stats = summarize_runs(timings, stages)
    print()
    print(format_report(stats, backend=settings.backend, runs=args.runs, streaming=args.stream))

    budget_failed = False
    if args.budget_ms is not None:
        stage = args.budget_stage or ("e2e_audio" if args.stream else "total")
        passed, observed = check_budget(stats, stage=stage, budget_s=args.budget_ms / 1000.0)
        budget_failed = not passed
        if observed is None:
            print(f"\nbudget: stage {stage!r} not measured — skipped")
        else:
            verdict = "PASS" if passed else "FAIL"
            print(
                f"\nbudget: {stage} median {observed * 1000:.0f} ms "
                f"vs {args.budget_ms:.0f} ms -> {verdict}"
            )

    if args.json is not None:
        payload = {
            "backend": settings.backend,
            "persona": args.persona,
            "mode": "stream" if args.stream else "turn",
            "runs": args.runs,
            "warmup": args.warmup,
            "stats": {stage: asdict(s) for stage, s in stats.items()},
            "samples": timings,
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote JSON report -> {args.json}")
    return 1 if budget_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

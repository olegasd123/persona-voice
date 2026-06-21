#!/usr/bin/env python3
"""Measure STT word-error-rate for a backend (automated eval).

    python scripts/eval_stt.py --backend mac  --manifest tests/data/stt_manifest.jsonl
    python scripts/eval_stt.py --backend cuda --manifest manifest.jsonl --json wer.json

The manifest is JSONL, one clip per line: {"audio": "clip1.wav", "text": "the reference ..."}.
Relative `audio` paths resolve against the manifest's directory. Each clip is transcribed by
the configured backend STT and scored against its reference; the report shows per-clip WER
and the pooled corpus WER (edits / total reference words). Needs the backend extra + STT model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from personavoice.adapters.factory import build_backend
from personavoice.audio import read_wav_file
from personavoice.eval.wer import corpus_wer, wer
from personavoice.server.config import Settings, load_backend_config


@dataclass
class ClipResult:
    audio: str
    reference: str
    hypothesis: str
    wer: float


def load_manifest(path: Path) -> list[dict[str, str]]:
    """Parse a JSONL manifest of {"audio", "text"} rows (blank lines ignored)."""
    rows: list[dict[str, str]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict) or "audio" not in obj or "text" not in obj:
            raise ValueError(f"{path}:{lineno}: each row needs 'audio' and 'text' keys")
        rows.append({"audio": str(obj["audio"]), "text": str(obj["text"])})
    if not rows:
        raise ValueError(f"{path}: no clips found")
    return rows


async def _run(settings: Settings, manifest: Path) -> list[ClipResult]:
    backend = build_backend(load_backend_config(settings))
    base = manifest.parent
    results: list[ClipResult] = []
    for row in load_manifest(manifest):
        audio_path = Path(row["audio"])
        if not audio_path.is_absolute():
            audio_path = base / audio_path
        transcript = await backend.stt.transcribe(read_wav_file(audio_path))
        hyp = transcript.text
        results.append(
            ClipResult(
                audio=row["audio"],
                reference=row["text"],
                hypothesis=hyp,
                wer=wer(row["text"], hyp),
            )
        )
        print(f"  {row['audio']}: wer={results[-1].wer:.3f}", file=sys.stderr)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="STT word-error-rate eval for a backend.")
    parser.add_argument("--backend", choices=("mac", "cuda"), default=None, help="override BACKEND")
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL of {audio,text} clips")
    parser.add_argument("--json", type=Path, default=None, help="also write the report as JSON")
    args = parser.parse_args(argv)

    if not args.manifest.is_file():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    try:
        settings = Settings.load(backend=args.backend)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        results = asyncio.run(_run(settings, args.manifest))
    except Exception as exc:  # surface backend/model errors without a traceback wall
        print(f"error: {exc}", file=sys.stderr)
        return 1

    pooled = corpus_wer([(r.reference, r.hypothesis) for r in results])
    print(f"\nSTT WER — backend={settings.backend}  clips={len(results)}")
    print(f"  corpus WER (pooled): {pooled:.3f}")

    if args.json is not None:
        payload = {
            "backend": settings.backend,
            "clips": len(results),
            "corpus_wer": pooled,
            "per_clip": [
                {"audio": r.audio, "wer": r.wer, "hypothesis": r.hypothesis} for r in results
            ],
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote JSON report -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

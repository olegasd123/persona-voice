"""`personavoice-stream-demo`: the M3 streaming voice loop on one machine (no LiveKit).

Same shape as the M1 `personavoice-demo`, but it streams: the reply is synthesized and
played **sentence-by-sentence**, so you hear the first sentence while the rest is still
being generated. It prints time-to-first-token and time-to-first-audio so the streaming win
over the turn-based loop is visible.

    personavoice-stream-demo --wav question.wav --persona companion --play
    personavoice-stream-demo --record 5 --persona language_teacher --play

STT is still one-shot here (transcribe the recorded utterance); live VAD endpointing and
barge-in arrive with the LiveKit agent (`personavoice serve`). This is a developer tool.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path

from ..adapters.factory import build_backend
from ..audio import read_wav_file, write_wav_file
from ..models import Persona
from ..persona.loader import load_persona
from ..persona.registry import PersonaRegistry
from ..server.config import ConfigError, Settings, load_backend_config
from .streaming import StreamingPipeline, StreamMetrics


def _resolve_persona(settings: Settings, ref: str) -> Persona:
    path = Path(ref)
    if path.suffix == ".yaml" or path.exists():
        return load_persona(path)
    return PersonaRegistry(settings.personas_dir).get(ref)


def _record_to_wav(seconds: float, path: Path, sample_rate: int = 16000) -> None:
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - needs the mac extra + a mic
        raise RuntimeError(
            "recording needs sounddevice + soundfile; install the Mac extra: "
            "`pip install -e '.[mac]'` (or pass --wav with a prerecorded file)"
        ) from exc

    print(f"Recording {seconds:.0f}s from the mic... speak now.", file=sys.stderr)
    frames = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    sf.write(str(path), frames.reshape(-1), sample_rate, format="WAV", subtype="PCM_16")
    print("Done recording.", file=sys.stderr)


def _play(path: Path) -> None:
    player = "afplay" if sys.platform == "darwin" else "aplay"
    try:
        subprocess.run([player, str(path)], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"could not play audio with {player!r}: {exc}", file=sys.stderr)


async def _run(
    settings: Settings, persona: Persona, audio_in: bytes, out_dir: Path, play: bool
) -> StreamMetrics:
    backend = build_backend(load_backend_config(settings))

    transcript = await backend.stt.transcribe(audio_in)
    print(f"\nyou said : {transcript.text!r}")
    print(f"{persona.id} : ", end="", flush=True)

    pipe = StreamingPipeline(backend, persona)
    metrics = StreamMetrics()

    idx = 0
    async for wav in pipe.stream_response(transcript.text, metrics=metrics):
        # Persist each sentence's audio and play it as it arrives. afplay blocks, which
        # serializes playback — the point is that sentence 1 starts before the whole reply
        # is generated, not overlapped playback.
        part = out_dir / f"reply_{idx:02d}.wav"
        write_wav_file(part, wav)
        idx += 1
        if play:
            _play(part)
    print(metrics.reply)
    return metrics


def _print_metrics(m: StreamMetrics) -> None:
    def fmt(v: float | None) -> str:
        return f"{v:.2f}s" if v is not None else "n/a"

    print()
    print(
        "timings  : "
        f"first_token={fmt(m.first_token)}  first_audio={fmt(m.first_audio)}  "
        f"total={fmt(m.total)}"
    )
    print("           (first_audio is the perceived latency — when speech starts)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="personavoice-stream-demo",
        description="Streaming voice loop: speak into a wav, hear a persona reply stream (M3).",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--wav", type=Path, help="input wav file (your spoken question)")
    src.add_argument("--record", type=float, metavar="SECONDS", help="record from the mic instead")
    parser.add_argument(
        "--persona", default="companion", help="persona id (config/personas) or a .yaml path"
    )
    parser.add_argument("--backend", choices=("mac", "cuda"), default=None, help="override BACKEND")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("reply_stream"), help="dir for per-sentence wavs"
    )
    parser.add_argument("--play", action="store_true", help="play each sentence as it streams")
    args = parser.parse_args(argv)

    try:
        settings = Settings.load(backend=args.backend)
        persona = _resolve_persona(settings, args.persona)
    except (ConfigError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.record is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            rec_path = Path(tmp.name)
        _record_to_wav(args.record, rec_path)
        audio_in = read_wav_file(rec_path)
    else:
        if not args.wav.is_file():
            print(f"error: input wav not found: {args.wav}", file=sys.stderr)
            return 2
        audio_in = read_wav_file(args.wav)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        metrics = asyncio.run(_run(settings, persona, audio_in, args.out_dir, args.play))
    except Exception as exc:  # surface backend/model errors without a traceback wall
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_metrics(metrics)
    print(f"\nwrote per-sentence audio -> {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

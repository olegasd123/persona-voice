"""`personavoice-demo`: the M1 offline voice loop — speak into a wav, hear a reply.

    personavoice-demo --wav question.wav --persona companion --play
    personavoice-demo --record 5 --persona language_teacher --play   # record from mic

It wires the configured backend (BACKEND=mac by default) to a persona and runs one turn
through the file-based pipeline, writing the spoken reply to a wav (and optionally playing
it). This is a developer tool; the live streaming server arrives in M3.
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
from ..server.config import ConfigError, Settings, load_backend_config, load_voice_registry
from .pipeline import Pipeline, TurnResult


def _resolve_persona(settings: Settings, ref: str) -> Persona:
    """Resolve a persona by id (from config/personas) or by an explicit file path."""
    path = Path(ref)
    if path.suffix == ".yaml" or path.exists():
        return load_persona(path)
    return PersonaRegistry(settings.personas_dir).get(ref)


def _record_to_wav(seconds: float, path: Path, sample_rate: int = 16000) -> None:
    """Record `seconds` of mono audio from the default mic into a wav file."""
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
    """Play a wav file (macOS `afplay`)."""
    player = "afplay" if sys.platform == "darwin" else "aplay"
    try:
        subprocess.run([player, str(path)], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"could not play audio with {player!r}: {exc}", file=sys.stderr)


def _print_result(result: TurnResult, persona: Persona) -> None:
    t = result.timings
    print()
    print(f"you said : {result.transcript.text!r}")
    print(f"{persona.id} : {result.reply}")
    print()
    print(
        "timings  : "
        f"stt={t.get('stt', 0):.2f}s  llm={t.get('llm', 0):.2f}s  "
        f"tts={t.get('tts', 0):.2f}s  total={t.get('total', 0):.2f}s"
    )


async def _run(settings: Settings, persona: Persona, audio_in: bytes) -> TurnResult:
    backend = build_backend(load_backend_config(settings))
    voices = load_voice_registry(settings)
    pipeline = Pipeline(backend, persona, voices)
    return await pipeline.run_turn(audio_in)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="personavoice-demo",
        description="Offline voice loop: speak into a wav, hear a persona reply (M1).",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--wav", type=Path, help="input wav file (your spoken question)")
    src.add_argument("--record", type=float, metavar="SECONDS", help="record from the mic instead")
    parser.add_argument(
        "--persona", default="companion", help="persona id (config/personas) or a .yaml path"
    )
    parser.add_argument("--backend", choices=("mac", "cuda"), default=None, help="override BACKEND")
    parser.add_argument("--out", type=Path, default=Path("reply.wav"), help="output wav path")
    parser.add_argument("--play", action="store_true", help="play the reply after synthesizing")
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

    try:
        result = asyncio.run(_run(settings, persona, audio_in))
    except Exception as exc:  # surface backend/model errors without a traceback wall
        print(f"error: {exc}", file=sys.stderr)
        return 1

    write_wav_file(args.out, result.audio)
    _print_result(result, persona)
    print(f"\nwrote reply audio -> {args.out}")
    if args.play:
        _play(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

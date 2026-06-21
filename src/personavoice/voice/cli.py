"""`personavoice-clone`: clone a voice from a short sample and assign it to a persona.

    personavoice-clone --sample me.wav  --name my_voice --assign companion
    personavoice-clone --record 10      --name my_voice --assign companion --say "Hi there." --play
    personavoice-clone --list
    personavoice-clone --unassign companion

It runs the configured cloning backend (f5_mlx on Mac, chatterbox on CUDA), stores the clone
under the clones dir, and (with `--assign`) makes the persona speak in that voice everywhere —
the demos and the live LiveKit agent pick it up via the voice registry. `--say ... --play`
synthesizes a line in the new voice right away as a quick check.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

from ..adapters.factory import build_backend
from ..audio import read_wav_file, write_wav_file
from ..orchestrator.demo import _play, _record_to_wav
from ..server.config import (
    ConfigError,
    Settings,
    load_backend_config,
    load_clones_store,
)
from .clone import CloneError, ClonesStore, VoiceCloner


def _print_clones(store: ClonesStore) -> None:
    if not len(store):
        print("No cloned voices yet. Create one with --sample/--record + --name.")
        return
    assigned_to = {name: pid for pid, name in store.assignments.items()}
    print(f"Cloned voices ({len(store)}):")
    for name in store.names():
        cv = store.get(name)
        who = f"  -> assigned to {assigned_to[name]}" if name in assigned_to else ""
        backend = f" [{cv.backend}]" if cv and cv.backend else ""
        print(f"  {name}{backend}{who}")


def _sample_bytes(args: argparse.Namespace) -> bytes:
    if args.record is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            rec_path = Path(tmp.name)
        _record_to_wav(args.record, rec_path)
        return read_wav_file(rec_path)
    if not args.sample.is_file():
        raise CloneError(f"sample wav not found: {args.sample}")
    return read_wav_file(args.sample)


async def _clone(settings: Settings, store: ClonesStore, args: argparse.Namespace) -> int:
    sample_wav = _sample_bytes(args)
    backend = build_backend(load_backend_config(settings))
    cloner = VoiceCloner(backend, store)
    voice = await cloner.clone(sample_wav, args.name, assign_to=args.assign)

    print(f"\ncloned '{voice.name}'  (backend={voice.backend})")
    print(f"  sample : {voice.sample_path}")
    if voice.ref_text:
        print(f"  ref_text: {voice.ref_text!r}")
    if args.assign:
        print(f"  assigned to persona: {args.assign}")

    if args.say:
        audio = await backend.tts.synthesize(args.say, voice)
        write_wav_file(args.out, audio)
        print(f"\nwrote sample line -> {args.out}")
        if args.play:
            _play(args.out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="personavoice-clone",
        description="Clone a voice from a short sample and assign it to a persona.",
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--sample", type=Path, help="reference wav (~10 s of clear speech)")
    src.add_argument(
        "--record", type=float, metavar="SECONDS", help="record the sample from the mic"
    )
    parser.add_argument("--name", help="name for the cloned voice (letters/digits/-/_)")
    parser.add_argument("--assign", metavar="PERSONA", help="assign the clone to this persona id")
    parser.add_argument("--list", action="store_true", help="list cloned voices and assignments")
    parser.add_argument("--unassign", metavar="PERSONA", help="remove a persona's clone assignment")
    parser.add_argument("--say", help="after cloning, synthesize this line in the cloned voice")
    parser.add_argument(
        "--out", type=Path, default=Path("clone_sample.wav"), help="--say output wav"
    )
    parser.add_argument("--play", action="store_true", help="play the --say output")
    parser.add_argument("--backend", choices=("mac", "cuda"), default=None, help="override BACKEND")
    args = parser.parse_args(argv)

    try:
        settings = Settings.load(backend=args.backend)
        store = load_clones_store(settings)
    except (ConfigError, CloneError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.list:
        _print_clones(store)
        return 0

    if args.unassign:
        removed = store.unassign(args.unassign)
        print(
            f"unassigned clone from {args.unassign!r}"
            if removed
            else f"no clone was assigned to {args.unassign!r}"
        )
        return 0

    if args.record is None and args.sample is None:
        parser.error("provide --sample or --record (or use --list / --unassign)")
    if not args.name:
        parser.error("--name is required when cloning")

    try:
        return asyncio.run(_clone(settings, store, args))
    except (CloneError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # surface backend/model errors without a traceback wall
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

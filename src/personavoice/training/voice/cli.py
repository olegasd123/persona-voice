"""`personavoice-voice-train`: build a dataset, fine-tune, A/B, and register a voice.

    personavoice-voice-train dataset  --voice my_voice --audio-dir clips/        # build metadata.csv
    personavoice-voice-train run      --voice my_voice --engine f5 [--dry-run]   # fine-tune (CUDA)
    personavoice-voice-train eval     --voice my_voice --clone my_clone --target-dir held_out/
    personavoice-voice-train register --voice my_voice --checkpoint ckpt/ --assign companion
    personavoice-voice-train list     [--unassign companion]

`dataset` uses the configured backend's STT to auto-transcribe clips; `run` shells out to the
engine's trainer (F5-TTS / Chatterbox) and is safe to preview with `--dry-run`; `eval` synthesizes
probes with the fine-tuned voice and the zero-shot clone and scores speaker similarity to held-out
target clips; `register` folds a trained checkpoint into the voice registry (it then takes
precedence over a clone for the assigned persona). Data lands under `training/voice/datasets/<voice>/`,
checkpoints under `<models>/finetuned/<voice>/`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from ...adapters.factory import build_backend
from ...audio import read_wav_file
from ...models import VoiceRef
from ...server.config import (
    ConfigError,
    Settings,
    load_backend_config,
    load_clones_store,
    load_finetuned_store,
)
from ...voice.clone import ClonesStore
from ...voice.finetuned import FinetunedVoice, FinetunedVoiceError, FinetunedVoicesStore
from . import dataset as ds
from . import evaluate as ev
from .config import VoiceTrainConfig, VoiceTrainConfigError, from_voice, load_config
from .finetune import VoiceTrainingError, run_finetune

_DATASETS_ROOT = Path("training/voice/datasets")

# Neutral sentences to synthesize for the A/B (content-independent; we score *who* it sounds
# like, not what it says).
DEFAULT_VOICE_PROBES = (
    "Thanks so much for being here today.",
    "Let me think about that for a moment.",
    "That's a really interesting question.",
    "I'd love to hear more about what you mean.",
    "We can take this one step at a time.",
)


def _data_dir(args: argparse.Namespace) -> Path:
    return Path(args.data_dir) if args.data_dir else _DATASETS_ROOT / args.voice


def _output_dir(settings: Settings, args: argparse.Namespace) -> Path:
    return Path(args.output) if args.output else settings.finetuned_dir / args.voice


def _wavs_in(directory: str) -> list[str]:
    """Sorted .wav paths under `directory` (sync — kept out of the async commands)."""
    return sorted(str(p) for p in Path(directory).glob("*.wav"))


def _read_target_clips(directory: str) -> list[bytes]:
    """Read every .wav under `directory` as bytes (sync)."""
    return [read_wav_file(p) for p in sorted(Path(directory).glob("*.wav"))]


# --------------------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------------------


async def _dataset(settings: Settings, args: argparse.Namespace) -> int:
    data_dir = _data_dir(args)
    if args.transcripts:
        clips = ds.read_metadata(args.transcripts)
        print(f"read {len(clips)} labeled clips from {args.transcripts}")
    else:
        wavs = _wavs_in(args.audio_dir)
        if not wavs:
            print(f"no .wav files under {args.audio_dir}", file=sys.stderr)
            return 1
        backend = build_backend(load_backend_config(settings))
        print(f"transcribing {len(wavs)} clips via {backend.stt.name} ...")
        clips = await ds.transcribe_clips(backend.stt, wavs)
        dropped = len(wavs) - len(clips)
        if dropped:
            print(f"  ({dropped} clip(s) produced no transcript and were dropped)")

    if args.probe_durations:
        clips = ds.probe_durations(clips, root=args.audio_dir)

    stats = ds.validate_dataset(
        clips,
        min_clips=args.min_clips,
        min_total_seconds=args.min_seconds,
    )
    out = ds.write_metadata(data_dir / "metadata.csv", clips)
    secs = f"{stats.total_seconds:.0f}s" if stats.known_durations else "duration unknown"
    print(f"wrote {stats.num_clips} clips ({secs}) -> {out}")
    return 0


# --------------------------------------------------------------------------------------
# run (fine-tune)
# --------------------------------------------------------------------------------------


def _train_config(settings: Settings, args: argparse.Namespace) -> VoiceTrainConfig:
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = from_voice(
            args.voice,
            engine=args.engine,
            data_dir=_data_dir(args),
            output_dir=_output_dir(settings, args),
        )
    if args.base_model:
        cfg.base_model = args.base_model
    return cfg


def _run_finetune(settings: Settings, args: argparse.Namespace) -> int:
    cfg = _train_config(settings, args)
    plan = run_finetune(cfg, dry_run=args.dry_run, skip_prepare=args.skip_prepare)
    if plan.prepare_command:
        print(f"prepare: {' '.join(plan.prepare_command)}")
    print(f"command: {' '.join(plan.command)}")
    if plan.config_filename:
        print(f"trainer config: {plan.config_filename}")
    if args.dry_run:
        print("(dry run — nothing launched)")
    else:
        print(f"fine-tuned checkpoint -> {cfg.output_dir}")
    return 0


# --------------------------------------------------------------------------------------
# eval (A/B vs the zero-shot clone)
# --------------------------------------------------------------------------------------


def _finetuned_ref(store: FinetunedVoicesStore, args: argparse.Namespace) -> VoiceRef:
    if args.checkpoint:
        return VoiceRef(id=args.voice, name=args.voice, model_path=args.checkpoint)
    fv = store.get(args.voice)
    if fv is None:
        raise FinetunedVoiceError(
            f"no fine-tuned voice {args.voice!r}; pass --checkpoint or register it first"
        )
    return VoiceRef(id=fv.name, name=fv.name, model_path=fv.checkpoint_path)


def _clone_ref(clones: ClonesStore, tts_name: str, name: str) -> VoiceRef:
    ref = clones.voice_ref(name, tts_name)
    if ref is None:
        known = ", ".join(clones.names()) or "(none)"
        raise FinetunedVoiceError(f"unknown clone {name!r}; cloned voices: {known}")
    return ref


def _load_probes(path: str | None) -> list[str]:
    if not path:
        return list(DEFAULT_VOICE_PROBES)
    raw = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(x) for x in data]
    except json.JSONDecodeError:
        pass
    return [line for line in raw.splitlines() if line.strip()]


async def _eval(settings: Settings, args: argparse.Namespace) -> int:
    store = load_finetuned_store(settings)
    clones = load_clones_store(settings)
    backend = build_backend(load_backend_config(settings))
    if not getattr(backend.tts, "supports_cloning", False):
        print(
            f"the active TTS backend {backend.tts.name!r} can't load fine-tuned/cloned voices; "
            "switch to a cloning backend (f5_mlx on Mac, chatterbox on CUDA)",
            file=sys.stderr,
        )
        return 2

    finetuned = _finetuned_ref(store, args)
    clone = _clone_ref(clones, backend.tts.name, args.clone)
    target_clips = _read_target_clips(args.target_dir)
    if not target_clips:
        print(f"no target .wav clips under {args.target_dir}", file=sys.stderr)
        return 1

    probes = _load_probes(args.probes)
    print(
        f"A/B: fine-tuned {finetuned.id!r} vs clone {clone.id!r} over {len(probes)} probes "
        f"against {len(target_clips)} target clips ..."
    )
    scores = await ev.evaluate_voices(
        backend.tts,
        probes=probes,
        finetuned_voice=finetuned,
        clone_voice=clone,
        target_clips=target_clips,
        embedder=ev.ResemblyzerEmbedder(),
        margin=args.margin,
    )
    print(f"\n  fine-tuned similarity : {scores.finetuned_similarity:.3f}")
    print(f"  clone similarity      : {scores.clone_similarity:.3f}")
    print(f"  delta (ft - clone)    : {scores.delta:+.3f}  (margin {scores.margin:.3f})")
    verdict = "PASS — fine-tune is clearly closer to the target" if scores.passes else "no clear win"
    print(f"  verdict               : {verdict}")

    if args.register and scores.passes:
        fv = store.get(args.voice)
        checkpoint = args.checkpoint or (fv.checkpoint_path if fv else "")
        store.record(
            FinetunedVoice(
                name=args.voice,
                checkpoint_path=checkpoint,
                engine=(fv.engine if fv else None),
                base_model=(fv.base_model if fv else None),
                speaker=(fv.speaker if fv else None),
                similarity=scores.finetuned_similarity,
                clone_similarity=scores.clone_similarity,
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
        )
        print(f"\nrecorded eval verdict on fine-tuned voice {args.voice!r}")
    return 0


# --------------------------------------------------------------------------------------
# register / list / unassign
# --------------------------------------------------------------------------------------


def _register(settings: Settings, args: argparse.Namespace) -> int:
    store = load_finetuned_store(settings)
    checkpoint = args.checkpoint or str(_output_dir(settings, args))
    store.record(
        FinetunedVoice(
            name=args.voice,
            checkpoint_path=checkpoint,
            engine=args.engine,
            base_model=args.base_model,
            speaker=args.speaker,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
    )
    print(f"registered fine-tuned voice {args.voice!r} -> {checkpoint}")
    if args.assign:
        store.assign(args.assign, args.voice)
        print(f"  assigned to persona: {args.assign}")
    return 0


def _print_voices(store: FinetunedVoicesStore) -> None:
    if not len(store):
        print("No fine-tuned voices yet. Train one with `run`, then `register` it.")
        return
    assigned_to = {name: pid for pid, name in store.assignments.items()}
    print(f"Fine-tuned voices ({len(store)}):")
    for name in store.names():
        fv = store.get(name)
        who = f"  -> assigned to {assigned_to[name]}" if name in assigned_to else ""
        engine = f" [{fv.engine}]" if fv and fv.engine else ""
        sim = ""
        if fv and fv.similarity is not None and fv.clone_similarity is not None:
            sim = f"  (sim {fv.similarity:.3f} vs clone {fv.clone_similarity:.3f})"
        print(f"  {name}{engine}{who}{sim}")


def _list(settings: Settings, args: argparse.Namespace) -> int:
    store = load_finetuned_store(settings)
    if args.unassign:
        removed = store.unassign(args.unassign)
        print(
            f"unassigned fine-tuned voice from {args.unassign!r}"
            if removed
            else f"no fine-tuned voice was assigned to {args.unassign!r}"
        )
        return 0
    _print_voices(store)
    return 0


# --------------------------------------------------------------------------------------
# argparse wiring
# --------------------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="personavoice-voice-train",
        description="Build a dataset, fine-tune, A/B, and register a high-fidelity voice.",
    )
    parser.add_argument("--backend", choices=("mac", "cuda"), default=None, help="override BACKEND")
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dataset", help="build/validate the target-speaker metadata.csv")
    d.add_argument("--voice", required=True)
    d.add_argument("--audio-dir", help="dir of target-speaker .wav clips (auto-transcribed)")
    d.add_argument("--transcripts", help="existing metadata.csv (audio_path|text) to use instead")
    d.add_argument("--data-dir", default=None, help="output dir (default training/voice/datasets/<voice>)")
    d.add_argument("--probe-durations", action="store_true", help="decode clips to fill durations")
    d.add_argument("--min-clips", type=int, default=10)
    d.add_argument("--min-seconds", type=float, default=120.0, help="min total audio (with --probe-durations)")

    r = sub.add_parser("run", help="run the voice trainer (f5-tts / chatterbox; CUDA)")
    r.add_argument("--voice", required=True)
    r.add_argument("--engine", choices=("f5", "chatterbox"), default="f5")
    r.add_argument("--config", default=None, help="a training/voice/configs/*.yaml")
    r.add_argument("--data-dir", default=None)
    r.add_argument("--output", default=None, help="checkpoint output dir")
    r.add_argument("--base-model", default=None, help="override the base model / exp_name")
    r.add_argument("--skip-prepare", action="store_true", help="skip the dataset-prep step")
    r.add_argument("--dry-run", action="store_true", help="write config + print commands, don't launch")

    e = sub.add_parser("eval", help="A/B speaker similarity: fine-tuned voice vs zero-shot clone")
    e.add_argument("--voice", required=True, help="fine-tuned voice name (in the store)")
    e.add_argument("--checkpoint", default=None, help="checkpoint dir (if not registered yet)")
    e.add_argument("--clone", required=True, help="clone name to compare against")
    e.add_argument("--target-dir", required=True, help="dir of held-out real target-speaker .wav clips")
    e.add_argument("--probes", default=None, help="probe lines (JSON array or one per line)")
    e.add_argument("--margin", type=float, default=0.0, help="delta a clear win must clear")
    e.add_argument("--register", action="store_true", help="on PASS, store the verdict on the voice")

    rg = sub.add_parser("register", help="fold a trained checkpoint into the voice registry")
    rg.add_argument("--voice", required=True)
    rg.add_argument("--checkpoint", default=None, help="checkpoint dir (default <models>/finetuned/<voice>)")
    rg.add_argument("--engine", choices=("f5", "chatterbox"), default="f5")
    rg.add_argument("--base-model", default=None)
    rg.add_argument("--speaker", default=None, help="target speaker label (provenance)")
    rg.add_argument("--output", default=None, help="checkpoint output dir (for the default path)")
    rg.add_argument("--assign", metavar="PERSONA", help="assign the voice to this persona id")

    li = sub.add_parser("list", help="list fine-tuned voices and assignments")
    li.add_argument("--unassign", metavar="PERSONA", help="remove a persona's fine-tuned-voice assignment")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        settings = Settings.load(backend=args.backend)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.cmd == "dataset" and not (args.audio_dir or args.transcripts):
        parser.error("provide --audio-dir (to auto-transcribe) or --transcripts")

    try:
        if args.cmd == "dataset":
            return asyncio.run(_dataset(settings, args))
        if args.cmd == "run":
            return _run_finetune(settings, args)
        if args.cmd == "eval":
            return asyncio.run(_eval(settings, args))
        if args.cmd == "register":
            return _register(settings, args)
        if args.cmd == "list":
            return _list(settings, args)
    except (ConfigError, VoiceTrainConfigError, ds.DatasetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (VoiceTrainingError, FinetunedVoiceError, ev.VoiceEvalError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # surface backend/model errors without a traceback wall
        print(f"error: {exc}", file=sys.stderr)
        return 1
    parser.error(f"unknown command {args.cmd!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

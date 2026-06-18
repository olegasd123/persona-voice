"""`personavoice-train`: curate data, train, merge, and eval per-persona LoRAs (M7).

    personavoice-train curate --persona hr_interviewer --num 20 --exchanges 4
    personavoice-train run    --persona hr_interviewer [--config cfg.yaml] [--dry-run]
    personavoice-train merge  --persona hr_interviewer --adapter models/adapters/hr_interviewer
    personavoice-train eval   --persona hr_interviewer --compare

`curate`/`eval` build the configured backend's LLM (talk to LM Studio on Mac / vLLM on CUDA);
`run`/`merge` shell out to the trainer and are safe to preview with `--dry-run`. Data lands under
`training/persona_lora/datasets/<persona>/`, adapters under `<models>/adapters/<persona>/`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from ..adapters.factory import build_backend
from ..models import Persona
from ..persona.loader import PersonaError
from ..server.config import (
    ConfigError,
    Settings,
    load_all_personas,
    load_backend_config,
)
from . import curate as curate_mod
from . import eval as eval_mod
from .config import LoRATrainConfig, TrainConfigError, from_persona, load_config
from .dataset import DatasetError, clean_example, split_examples, write_dataset, write_jsonl
from .merge import MergeError, run_merge
from .train import TrainingError, run_training

_DATASETS_ROOT = Path("training/persona_lora/datasets")


def _resolve_persona(settings: Settings, persona_id: str) -> Persona:
    personas = load_all_personas(settings)
    if persona_id not in personas:
        known = ", ".join(sorted(personas)) or "(none)"
        raise ConfigError(f"unknown persona {persona_id!r}; known: {known}")
    return personas[persona_id]


def _data_dir(args: argparse.Namespace) -> Path:
    return Path(args.data_dir) if args.data_dir else _DATASETS_ROOT / args.persona


def _adapter_dir(settings: Settings, args: argparse.Namespace) -> Path:
    return Path(args.adapter) if args.adapter else settings.models_dir / "adapters" / args.persona


# --------------------------------------------------------------------------------------
# curate
# --------------------------------------------------------------------------------------


async def _curate(settings: Settings, args: argparse.Namespace) -> int:
    persona = _resolve_persona(settings, args.persona)
    backend = build_backend(load_backend_config(settings))
    print(f"curating {args.num} dialogues for {persona.id} via {backend.llm.name} ...")
    examples = await curate_mod.generate_dataset(
        backend.llm,
        persona,
        num_dialogues=args.num,
        num_exchanges=args.exchanges,
    )
    if not examples:
        print("no usable dialogues were generated", file=sys.stderr)
        return 1

    if args.spoken_clean:
        examples = [clean_example(ex) for ex in examples]

    data_dir = _data_dir(args)
    if args.fmt == "sharegpt":
        out = data_dir / f"{persona.id}.sharegpt.json"
        write_dataset(out, examples, fmt="sharegpt")
        print(f"wrote {len(examples)} examples -> {out}")
    else:
        train, valid = split_examples(examples, valid_fraction=args.valid_fraction, seed=args.seed)
        write_jsonl(data_dir / "train.jsonl", train)
        print(f"wrote {len(train)} train examples -> {data_dir / 'train.jsonl'}")
        if valid:
            write_jsonl(data_dir / "valid.jsonl", valid)
            print(f"wrote {len(valid)} valid examples -> {data_dir / 'valid.jsonl'}")
    return 0


# --------------------------------------------------------------------------------------
# run (train)
# --------------------------------------------------------------------------------------


def _train_config(settings: Settings, args: argparse.Namespace) -> LoRATrainConfig:
    if args.config:
        cfg = load_config(args.config)
    else:
        persona = _resolve_persona(settings, args.persona)
        cfg = from_persona(
            persona,
            backend=settings.backend,
            data_dir=_data_dir(args),
            adapter_dir=_adapter_dir(settings, args),
        )
    if args.base_model:
        cfg.base_model = args.base_model
    return cfg


def _run_train(settings: Settings, args: argparse.Namespace) -> int:
    cfg = _train_config(settings, args)
    plan = run_training(cfg, dry_run=args.dry_run)
    print(f"trainer config: {plan.config_filename}")
    print(f"command: {' '.join(plan.command)}")
    if args.dry_run:
        print("(dry run — nothing launched)")
    return 0


# --------------------------------------------------------------------------------------
# merge
# --------------------------------------------------------------------------------------


def _run_merge(settings: Settings, args: argparse.Namespace) -> int:
    persona = _resolve_persona(settings, args.persona)
    adapter = _adapter_dir(settings, args)
    out = Path(args.out) if args.out else settings.models_dir / "merged" / persona.id
    command = run_merge(
        backend=settings.backend,
        base_model=args.base_model or persona.llm.base_model,
        adapter_dir=adapter,
        out_dir=out,
        dry_run=args.dry_run,
    )
    print(f"command: {' '.join(command)}")
    if args.dry_run:
        print("(dry run — nothing launched)")
    else:
        print(f"merged model -> {out}")
    return 0


# --------------------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------------------


def _load_probes(path: str | None) -> list[str] | None:
    if not path:
        return None
    raw = Path(path).read_text(encoding="utf-8")
    # Accept a JSON array or one probe per line.
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(x) for x in data]
    except json.JSONDecodeError:
        pass
    return [line for line in raw.splitlines() if line.strip()]


def _print_scores(label: str, scores: eval_mod.EvalScores) -> None:
    print(f"\n{label}  (n={scores.n})")
    print(f"  persona_adherence : {scores.persona_adherence:.3f}")
    print(f"  turn_style_fit    : {scores.turn_style_fit:.3f}")
    print(f"  spoken_clean_rate : {scores.spoken_clean_rate:.3f}")
    print(f"  question_rate     : {scores.question_rate:.3f}")
    print(f"  follow_up_align   : {scores.follow_up_alignment:.3f}")
    print(f"  keyword_coverage  : {scores.keyword_coverage:.3f}")
    print(f"  mean_words        : {scores.mean_words:.1f}")


async def _run_eval(settings: Settings, args: argparse.Namespace) -> int:
    persona = _resolve_persona(settings, args.persona)
    backend = build_backend(load_backend_config(settings))
    probes = _load_probes(args.probes)
    keywords = [k.strip() for k in args.keywords.split(",")] if args.keywords else None

    base = await eval_mod.evaluate(
        backend.llm, persona, probes=probes, keywords=keywords, bare_prompt=args.bare_prompt
    )
    _print_scores("prompt-only" + (" [bare]" if args.bare_prompt else ""), base)

    if args.compare:
        lora_persona = persona.model_copy(deep=True)
        lora_persona.llm.lora = args.lora or persona.id
        lora = await eval_mod.evaluate(
            backend.llm, lora_persona, probes=probes, keywords=keywords, bare_prompt=args.bare_prompt
        )
        _print_scores(f"LoRA ({lora_persona.llm.lora})", lora)
        print("\ndelta (lora - prompt-only):")
        for name, (_b, _l, d) in eval_mod.compare(base, lora).items():
            print(f"  {name:18s}: {d:+.3f}")
    return 0


# --------------------------------------------------------------------------------------
# argparse wiring
# --------------------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="personavoice-train",
        description="Curate data, train, merge, and eval per-persona LoRAs (M7).",
    )
    parser.add_argument("--backend", choices=("mac", "cuda"), default=None, help="override BACKEND")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cur = sub.add_parser("curate", help="generate in-character dialogues with the persona LLM")
    cur.add_argument("--persona", required=True)
    cur.add_argument("--num", type=int, default=20, help="number of dialogues")
    cur.add_argument(
        "--exchanges", type=int, default=4, help="user/assistant exchanges per dialogue"
    )
    cur.add_argument("--fmt", choices=("mlx_chat", "sharegpt"), default="mlx_chat")
    cur.add_argument(
        "--data-dir",
        default=None,
        help="output dir (default training/persona_lora/datasets/<persona>)",
    )
    cur.add_argument("--valid-fraction", type=float, default=0.1)
    cur.add_argument("--seed", type=int, default=0)
    cur.add_argument(
        "--spoken-clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="strip markdown from turns so targets read cleanly through TTS (default: on)",
    )

    run = sub.add_parser("run", help="run the LoRA trainer (mlx-lm on mac / LLaMA-Factory on cuda)")
    run.add_argument("--persona", required=True)
    run.add_argument("--config", default=None, help="a training/persona_lora/configs/*.yaml")
    run.add_argument("--data-dir", default=None)
    run.add_argument("--adapter", default=None, help="adapter output dir")
    run.add_argument("--base-model", default=None, help="override the base model")
    run.add_argument(
        "--dry-run", action="store_true", help="write config + print command, don't launch"
    )

    mrg = sub.add_parser("merge", help="fuse a trained adapter into base weights for serving")
    mrg.add_argument("--persona", required=True)
    mrg.add_argument("--adapter", default=None, help="trained adapter dir")
    mrg.add_argument("--out", default=None, help="merged model output dir")
    mrg.add_argument("--base-model", default=None)
    mrg.add_argument("--dry-run", action="store_true")

    ev = sub.add_parser("eval", help="score persona adherence (prompt-only vs LoRA)")
    ev.add_argument("--persona", required=True)
    ev.add_argument("--probes", default=None, help="probes file (JSON array or one per line)")
    ev.add_argument("--keywords", default=None, help="comma-separated expected vocabulary")
    ev.add_argument("--compare", action="store_true", help="also run with the LoRA and show deltas")
    ev.add_argument("--lora", default=None, help="lora name to request (default: persona id)")
    ev.add_argument(
        "--bare-prompt",
        action="store_true",
        help="send only the authored system_prompt (drop turn-style + spoken-clean directives), "
        "to test whether the LoRA internalizes persona behavior beyond prompting",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        settings = Settings.load(backend=args.backend)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.cmd == "curate":
            return asyncio.run(_curate(settings, args))
        if args.cmd == "run":
            return _run_train(settings, args)
        if args.cmd == "merge":
            return _run_merge(settings, args)
        if args.cmd == "eval":
            return asyncio.run(_run_eval(settings, args))
    except (ConfigError, PersonaError, DatasetError, TrainConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (TrainingError, MergeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # surface backend/model errors without a traceback wall
        print(f"error: {exc}", file=sys.stderr)
        return 1
    parser.error(f"unknown command {args.cmd!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

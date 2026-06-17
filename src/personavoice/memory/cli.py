"""`personavoice-memory`: inspect and manage per-user conversation memory (M8).

    personavoice-memory --list                       # users, turn/session counts, consent
    personavoice-memory --show alice                 # profile (facts + summary) + recent turns
    personavoice-memory --grant alice [--training]   # opt a user in (recording / training use)
    personavoice-memory --revoke alice               # withdraw consent
    personavoice-memory --consolidate alice          # (re)distill the profile via the LLM
    personavoice-memory --export alice --out a.json  # portable dump (privacy)
    personavoice-memory --delete alice               # wipe everything for a user (privacy)
    personavoice-memory --distill alice --out a.jsonl  # transcripts -> persona-LoRA dataset
    personavoice-memory --gen-key                    # generate a PERSONAVOICE_MEMORY_KEY

This is the privacy + operations surface for the acceptance criterion "user can wipe their
data": `--export` and `--delete` are the per-user export/erase controls, and consent is set
here (the cascade refuses to record without it).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from ..persona.loader import load_personas
from ..persona.prompt import render_system_prompt
from ..server.config import (
    ConfigError,
    Settings,
    build_conversation_memory,
    load_memory_store,
)
from .distill import distill_user
from .profile import UserProfile
from .store import FernetCipher, MemoryStore, MemoryStoreError


def _load_profile(store: MemoryStore, user_id: str) -> UserProfile:
    raw = store.load_profile_raw(user_id)
    return UserProfile(user_id=user_id) if raw is None else UserProfile.model_validate(raw)


def _cmd_list(store: MemoryStore) -> int:
    users = store.users()
    if not users:
        print(f"No stored memory yet (dir: {store.dir}).")
        return 0
    enc = "encrypted" if store.encrypted else "plaintext"
    print(f"Memory store: {store.dir}  ({enc})")
    for uid in users:
        consent = store.get_consent(uid)
        turns = store.read_turns(uid)
        sessions = store.session_ids(uid)
        flags = []
        if consent.granted:
            flags.append("consent")
        if consent.allow_training:
            flags.append("training")
        tag = f"[{', '.join(flags)}]" if flags else "[no consent]"
        print(f"  {uid:<24} {len(turns):>4} turns  {len(sessions):>3} sessions  {tag}")
    return 0


def _cmd_show(store: MemoryStore, user_id: str, *, limit: int) -> int:
    if not store.has_user(user_id):
        print(f"no stored memory for user {user_id!r}", file=sys.stderr)
        return 1
    consent = store.get_consent(user_id)
    profile = _load_profile(store, user_id)
    print(f"User: {user_id}")
    print(f"  consent: granted={consent.granted} allow_training={consent.allow_training}")
    if profile.summary:
        print(f"\nSummary:\n  {profile.summary}")
    if profile.facts:
        print(f"\nFacts ({len(profile.facts)}):")
        for fact in profile.fact_texts():
            print(f"  - {fact}")
    turns = store.read_turns(user_id, limit=limit)
    if turns:
        print(f"\nRecent turns (last {len(turns)}):")
        for t in turns:
            print(f"  [{t.persona_id}] {t.role.value}: {t.content}")
    return 0


def _cmd_distill(settings: Settings, store: MemoryStore, user_id: str, args: argparse.Namespace) -> int:
    personas = load_personas(settings.personas_dir)
    system_prompts = {pid: render_system_prompt(p) for pid, p in personas.items()}
    examples = distill_user(store, user_id, system_prompts=system_prompts, persona_id=args.persona)
    if not examples:
        print(f"no usable training examples distilled for user {user_id!r}")
        return 0
    from ..training.dataset import write_dataset

    out = args.out or Path(f"{user_id}_persona.jsonl")
    write_dataset(out, examples, fmt="mlx_chat")
    print(f"distilled {len(examples)} example(s) -> {out}")
    return 0


async def _cmd_consolidate(settings: Settings, user_id: str, args: argparse.Namespace) -> int:
    from ..adapters.factory import build_backend
    from ..server.config import load_backend_config

    backend = build_backend(load_backend_config(settings))
    memory = build_conversation_memory(settings, backend)
    profile = await memory.consolidate(user_id, max_turns=args.max_turns)
    await memory.aclose()
    if profile is None:
        print(f"nothing consolidated for {user_id!r} (no consent, no LLM, or no turns)")
        return 0
    print(f"profile for {user_id!r}: {len(profile.facts)} fact(s)")
    for fact in profile.fact_texts():
        print(f"  - {fact}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="personavoice-memory",
        description="Inspect and manage per-user conversation memory (M8).",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="list users + consent + counts")
    action.add_argument("--show", metavar="USER", help="show a user's profile + recent turns")
    action.add_argument("--grant", metavar="USER", help="grant recording consent for a user")
    action.add_argument("--revoke", metavar="USER", help="withdraw a user's consent")
    action.add_argument("--consolidate", metavar="USER", help="(re)distill a user's profile via LLM")
    action.add_argument("--export", metavar="USER", help="export everything stored for a user (JSON)")
    action.add_argument("--delete", metavar="USER", help="wipe everything stored for a user")
    action.add_argument("--distill", metavar="USER", help="export a user's transcripts as LoRA data")
    action.add_argument("--gen-key", action="store_true", help="generate a Fernet memory key")

    parser.add_argument("--training", action="store_true", help="with --grant: also opt into training use")
    parser.add_argument("--out", type=Path, help="output file (--export / --distill)")
    parser.add_argument("--persona", metavar="ID", help="with --distill: only this persona's turns")
    parser.add_argument("--limit", type=int, default=20, help="with --show: number of recent turns")
    parser.add_argument("--max-turns", type=int, default=40, help="with --consolidate: turns to distill")
    parser.add_argument("--yes", action="store_true", help="with --delete: skip the confirmation")
    parser.add_argument("--backend", choices=("mac", "cuda"), default=None, help="override BACKEND")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.gen_key:
        try:
            print(FernetCipher.generate_key())
        except ImportError:
            print("cryptography isn't installed; `pip install -e '.[memory]'`", file=sys.stderr)
            return 2
        return 0

    try:
        settings = Settings.load(backend=args.backend)
        store = load_memory_store(settings)
    except (ConfigError, MemoryStoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.list:
            return _cmd_list(store)
        if args.show:
            return _cmd_show(store, args.show, limit=args.limit)
        if args.grant:
            c = store.set_consent(args.grant, granted=True, allow_training=args.training)
            print(f"granted consent for {args.grant!r} (allow_training={c.allow_training})")
            return 0
        if args.revoke:
            store.set_consent(args.revoke, granted=False, allow_training=False)
            print(f"revoked consent for {args.revoke!r} (existing data is kept; use --delete to wipe)")
            return 0
        if args.consolidate:
            return asyncio.run(_cmd_consolidate(settings, args.consolidate, args))
        if args.export:
            data = store.export_user(args.export)
            text = json.dumps(data, indent=2, ensure_ascii=False)
            if args.out:
                args.out.write_text(text, encoding="utf-8")
                print(f"exported {args.export!r} -> {args.out}")
            else:
                print(text)
            return 0
        if args.delete:
            if not store.has_user(args.delete):
                print(f"no stored memory for user {args.delete!r}")
                return 0
            if not args.yes:
                reply = input(f"Permanently delete ALL memory for {args.delete!r}? [y/N] ")
                if reply.strip().lower() not in {"y", "yes"}:
                    print("aborted.")
                    return 0
            store.delete_user(args.delete)
            print(f"deleted all memory for {args.delete!r}")
            return 0
        if args.distill:
            return _cmd_distill(settings, store, args.distill, args)
    except MemoryStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # surface backend/model errors without a traceback wall
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

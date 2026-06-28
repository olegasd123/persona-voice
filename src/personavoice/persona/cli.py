"""`personavoice-persona`: author personas from the command line (Feature K).

    personavoice-persona draft "a patient French tutor who only speaks in B1"
    personavoice-persona draft "a blunt PM interviewer" --out config/personas/pm2.yaml
    personavoice-persona draft "..." --user alice --backend cuda

The `draft` verb turns a plain-English description into a validated `Persona` and prints it as
YAML (to stdout, or `--out FILE`). It uses the already-configured cascade LLM (LM Studio on
Mac), so it stays fully offline. The legal `voice.ref` ids and served LoRA names are passed to
the drafter so a draft can't invent an unservable voice/LoRA, and the id is de-duped against the
curated personas (and `--user`'s own). It mirrors the `--check` validation path: what it prints
is guaranteed to load — drop it under `config/personas/` (or save it via `POST /personas`).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import yaml

from ..adapters.factory import build_backend
from ..models import Persona
from ..persona.author import PersonaDraftError, draft_persona
from ..persona.lora import served_loras
from ..persona.registry import PersonaRegistry
from ..server.config import (
    ConfigError,
    Settings,
    load_backend_config,
    load_user_persona_store,
    load_voice_registry,
)


def _default_curated(registry: PersonaRegistry) -> Persona | None:
    """The curated persona whose voice/base-model seed a minimal draft (PERSONAVOICE_PERSONA → first)."""
    default = os.getenv("PERSONAVOICE_PERSONA", "").strip()
    if default in registry:
        return registry.get(default)
    ids = registry.ids()
    return registry.get(ids[0]) if ids else None


async def _draft(settings: Settings, args: argparse.Namespace) -> int:
    registry = PersonaRegistry(settings.personas_dir)
    voices = [f"voices/{vid}" for vid in load_voice_registry(settings).ids()]

    existing = set(registry.ids())
    if args.user:
        existing |= set(load_user_persona_store(settings).ids_for(args.user))

    backend = build_backend(load_backend_config(settings))
    lora_modules = (
        os.getenv("PERSONAVOICE_LORA_MODULES") or os.getenv("VLLM_LORA_MODULES") or ""
    ).strip() or None
    # `supports_lora` is a capability class-attr on LoRA-capable adapters (vLLM); absent on others
    # — read it tolerantly, mirroring the token server's `_active_llm`.
    supports_lora = bool(getattr(backend.llm, "supports_lora", False))
    loras = [o.id for o in served_loras(supports_lora=supports_lora, lora_modules=lora_modules)]

    curated = _default_curated(registry)
    persona = await draft_persona(
        backend.llm,
        args.description,
        voices=voices,
        loras=loras,
        existing_ids=existing,
        default_voice=curated.voice.ref if curated else None,
        default_base_model=curated.llm.base_model if curated else None,
    )

    body = yaml.safe_dump(persona.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    if args.out:
        args.out.write_text(body)
        print(f"wrote persona draft '{persona.id}' -> {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(body)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="personavoice-persona",
        description="Author personas from the command line.",
    )
    parser.add_argument("--backend", choices=("mac", "cuda"), default=None, help="override BACKEND")
    sub = parser.add_subparsers(dest="command", required=True)

    draft_p = sub.add_parser("draft", help="draft a persona from a plain-English description")
    draft_p.add_argument("description", help="e.g. 'a patient French tutor who only speaks in B1'")
    draft_p.add_argument("--out", type=Path, help="write the YAML here instead of stdout")
    draft_p.add_argument("--user", help="also avoid clashing with this user's custom persona ids")
    args = parser.parse_args(argv)

    try:
        settings = Settings.load(backend=args.backend)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        return asyncio.run(_draft(settings, args))
    except (PersonaDraftError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # surface backend/model errors without a traceback wall
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

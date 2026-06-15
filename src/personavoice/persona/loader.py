"""Load persona definitions from YAML into validated `Persona` models."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from ..models import Persona


class PersonaError(ValueError):
    """A persona file is missing, malformed, or fails validation."""


def load_persona(path: str | Path) -> Persona:
    path = Path(path)
    if not path.is_file():
        raise PersonaError(f"persona file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise PersonaError(f"{path}: invalid YAML: {exc}") from exc
    try:
        return Persona.model_validate(raw)
    except ValidationError as exc:
        raise PersonaError(f"{path}: invalid persona: {exc}") from exc


def load_personas(personas_dir: str | Path) -> dict[str, Persona]:
    """Load every `*.yaml` under `personas_dir`, keyed by persona id.

    Raises if two files declare the same id, or a filename's stem disagrees with the
    declared id (a common copy-paste mistake).
    """
    personas_dir = Path(personas_dir)
    if not personas_dir.is_dir():
        raise PersonaError(f"personas dir not found: {personas_dir}")

    out: dict[str, Persona] = {}
    for path in sorted(personas_dir.glob("*.yaml")):
        persona = load_persona(path)
        if persona.id != path.stem:
            raise PersonaError(
                f"{path}: persona id {persona.id!r} does not match filename stem {path.stem!r}"
            )
        if persona.id in out:
            raise PersonaError(f"duplicate persona id {persona.id!r}")
        out[persona.id] = persona
    return out

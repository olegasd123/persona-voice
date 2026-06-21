"""Runtime persona registry with hot-reload (the runtime API is built on top of this)."""

from __future__ import annotations

from pathlib import Path

from ..models import Persona
from .loader import PersonaError, load_personas


class PersonaRegistry:
    """Holds the loaded personas and supports reload-without-restart."""

    def __init__(self, personas_dir: str | Path) -> None:
        self._dir = Path(personas_dir)
        self._personas: dict[str, Persona] = {}
        self.reload()

    def reload(self) -> None:
        """Re-read every persona file from disk (hot-swap)."""
        self._personas = load_personas(self._dir)

    def get(self, persona_id: str) -> Persona:
        try:
            return self._personas[persona_id]
        except KeyError:
            known = ", ".join(sorted(self._personas)) or "(none)"
            raise PersonaError(f"unknown persona {persona_id!r}; known: {known}") from None

    def ids(self) -> list[str]:
        return sorted(self._personas)

    def all(self) -> list[Persona]:
        return [self._personas[i] for i in self.ids()]

    def __len__(self) -> int:
        return len(self._personas)

    def __contains__(self, persona_id: object) -> bool:
        return persona_id in self._personas

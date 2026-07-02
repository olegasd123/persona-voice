"""Multi-user store for user-authored ("custom") personas (N3).

The four curated personas (`config/personas/*.yaml`) are read-only and shared by everyone.
This store holds the personas a *user* creates at runtime, namespaced per `user_id`, and
persists them to a single writable JSON file (`PERSONAVOICE_USER_PERSONAS` →
`user_personas.json`) so they survive a restart and the live agent can pick them up.

It mirrors `voice/clone.py`'s `ClonesStore` (load / record / remove / save), but keyed by
user: `{user_id: {persona_id: Persona}}`. Curated personas always win on id clash and are
never stored here, so a user can't shadow or delete a curated persona (the routes and the
agent's resolver both enforce that — the store just keeps each user's own set isolated).
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .._fsio import atomic_write_text
from ..models import Persona


class UserPersonaError(ValueError):
    """The user-persona manifest is missing data or fails validation."""


class _Manifest(BaseModel):
    """On-disk shape of `user_personas.json`: `{user_id: {persona_id: Persona}}`."""

    model_config = ConfigDict(extra="ignore")

    users: dict[str, dict[str, Persona]] = Field(default_factory=dict)


class UserPersonaStore:
    """Persisted, per-user catalog of custom personas."""

    def __init__(
        self, path: str | Path, users: dict[str, dict[str, Persona]] | None = None
    ) -> None:
        # `path` is the JSON file itself (not a directory) — it's a single shared manifest.
        self._path = Path(path).expanduser()
        self._users: dict[str, dict[str, Persona]] = users or {}

    # -- loading / saving --------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> UserPersonaStore:
        """Load the store from `path` (a missing file → empty store)."""
        path = Path(path).expanduser()
        if not path.is_file():
            return cls(path)
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise UserPersonaError(f"{path}: invalid user-persona manifest: {exc}") from exc
        try:
            manifest = _Manifest.model_validate(raw)
        except ValidationError as exc:
            raise UserPersonaError(f"{path}: invalid user-persona manifest: {exc}") from exc
        return cls(path, {uid: dict(personas) for uid, personas in manifest.users.items()})

    def reload(self) -> None:
        """Re-read the manifest from disk (hot-pickup of personas added since startup)."""
        self._users = UserPersonaStore.load(self._path)._users

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        manifest = _Manifest(users=self._users)
        atomic_write_text(self._path, manifest.model_dump_json(indent=2))

    # -- access ------------------------------------------------------------------------
    @property
    def path(self) -> Path:
        return self._path

    def get(self, user_id: str, persona_id: str) -> Persona | None:
        return self._users.get(user_id, {}).get(persona_id)

    def has(self, user_id: str, persona_id: str) -> bool:
        return persona_id in self._users.get(user_id, {})

    def list_for(self, user_id: str) -> list[Persona]:
        """A user's personas, ordered by id (stable for a picker)."""
        personas = self._users.get(user_id, {})
        return [personas[pid] for pid in sorted(personas)]

    def ids_for(self, user_id: str) -> list[str]:
        return sorted(self._users.get(user_id, {}))

    def users(self) -> list[str]:
        return sorted(self._users)

    # -- mutation ----------------------------------------------------------------------
    def record(self, user_id: str, persona: Persona) -> None:
        """Add or replace a user's persona (keyed by `persona.id`), then persist."""
        if not user_id:
            raise UserPersonaError("a user id is required to store a persona")
        self._users.setdefault(user_id, {})[persona.id] = persona
        self.save()

    def remove(self, user_id: str, persona_id: str) -> bool:
        """Delete a user's persona. Returns True if one was removed (then persists)."""
        personas = self._users.get(user_id)
        if not personas or persona_id not in personas:
            return False
        del personas[persona_id]
        if not personas:
            del self._users[user_id]
        self.save()
        return True

    def __len__(self) -> int:
        return sum(len(p) for p in self._users.values())

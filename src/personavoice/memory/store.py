"""Per-user memory persistence + privacy controls.

The store is the on-disk root for everything the assistant remembers about a user across
sessions. It is intentionally decoupled from the live cascade: the orchestrator records
turns and reads back a profile/recall block, but the store itself only does storage and
the privacy operations the acceptance criteria call for (per-user **delete** and
**export**). Layout under `<memory_dir>`:

    <memory_dir>/
      <user_id>/
        consent.json        # opt-in gate (+ training opt-in)
        transcript.jsonl     # one MemoryTurn per line (append-only)
        profile.json         # rolling summary + durable facts (see profile.py)

**Consent first.** Nothing is recorded for a user until they've granted consent; the
`ConversationMemory` facade enforces that, and `record_turn` refuses without it so the
storage layer can't be bypassed.

**Encryption at rest is optional.** A `Cipher` wraps every byte written: the default
`NullCipher` stores plaintext (a dev convenience, flagged by `server --check`), while
`FernetCipher` (lazy `cryptography`, the `memory` extra, keyed from
`PERSONAVOICE_MEMORY_KEY`) encrypts each JSONL line and each blob independently — so
append stays line-at-a-time and a corrupt/foreign line never poisons the rest.

Heavy crypto is lazy-imported so the core install and the unit tests run without the extra;
the keyword recall path (`rag.py`) needs nothing beyond the standard library either.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..models import Role

_CONSENT_NAME = "consent.json"
_TRANSCRIPT_NAME = "transcript.jsonl"
_PROFILE_NAME = "profile.json"

# A user id becomes a directory name and must stay one path segment. Participant identities
# are usually emails/handles, so allow those characters but never a separator or `..`.
_USER_ID_RE = re.compile(r"[A-Za-z0-9_.@+-]+")


class MemoryStoreError(ValueError):
    """A user id is invalid, consent is missing, or stored data is unreadable.

    (Not named ``MemoryError`` — that's a builtin.)
    """


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _safe_user_id(user_id: str) -> str:
    """Validate `user_id` for use as a directory name, or raise `MemoryStoreError`."""
    uid = (user_id or "").strip()
    if not uid or uid in {".", ".."} or not _USER_ID_RE.fullmatch(uid):
        raise MemoryStoreError(
            f"invalid user id {user_id!r}; use letters, digits, '_.@+-' (no path separators)"
        )
    return uid


# --------------------------------------------------------------------------------------
# Encryption at rest (optional)
# --------------------------------------------------------------------------------------


class Cipher:
    """Protects a single stored string. The default is a no-op (plaintext on disk)."""

    encrypted = False

    def protect(self, plaintext: str) -> str:
        return plaintext

    def reveal(self, stored: str) -> str:
        return stored


class NullCipher(Cipher):
    """Stores plaintext. Used when no `PERSONAVOICE_MEMORY_KEY` is configured."""


class FernetCipher(Cipher):
    """Authenticated encryption via `cryptography`'s Fernet (the `memory` extra).

    Each value is an independent token (urlsafe-base64 ASCII), so a JSONL file is just one
    token per line — append-friendly, and a bad line is isolated.
    """

    encrypted = True

    def __init__(self, key: str | bytes) -> None:
        try:
            from cryptography.fernet import Fernet
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise MemoryStoreError(
                "PERSONAVOICE_MEMORY_KEY is set but `cryptography` isn't installed; "
                "install the memory extra: `pip install -e '.[memory]'`"
            ) from exc
        key_bytes = key.encode() if isinstance(key, str) else key
        try:
            self._fernet = Fernet(key_bytes)
        except (ValueError, TypeError) as exc:
            raise MemoryStoreError(
                "PERSONAVOICE_MEMORY_KEY is not a valid Fernet key; generate one with "
                "`personavoice-memory --gen-key`"
            ) from exc

    @staticmethod
    def generate_key() -> str:
        from cryptography.fernet import Fernet

        return Fernet.generate_key().decode("ascii")

    def protect(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode()).decode("ascii")

    def reveal(self, stored: str) -> str:
        from cryptography.fernet import InvalidToken

        try:
            return self._fernet.decrypt(stored.encode()).decode()
        except InvalidToken as exc:
            raise MemoryStoreError(
                "could not decrypt memory; the PERSONAVOICE_MEMORY_KEY likely doesn't match "
                "the key this data was written with"
            ) from exc


def cipher_from_key(key: str | None) -> Cipher:
    """A `FernetCipher` when a non-empty key is given, else a plaintext `NullCipher`."""
    if key and key.strip():
        return FernetCipher(key.strip())
    return NullCipher()


# --------------------------------------------------------------------------------------
# Stored values
# --------------------------------------------------------------------------------------


class MemoryTurn(BaseModel):
    """One recorded conversation turn (a line in `transcript.jsonl`)."""

    model_config = ConfigDict(extra="ignore")

    session_id: str
    persona_id: str
    role: Role
    content: str
    ts: str = Field(default_factory=_utcnow)


class Consent(BaseModel):
    """A user's recording consent. No `granted` consent → nothing is stored for them."""

    model_config = ConfigDict(extra="ignore")

    user_id: str
    granted: bool = False
    # Separate, stricter opt-in: may this user's transcripts feed a training distill?
    allow_training: bool = False
    updated_at: str = Field(default_factory=_utcnow)
    note: str | None = None


# --------------------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------------------


class MemoryStore:
    """Per-user transcript + profile + consent storage, with optional at-rest encryption."""

    def __init__(self, directory: str | Path, *, cipher: Cipher | None = None) -> None:
        self._dir = Path(directory).expanduser()
        self._cipher = cipher or NullCipher()

    @property
    def dir(self) -> Path:
        return self._dir

    @property
    def encrypted(self) -> bool:
        return self._cipher.encrypted

    def _user_dir(self, user_id: str) -> Path:
        return self._dir / _safe_user_id(user_id)

    # -- consent -----------------------------------------------------------------------
    def get_consent(self, user_id: str) -> Consent:
        """The user's consent record (a missing file → an ungranted default)."""
        path = self._user_dir(user_id) / _CONSENT_NAME
        if not path.is_file():
            return Consent(user_id=_safe_user_id(user_id))
        raw = self._read_blob(path)
        try:
            return Consent.model_validate(raw)
        except ValidationError as exc:
            raise MemoryStoreError(f"{path}: invalid consent record: {exc}") from exc

    def set_consent(
        self, user_id: str, *, granted: bool, allow_training: bool = False, note: str | None = None
    ) -> Consent:
        consent = Consent(
            user_id=_safe_user_id(user_id),
            granted=granted,
            allow_training=allow_training,
            note=note,
        )
        self._write_blob(self._user_dir(user_id) / _CONSENT_NAME, consent.model_dump())
        return consent

    def has_consent(self, user_id: str) -> bool:
        return self.get_consent(user_id).granted

    # -- transcripts -------------------------------------------------------------------
    def record_turn(self, user_id: str, turn: MemoryTurn) -> None:
        """Append one turn for `user_id`. Refuses unless the user has granted consent."""
        if not self.has_consent(user_id):
            raise MemoryStoreError(
                f"user {user_id!r} has not granted recording consent; nothing was stored"
            )
        path = self._user_dir(user_id) / _TRANSCRIPT_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        line = self._cipher.protect(turn.model_dump_json())
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def read_turns(
        self,
        user_id: str,
        *,
        persona_id: str | None = None,
        session_id: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryTurn]:
        """Read stored turns oldest-first, optionally filtered; `limit` keeps the last N."""
        turns = [
            t
            for t in self._iter_turns(user_id)
            if (persona_id is None or t.persona_id == persona_id)
            and (session_id is None or t.session_id == session_id)
        ]
        if limit is not None and limit >= 0:
            turns = turns[-limit:]
        return turns

    def _iter_turns(self, user_id: str) -> Iterator[MemoryTurn]:
        path = self._user_dir(user_id) / _TRANSCRIPT_NAME
        if not path.is_file():
            return
        for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                yield MemoryTurn.model_validate_json(self._cipher.reveal(raw))
            except ValidationError as exc:
                raise MemoryStoreError(f"{path}:{lineno}: invalid transcript line: {exc}") from exc

    def session_ids(self, user_id: str) -> list[str]:
        """Distinct session ids for a user, in first-seen order."""
        seen: dict[str, None] = {}
        for t in self._iter_turns(user_id):
            seen.setdefault(t.session_id, None)
        return list(seen)

    # -- profile -----------------------------------------------------------------------
    def load_profile_raw(self, user_id: str) -> dict[str, Any] | None:
        """Raw profile dict (parsed by `profile.py`), or None if none stored yet."""
        path = self._user_dir(user_id) / _PROFILE_NAME
        if not path.is_file():
            return None
        return self._read_blob(path)

    def save_profile_raw(self, user_id: str, data: dict[str, Any]) -> None:
        self._write_blob(self._user_dir(user_id) / _PROFILE_NAME, data)

    # -- users / privacy ---------------------------------------------------------------
    def users(self) -> list[str]:
        """Every user that has a directory under the store, sorted."""
        if not self._dir.is_dir():
            return []
        return sorted(p.name for p in self._dir.iterdir() if p.is_dir())

    def has_user(self, user_id: str) -> bool:
        return self._user_dir(user_id).is_dir()

    def delete_user(self, user_id: str) -> bool:
        """Privacy: wipe everything stored for a user. Returns True if anything was removed."""
        path = self._user_dir(user_id)
        if not path.is_dir():
            return False
        shutil.rmtree(path)
        return True

    def export_user(self, user_id: str) -> dict[str, Any]:
        """Privacy: a portable, decrypted dump of everything stored for a user."""
        uid = _safe_user_id(user_id)
        if not self.has_user(uid):
            raise MemoryStoreError(f"no stored memory for user {user_id!r}")
        return {
            "user_id": uid,
            "exported_at": _utcnow(),
            "consent": self.get_consent(uid).model_dump(),
            "profile": self.load_profile_raw(uid),
            "turns": [t.model_dump() for t in self._iter_turns(uid)],
        }

    # -- blob helpers ------------------------------------------------------------------
    def _read_blob(self, path: Path) -> dict[str, Any]:
        try:
            revealed = self._cipher.reveal(path.read_text(encoding="utf-8"))
            return json.loads(revealed)
        except json.JSONDecodeError as exc:
            raise MemoryStoreError(f"{path}: invalid JSON: {exc}") from exc

    def _write_blob(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._cipher.protect(json.dumps(data, indent=2)), encoding="utf-8")

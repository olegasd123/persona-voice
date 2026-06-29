"""The multi-user custom-persona store: CRUD, per-user isolation, persistence, reload."""

from __future__ import annotations

from pathlib import Path

import pytest

from personavoice.models import Persona
from personavoice.persona.store import UserPersonaError, UserPersonaStore


def _persona(persona_id: str, name: str | None = None) -> Persona:
    return Persona.model_validate(
        {
            "id": persona_id,
            "name": name or persona_id.title(),
            "system_prompt": "You are a test persona.",
            "llm": {"base_model": "qwen2.5-7b-instruct"},
            "voice": {"ref": "voices/companion_soft"},
        }
    )


def test_record_get_and_list(tmp_path: Path) -> None:
    store = UserPersonaStore(tmp_path / "u.json")
    store.record("alice", _persona("tutor"))
    store.record("alice", _persona("coach"))
    assert store.get("alice", "tutor").id == "tutor"
    assert store.ids_for("alice") == ["coach", "tutor"]  # sorted
    assert [p.id for p in store.list_for("alice")] == ["coach", "tutor"]
    assert len(store) == 2


def test_per_user_isolation(tmp_path: Path) -> None:
    store = UserPersonaStore(tmp_path / "u.json")
    store.record("alice", _persona("tutor"))
    store.record("bob", _persona("tutor"))  # same id, different user — independent
    assert store.get("alice", "tutor") is not None
    assert store.get("bob", "tutor") is not None
    assert store.remove("alice", "tutor") is True
    # Bob's same-named persona is untouched.
    assert store.get("alice", "tutor") is None
    assert store.get("bob", "tutor") is not None


def test_remove_unknown_is_false(tmp_path: Path) -> None:
    store = UserPersonaStore(tmp_path / "u.json")
    assert store.remove("alice", "ghost") is False
    store.record("alice", _persona("tutor"))
    assert store.remove("alice", "ghost") is False  # unknown id for a known user


def test_remove_prunes_empty_user(tmp_path: Path) -> None:
    store = UserPersonaStore(tmp_path / "u.json")
    store.record("alice", _persona("tutor"))
    store.remove("alice", "tutor")
    assert store.users() == []


def test_record_requires_user(tmp_path: Path) -> None:
    store = UserPersonaStore(tmp_path / "u.json")
    with pytest.raises(UserPersonaError, match="user id is required"):
        store.record("", _persona("tutor"))


def test_persistence_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "u.json"
    store = UserPersonaStore(path)
    store.record("alice", _persona("tutor", name="French Tutor"))
    # A fresh load sees the persisted persona with its fields intact.
    reloaded = UserPersonaStore.load(path)
    p = reloaded.get("alice", "tutor")
    assert p is not None and p.name == "French Tutor"
    assert reloaded.has("alice", "tutor")


def test_load_missing_file_is_empty(tmp_path: Path) -> None:
    store = UserPersonaStore.load(tmp_path / "nope.json")
    assert len(store) == 0 and store.users() == []


def test_load_rejects_corrupt_manifest(tmp_path: Path) -> None:
    path = tmp_path / "u.json"
    path.write_text("{not json}")
    with pytest.raises(UserPersonaError, match="invalid user-persona manifest"):
        UserPersonaStore.load(path)


def test_reload_picks_up_external_writes(tmp_path: Path) -> None:
    path = tmp_path / "u.json"
    store = UserPersonaStore(path)
    store.record("alice", _persona("tutor"))
    # A second writer (e.g. the token server process) adds a persona to the same file.
    other = UserPersonaStore.load(path)
    other.record("alice", _persona("coach"))
    assert store.get("alice", "coach") is None  # not seen yet
    store.reload()
    assert store.get("alice", "coach") is not None

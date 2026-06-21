"""MemoryStore: consent gating, transcripts, sessions, profile, privacy, encryption."""

from __future__ import annotations

from pathlib import Path

import pytest

from personavoice.memory import (
    Consent,
    MemoryStore,
    MemoryStoreError,
    MemoryTurn,
    NullCipher,
    cipher_from_key,
)
from personavoice.models import Role


def _turn(
    content: str, *, session: str = "s1", persona: str = "companion", role: Role = Role.user
) -> MemoryTurn:
    return MemoryTurn(session_id=session, persona_id=persona, role=role, content=content)


# --------------------------------------------------------------------------------------
# consent gating
# --------------------------------------------------------------------------------------


def test_record_refused_without_consent(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    with pytest.raises(MemoryStoreError, match="consent"):
        store.record_turn("alice", _turn("hello"))


def test_default_consent_is_ungranted(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    consent = store.get_consent("alice")
    assert isinstance(consent, Consent)
    assert consent.granted is False
    assert store.has_consent("alice") is False


def test_set_and_read_consent(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.set_consent("alice", granted=True, allow_training=True, note="opted in")
    consent = store.get_consent("alice")
    assert consent.granted and consent.allow_training
    assert consent.note == "opted in"
    # Survives a reopen.
    assert MemoryStore(tmp_path).has_consent("alice")


# --------------------------------------------------------------------------------------
# transcripts + sessions
# --------------------------------------------------------------------------------------


def test_record_and_read_turns(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.set_consent("alice", granted=True)
    store.record_turn("alice", _turn("my name is Sam", role=Role.user))
    store.record_turn("alice", _turn("nice to meet you Sam", role=Role.assistant))

    turns = store.read_turns("alice")
    assert [t.content for t in turns] == ["my name is Sam", "nice to meet you Sam"]
    assert turns[0].role is Role.user and turns[1].role is Role.assistant


def test_read_turns_filters_and_limit(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.set_consent("alice", granted=True)
    store.record_turn("alice", _turn("a", session="s1", persona="companion"))
    store.record_turn("alice", _turn("b", session="s2", persona="hr_interviewer"))
    store.record_turn("alice", _turn("c", session="s2", persona="hr_interviewer"))

    assert [t.content for t in store.read_turns("alice", session_id="s2")] == ["b", "c"]
    assert [t.content for t in store.read_turns("alice", persona_id="companion")] == ["a"]
    assert [t.content for t in store.read_turns("alice", limit=1)] == ["c"]
    assert store.session_ids("alice") == ["s1", "s2"]


def test_read_turns_empty_for_unknown_user(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    assert store.read_turns("nobody") == []
    assert store.session_ids("nobody") == []


# --------------------------------------------------------------------------------------
# profile
# --------------------------------------------------------------------------------------


def test_profile_roundtrip(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    assert store.load_profile_raw("alice") is None
    store.save_profile_raw("alice", {"user_id": "alice", "summary": "likes jazz", "facts": []})
    assert store.load_profile_raw("alice") == {
        "user_id": "alice",
        "summary": "likes jazz",
        "facts": [],
    }


# --------------------------------------------------------------------------------------
# privacy: users / delete / export
# --------------------------------------------------------------------------------------


def test_users_listing(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.set_consent("bob", granted=True)
    store.set_consent("alice", granted=True)
    assert store.users() == ["alice", "bob"]


def test_delete_user_wipes_everything(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.set_consent("alice", granted=True)
    store.record_turn("alice", _turn("secret"))
    store.save_profile_raw("alice", {"user_id": "alice", "summary": "x", "facts": []})

    assert store.has_user("alice")
    assert store.delete_user("alice") is True
    assert not store.has_user("alice")
    assert store.read_turns("alice") == []
    assert store.delete_user("alice") is False  # already gone


def test_export_user_dumps_all(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.set_consent("alice", granted=True, allow_training=True)
    store.record_turn("alice", _turn("hi there"))
    store.save_profile_raw("alice", {"user_id": "alice", "summary": "s", "facts": []})

    dump = store.export_user("alice")
    assert dump["user_id"] == "alice"
    assert dump["consent"]["granted"] is True
    assert dump["profile"]["summary"] == "s"
    assert [t["content"] for t in dump["turns"]] == ["hi there"]


def test_export_unknown_user_raises(tmp_path: Path) -> None:
    with pytest.raises(MemoryStoreError, match="no stored memory"):
        MemoryStore(tmp_path).export_user("ghost")


# --------------------------------------------------------------------------------------
# invalid user ids
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "  ", "..", "a/b", "a\\b", "../etc"])
def test_invalid_user_ids_rejected(tmp_path: Path, bad: str) -> None:
    store = MemoryStore(tmp_path)
    with pytest.raises(MemoryStoreError, match="invalid user id"):
        store.get_consent(bad)


def test_valid_email_like_user_id(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.set_consent("sam.doe@example.com", granted=True)
    store.record_turn("sam.doe@example.com", _turn("hi"))
    assert store.read_turns("sam.doe@example.com")[0].content == "hi"


# --------------------------------------------------------------------------------------
# corrupt data tolerance
# --------------------------------------------------------------------------------------


def test_corrupt_transcript_line_raises(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.set_consent("alice", granted=True)
    store.record_turn("alice", _turn("ok"))
    (tmp_path / "alice" / "transcript.jsonl").write_text('{"not":"a turn"}\n')
    with pytest.raises(MemoryStoreError, match="invalid transcript line"):
        store.read_turns("alice")


# --------------------------------------------------------------------------------------
# encryption at rest
# --------------------------------------------------------------------------------------


def test_cipher_from_key_defaults_to_plaintext() -> None:
    assert isinstance(cipher_from_key(None), NullCipher)
    assert isinstance(cipher_from_key(""), NullCipher)
    assert cipher_from_key(None).encrypted is False


def test_encrypted_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    from personavoice.memory import FernetCipher

    key = FernetCipher.generate_key()
    store = MemoryStore(tmp_path, cipher=FernetCipher(key))
    assert store.encrypted is True
    store.set_consent("alice", granted=True)
    store.record_turn("alice", _turn("encrypt me"))

    # Bytes on disk must not contain the plaintext.
    raw = (tmp_path / "alice" / "transcript.jsonl").read_text()
    assert "encrypt me" not in raw

    # A store with the same key reads it back; a different key fails loudly.
    assert MemoryStore(tmp_path, cipher=FernetCipher(key)).read_turns("alice")[0].content == (
        "encrypt me"
    )
    with pytest.raises(MemoryStoreError, match="could not decrypt"):
        MemoryStore(tmp_path, cipher=FernetCipher(FernetCipher.generate_key())).read_turns("alice")


def test_bad_fernet_key_raises() -> None:
    pytest.importorskip("cryptography")
    from personavoice.memory import FernetCipher

    with pytest.raises(MemoryStoreError, match="valid Fernet key"):
        FernetCipher("not-a-real-key")

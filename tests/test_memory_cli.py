"""personavoice-memory CLI (M8): consent, list/show, export, delete, distill, gen-key."""

from __future__ import annotations

from pathlib import Path

import pytest

from personavoice.memory import MemoryStore, MemoryTurn
from personavoice.memory.cli import main
from personavoice.models import Role


@pytest.fixture
def mem_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the CLI's store at a temp dir (plaintext) and return that dir."""
    monkeypatch.setenv("PERSONAVOICE_MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("PERSONAVOICE_MEMORY_KEY", "")
    monkeypatch.setenv("BACKEND", "mac")
    return tmp_path


def _store(d: Path) -> MemoryStore:
    return MemoryStore(d)


def test_grant_and_list(mem_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--grant", "alice", "--training", "--backend", "mac"]) == 0
    consent = _store(mem_env).get_consent("alice")
    assert consent.granted and consent.allow_training

    assert main(["--list", "--backend", "mac"]) == 0
    out = capsys.readouterr().out
    assert "alice" in out
    assert "training" in out


def test_revoke(mem_env: Path) -> None:
    store = _store(mem_env)
    store.set_consent("alice", granted=True, allow_training=True)
    assert main(["--revoke", "alice", "--backend", "mac"]) == 0
    assert _store(mem_env).get_consent("alice").granted is False


def test_show(mem_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = _store(mem_env)
    store.set_consent("alice", granted=True)
    store.record_turn("alice", MemoryTurn(session_id="s1", persona_id="companion", role=Role.user, content="hi friend"))
    store.save_profile_raw("alice", {"user_id": "alice", "summary": "A friend.", "facts": [{"text": "likes tea"}]})

    assert main(["--show", "alice", "--backend", "mac"]) == 0
    out = capsys.readouterr().out
    assert "A friend." in out
    assert "likes tea" in out
    assert "hi friend" in out


def test_show_unknown_user(mem_env: Path) -> None:
    assert main(["--show", "ghost", "--backend", "mac"]) == 1


def test_export_to_file(mem_env: Path, tmp_path: Path) -> None:
    store = _store(mem_env)
    store.set_consent("alice", granted=True)
    store.record_turn("alice", MemoryTurn(session_id="s1", persona_id="companion", role=Role.user, content="hello"))
    out = tmp_path / "dump.json"
    assert main(["--export", "alice", "--out", str(out), "--backend", "mac"]) == 0
    assert "hello" in out.read_text()


def test_delete_with_yes(mem_env: Path) -> None:
    store = _store(mem_env)
    store.set_consent("alice", granted=True)
    store.record_turn("alice", MemoryTurn(session_id="s1", persona_id="companion", role=Role.user, content="x"))
    assert store.has_user("alice")
    assert main(["--delete", "alice", "--yes", "--backend", "mac"]) == 0
    assert not _store(mem_env).has_user("alice")


def test_delete_unknown_user_is_noop(mem_env: Path) -> None:
    assert main(["--delete", "ghost", "--backend", "mac"]) == 0


def test_distill_requires_optin(mem_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = _store(mem_env)
    store.set_consent("alice", granted=True, allow_training=False)
    store.record_turn("alice", MemoryTurn(session_id="s1", persona_id="companion", role=Role.user, content="hi"))
    store.record_turn("alice", MemoryTurn(session_id="s1", persona_id="companion", role=Role.assistant, content="hello"))
    rc = main(["--distill", "alice", "--backend", "mac"])
    assert rc == 2
    assert "opted in to training" in capsys.readouterr().err


def test_distill_writes_dataset(mem_env: Path, tmp_path: Path) -> None:
    store = _store(mem_env)
    store.set_consent("alice", granted=True, allow_training=True)
    store.record_turn("alice", MemoryTurn(session_id="s1", persona_id="companion", role=Role.user, content="hi"))
    store.record_turn("alice", MemoryTurn(session_id="s1", persona_id="companion", role=Role.assistant, content="hello"))
    out = tmp_path / "alice.jsonl"
    assert main(["--distill", "alice", "--out", str(out), "--backend", "mac"]) == 0
    assert out.is_file()
    assert "hello" in out.read_text()


def test_gen_key(capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("cryptography")
    from personavoice.memory import FernetCipher

    assert main(["--gen-key"]) == 0
    key = capsys.readouterr().out.strip()
    # The printed key must be usable.
    FernetCipher(key)

"""The cross-process session heartbeat (`obs/sessions.py`).

Pure filesystem/JSON — no worker, no Prometheus. Covers path resolution, the write→read round trip,
the freshness guard, malformed/absent tolerance, and the heartbeat-first `read_live_sessions`
fallback to the Prometheus gauge.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from personavoice.obs import sessions


def test_path_prefers_explicit_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PERSONAVOICE_SESSIONS_FILE", str(tmp_path / "s.json"))
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path / "metrics"))
    assert sessions.sessions_file_path() == tmp_path / "s.json"


def test_path_defaults_into_multiproc_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("PERSONAVOICE_SESSIONS_FILE", raising=False)
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    assert sessions.sessions_file_path() == tmp_path / "sessions.json"


def test_path_none_when_nothing_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PERSONAVOICE_SESSIONS_FILE", raising=False)
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    assert sessions.sessions_file_path() is None


def test_write_read_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    sessions.write_sessions(1, 2, path=path)
    assert sessions.read_sessions_file(path=path) == (1, 2)
    # Atomic write leaves no temp file behind.
    assert not (tmp_path / "sessions.json.tmp").exists()
    body = json.loads(path.read_text())
    assert body["active"] == 1 and body["capacity"] == 2 and isinstance(body["ts"], (int, float))


def test_write_creates_parent_dir(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "sessions.json"
    sessions.write_sessions(0, 1, path=path)
    assert sessions.read_sessions_file(path=path) == (0, 1)


def test_write_is_noop_without_a_configured_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PERSONAVOICE_SESSIONS_FILE", raising=False)
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    sessions.write_sessions(1, 1)  # must not raise
    assert sessions.read_sessions_file() is None


def test_read_absent_file_is_none(tmp_path: Path) -> None:
    assert sessions.read_sessions_file(path=tmp_path / "nope.json") is None


def test_read_stale_file_is_none(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    path.write_text(json.dumps({"active": 1, "capacity": 1, "ts": time.time() - 3600}))
    assert sessions.read_sessions_file(path=path) is None
    # A generous max age accepts the same file.
    assert sessions.read_sessions_file(path=path, max_age_s=7200) == (1, 1)


def test_read_future_ts_is_not_stale(tmp_path: Path) -> None:
    # Modest host/container clock skew (a `ts` slightly in the future) must not read as stale.
    path = tmp_path / "sessions.json"
    path.write_text(json.dumps({"active": 2, "capacity": 3, "ts": time.time() + 5}))
    assert sessions.read_sessions_file(path=path) == (2, 3)


@pytest.mark.parametrize(
    "raw",
    ["not json", "[]", "{}", '{"active": 1}', '{"active": "x", "capacity": 1, "ts": 0}'],
)
def test_read_malformed_file_is_none(tmp_path: Path, raw: str) -> None:
    path = tmp_path / "sessions.json"
    path.write_text(raw)
    # A fresh-enough but malformed file is treated as "unknown", never crashes the reader.
    assert sessions.read_sessions_file(path=path, max_age_s=0) is None


def test_read_live_prefers_file_over_gauge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PERSONAVOICE_SESSIONS_FILE", str(tmp_path / "sessions.json"))
    monkeypatch.setattr(sessions, "read_sessions", lambda: (9, 9))  # gauge disagrees
    sessions.write_sessions(1, 4)
    assert sessions.read_live_sessions() == (1, 4)


def test_read_live_falls_back_to_gauge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # No heartbeat file (empty dir) → fall back to the Prometheus gauge reader.
    monkeypatch.setenv("PERSONAVOICE_SESSIONS_FILE", str(tmp_path / "sessions.json"))
    monkeypatch.setattr(sessions, "read_sessions", lambda: (2, 3))
    assert sessions.read_live_sessions() == (2, 3)


def test_read_live_none_when_nothing_known(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PERSONAVOICE_SESSIONS_FILE", raising=False)
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    monkeypatch.setattr(sessions, "read_sessions", lambda: None)
    assert sessions.read_live_sessions() is None

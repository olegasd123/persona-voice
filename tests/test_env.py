"""Env interpolation used by the backend config loader."""

from __future__ import annotations

from personavoice._env import expand_env_vars


def test_expands_set_var(monkeypatch) -> None:
    monkeypatch.setenv("FOO", "bar")
    assert expand_env_vars("x=${FOO}") == "x=bar"


def test_uses_default_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("MISSING", raising=False)
    assert expand_env_vars("u=${MISSING:-http://localhost:1234}") == "u=http://localhost:1234"


def test_set_var_overrides_default(monkeypatch) -> None:
    monkeypatch.setenv("URL", "http://real")
    assert expand_env_vars("${URL:-http://default}") == "http://real"


def test_unset_no_default_is_empty(monkeypatch) -> None:
    monkeypatch.delenv("NOPE", raising=False)
    assert expand_env_vars("a${NOPE}b") == "ab"

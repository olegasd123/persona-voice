"""The `personavoice.server` CLI dispatch (--check covered in test_check; here: --serve)."""

from __future__ import annotations

import pytest

import personavoice.orchestrator.agent as agent
from personavoice.server import __main__ as cli


def test_serve_dispatches_to_the_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, bool] = {}
    monkeypatch.setattr(agent, "run", lambda: called.setdefault("ran", True))

    rc = cli.main(["--serve", "--backend", "mac"])
    assert rc == 0
    assert called.get("ran") is True


def test_serve_without_livekit_extra_returns_2(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> None:
        raise RuntimeError("livekit is not installed; install the streaming extra")

    monkeypatch.setattr(agent, "run", boom)
    rc = cli.main(["--serve", "--backend", "mac"])
    assert rc == 2


def test_token_server_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    from personavoice.server import token_server

    called: dict[str, bool] = {}
    monkeypatch.setattr(token_server, "run", lambda _settings=None: called.setdefault("ran", True))

    rc = cli.main(["--token-server", "--backend", "mac"])
    assert rc == 0
    assert called.get("ran") is True


def test_no_subcommand_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["--backend", "mac"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "--check" in out and "--serve" in out

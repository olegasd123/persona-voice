"""The `personavoice-demo` CLI, wired to a fake backend (no models/network)."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from personavoice.orchestrator import demo

from .fakes import make_backend

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_tiny_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)  # 0.1s of silence


@pytest.fixture(autouse=True)
def _config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Make persona/backend config resolution independent of the working directory.
    monkeypatch.setenv("PERSONAVOICE_CONFIG_DIR", str(REPO_ROOT / "config"))
    monkeypatch.setenv("BACKEND", "mac")


def test_demo_runs_a_turn_and_writes_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    _write_tiny_wav(in_wav)

    backend = make_backend(stt_text="hello", llm_reply="hi friend")
    monkeypatch.setattr(demo, "build_backend", lambda cfg: backend)

    rc = demo.main(["--wav", str(in_wav), "--persona", "companion", "--out", str(out_wav)])

    assert rc == 0
    assert out_wav.read_bytes() == b"RIFF" + b"hi friend"
    out = capsys.readouterr().out
    assert "hi friend" in out
    assert "timings" in out


def test_demo_persona_by_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    _write_tiny_wav(in_wav)
    monkeypatch.setattr(demo, "build_backend", lambda cfg: make_backend(llm_reply="ok"))

    persona_path = REPO_ROOT / "config" / "personas" / "language_teacher.yaml"
    rc = demo.main(["--wav", str(in_wav), "--persona", str(persona_path), "--out", str(out_wav)])
    assert rc == 0
    assert out_wav.read_bytes() == b"RIFF" + b"ok"


def test_demo_missing_wav_returns_2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(demo, "build_backend", lambda cfg: make_backend())
    rc = demo.main(["--wav", str(tmp_path / "nope.wav"), "--persona", "companion"])
    assert rc == 2


def test_demo_unknown_persona_returns_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    in_wav = tmp_path / "in.wav"
    _write_tiny_wav(in_wav)
    monkeypatch.setattr(demo, "build_backend", lambda cfg: make_backend())
    rc = demo.main(["--wav", str(in_wav), "--persona", "does_not_exist"])
    assert rc == 2


def test_session_option_args_parse() -> None:
    import argparse

    from personavoice.models import CEFRLevel, Demeanor

    parser = argparse.ArgumentParser()
    demo.add_session_option_args(parser)
    args = parser.parse_args(["--voice", "libby", "--cefr", "b1", "--demeanor", "RUDE"])
    opts = demo.session_options_from_args(args)
    assert opts.voice == "libby"
    assert opts.cefr is CEFRLevel.b1
    assert opts.demeanor is Demeanor.rude


def test_demo_session_flags_thread_into_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    _write_tiny_wav(in_wav)
    backend = make_backend(stt_text="hello", llm_reply="ok")
    monkeypatch.setattr(demo, "build_backend", lambda cfg: backend)

    rc = demo.main(
        [
            "--wav",
            str(in_wav),
            "--persona",
            "companion",
            "--out",
            str(out_wav),
            "--demeanor",
            "rude",
            "--cefr",
            "b1",
        ]
    )
    assert rc == 0
    system_prompt = backend.llm.last_messages[0].content  # type: ignore[index]
    assert "brusque" in system_prompt  # rude directive reached the model
    assert "B1" in system_prompt  # cefr directive too

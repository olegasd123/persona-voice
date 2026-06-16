"""The `personavoice-stream-demo` CLI, wired to a fake backend (no models/network)."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from personavoice.orchestrator import stream_demo

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
    monkeypatch.setenv("PERSONAVOICE_CONFIG_DIR", str(REPO_ROOT / "config"))
    monkeypatch.setenv("BACKEND", "mac")


def test_stream_demo_writes_one_wav_per_sentence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    in_wav = tmp_path / "in.wav"
    out_dir = tmp_path / "stream"
    _write_tiny_wav(in_wav)

    backend = make_backend(stt_text="hi", llm_reply="Hello there. How are you?")
    monkeypatch.setattr(stream_demo, "build_backend", lambda cfg: backend)

    rc = stream_demo.main(
        ["--wav", str(in_wav), "--persona", "companion", "--out-dir", str(out_dir)]
    )

    assert rc == 0
    # One wav per streamed sentence.
    wavs = sorted(out_dir.glob("reply_*.wav"))
    assert [w.read_bytes() for w in wavs] == [
        b"RIFF" + b"Hello there.",
        b"RIFF" + b"How are you?",
    ]
    out = capsys.readouterr().out
    assert "first_audio" in out
    assert "Hello there. How are you?" in out


def test_stream_demo_missing_wav_returns_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(stream_demo, "build_backend", lambda cfg: make_backend())
    rc = stream_demo.main(["--wav", str(tmp_path / "nope.wav"), "--persona", "companion"])
    assert rc == 2

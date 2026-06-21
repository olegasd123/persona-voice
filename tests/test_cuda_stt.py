"""CUDA STT adapters: pure result-parsing helpers (no model weights, no GPU)."""

from __future__ import annotations

from dataclasses import dataclass

from personavoice.adapters.stt.faster_whisper import FasterWhisperSTT, _join_segments
from personavoice.adapters.stt.parakeet import ParakeetSTT, _first_text


@dataclass
class _Seg:
    text: str


@dataclass
class _Hyp:
    text: str


def test_faster_whisper_flags() -> None:
    adapter = FasterWhisperSTT(model="large-v3-turbo")
    assert adapter.name == "faster_whisper"
    assert adapter.implemented is True
    assert adapter.check().ok


def test_join_segments_concatenates_text() -> None:
    segments = iter([_Seg(" Hello"), _Seg(" there"), _Seg(".")])
    assert _join_segments(segments) == " Hello there."
    assert _join_segments(iter([])) == ""


def test_parakeet_flags() -> None:
    adapter = ParakeetSTT()
    assert adapter.name == "parakeet"
    assert adapter.implemented is True
    assert adapter.check().ok


def test_first_text_handles_nemo_return_shapes() -> None:
    # list[str]
    assert _first_text(["hello world"]) == "hello world"
    # list[Hypothesis]
    assert _first_text([_Hyp("from hyp")]) == "from hyp"
    # (best, all) tuple of lists
    assert _first_text(([_Hyp("best one")], [_Hyp("best one"), _Hyp("alt")])) == "best one"
    # empty
    assert _first_text([]) == ""
    assert _first_text(()) == ""

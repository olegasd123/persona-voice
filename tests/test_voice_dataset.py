"""M9 target-speaker dataset: parse/validate/metadata + STT auto-transcription."""

from __future__ import annotations

from pathlib import Path

import pytest

from personavoice.training.voice.dataset import (
    DatasetError,
    SpeakerClip,
    metadata_line,
    parse_metadata_line,
    read_metadata,
    transcribe_clips,
    validate_clip,
    validate_dataset,
    write_metadata,
)


def _clips(n: int, *, duration: float | None = None) -> list[SpeakerClip]:
    return [
        SpeakerClip(audio_path=f"wavs/clip_{i:04d}.wav", text=f"line number {i}", duration=duration)
        for i in range(n)
    ]


def test_metadata_line_collapses_whitespace() -> None:
    line = metadata_line(SpeakerClip(audio_path="wavs/a.wav", text="hello\n  there  world"))
    assert line == "wavs/a.wav|hello there world"


def test_parse_metadata_line_roundtrip() -> None:
    clip = parse_metadata_line("wavs/a.wav|hello there")
    assert clip.audio_path == "wavs/a.wav" and clip.text == "hello there"


def test_parse_metadata_line_keeps_text_pipes() -> None:
    # Only the first '|' separates path from text.
    clip = parse_metadata_line("wavs/a.wav|a | b | c")
    assert clip.text == "a | b | c"


@pytest.mark.parametrize("bad", ["no pipe here", "|missing path", "wavs/a.wav|"])
def test_parse_metadata_line_rejects_malformed(bad: str) -> None:
    with pytest.raises(DatasetError):
        parse_metadata_line(bad)


def test_metadata_roundtrip_file(tmp_path: Path) -> None:
    clips = _clips(3)
    out = write_metadata(tmp_path / "metadata.csv", clips)
    again = read_metadata(out)
    assert [c.audio_path for c in again] == [c.audio_path for c in clips]
    assert [c.text for c in again] == [c.text for c in clips]


def test_read_metadata_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text("wavs/a.wav|one\n\n  \nwavs/b.wav|two\n", encoding="utf-8")
    assert len(read_metadata(path)) == 2


def test_read_metadata_missing_file() -> None:
    with pytest.raises(DatasetError, match="not found"):
        read_metadata("/nonexistent/metadata.csv")


def test_validate_clip_rejects_empty_text() -> None:
    with pytest.raises(DatasetError, match="no transcript"):
        validate_clip(SpeakerClip(audio_path="a.wav", text="   "))


def test_validate_clip_bounds_duration() -> None:
    with pytest.raises(DatasetError, match="too short"):
        validate_clip(SpeakerClip(audio_path="a.wav", text="hi", duration=0.2))
    with pytest.raises(DatasetError, match="too long"):
        validate_clip(SpeakerClip(audio_path="a.wav", text="hi", duration=99.0))


def test_validate_dataset_reports_stats() -> None:
    stats = validate_dataset(_clips(12, duration=15.0))
    assert stats.num_clips == 12
    assert stats.total_seconds == pytest.approx(180.0)
    assert stats.known_durations == 12
    assert stats.min_seconds == 15.0


def test_validate_dataset_rejects_too_few_clips() -> None:
    with pytest.raises(DatasetError, match="too few clips"):
        validate_dataset(_clips(3))


def test_validate_dataset_rejects_too_little_audio() -> None:
    # Enough clips, but only when durations are fully known is the minutes floor enforced.
    with pytest.raises(DatasetError, match="want at least"):
        validate_dataset(_clips(12, duration=2.0))  # 24s total < 120s


def test_validate_dataset_skips_minutes_floor_without_durations() -> None:
    # No durations known → clip-count check still applies, minutes floor doesn't.
    stats = validate_dataset(_clips(12))
    assert stats.total_seconds == 0.0 and stats.known_durations == 0


def test_validate_dataset_empty() -> None:
    with pytest.raises(DatasetError, match="empty dataset"):
        validate_dataset([])


# --------------------------------------------------------------------------------------
# transcribe_clips (injected STT)
# --------------------------------------------------------------------------------------


class _FakeSTT:
    def __init__(self, text: str = "spoken words") -> None:
        self._text = text

    async def transcribe(self, audio: bytes):
        from personavoice.models import Transcript

        # Empty bytes → empty transcript (so the clip is dropped).
        return Transcript(text="" if not audio else self._text, is_final=True)


async def test_transcribe_clips_labels_and_drops_empty(tmp_path: Path) -> None:
    good = tmp_path / "a.wav"
    good.write_bytes(b"RIFFdata")
    empty = tmp_path / "b.wav"
    empty.write_bytes(b"")

    clips = await transcribe_clips(_FakeSTT("hello world"), [good, empty])
    assert len(clips) == 1  # the empty-transcript clip is dropped
    assert clips[0].text == "hello world"
    assert clips[0].audio_path == str(good)

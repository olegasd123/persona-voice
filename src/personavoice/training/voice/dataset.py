"""Target-speaker dataset for voice fine-tuning.

A voice fine-tune learns one speaker from a small set of **(clip, transcript)** pairs. The
shared, human-readable manifest is a `metadata.csv` of pipe-separated `audio_path|text` lines
(the `audio_file|text` metadata shape), which the trainer consumes.

Everything here is pure and unit-tested except `transcribe_clips`, which — like curation's
self-chat — takes an injected STT adapter to auto-fill missing transcripts. Reading and
writing are UTF-8 (the Windows-cp1252 lesson from the persona-LoRA datasets).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

# Per-clip clear-speech bounds (seconds). Very short clips carry little timbre; very long ones
# bloat training and often hold pauses/noise. A few seconds each is the sweet spot.
_MIN_CLIP_SECONDS = 1.0
_MAX_CLIP_SECONDS = 30.0
# A usable fine-tune wants at least a few minutes of the target speaker.
_MIN_TOTAL_SECONDS = 120.0
_MIN_CLIPS = 10


class DatasetError(ValueError):
    """A clip is unusable, the manifest is malformed, or the dataset is too small."""


class SpeakerClip(BaseModel):
    """One labeled clip of the target speaker: an audio path + its transcript."""

    model_config = ConfigDict(extra="ignore")

    audio_path: str
    text: str
    duration: float | None = None  # seconds, when known (filled by `probe_durations`)


class DatasetStats(BaseModel):
    """Summary of a target-speaker dataset (what `validate_dataset` reports)."""

    num_clips: int
    total_seconds: float  # 0.0 when no durations are known
    known_durations: int
    min_seconds: float | None = None
    max_seconds: float | None = None


def metadata_line(clip: SpeakerClip) -> str:
    """`audio_path|text` (newlines in the transcript collapsed to spaces)."""
    text = " ".join(clip.text.split())
    return f"{clip.audio_path}|{text}"


def parse_metadata_line(line: str) -> SpeakerClip:
    """Parse one `audio_path|text` line. Raises `DatasetError` on a malformed line."""
    if "|" not in line:
        raise DatasetError(f"malformed metadata line (expected 'audio_path|text'): {line!r}")
    audio_path, text = line.split("|", 1)
    audio_path, text = audio_path.strip(), text.strip()
    if not audio_path or not text:
        raise DatasetError(f"metadata line missing audio path or text: {line!r}")
    return SpeakerClip(audio_path=audio_path, text=text)


def read_metadata(path: str | Path) -> list[SpeakerClip]:
    """Read a `metadata.csv` (`audio_path|text` per line; blank lines skipped)."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"metadata file not found: {path}")
    clips: list[SpeakerClip] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            clips.append(parse_metadata_line(line))
    return clips


def write_metadata(path: str | Path, clips: Sequence[SpeakerClip]) -> Path:
    """Write `clips` as a `metadata.csv` (UTF-8). Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(metadata_line(c) for c in clips)
    path.write_text(body + "\n" if body else "", encoding="utf-8")
    return path


def validate_clip(
    clip: SpeakerClip,
    *,
    min_seconds: float | None = _MIN_CLIP_SECONDS,
    max_seconds: float | None = _MAX_CLIP_SECONDS,
) -> None:
    """Sanity-check one clip's transcript and (when known) its duration."""
    if not clip.audio_path.strip():
        raise DatasetError("clip is missing an audio path")
    if not clip.text.strip():
        raise DatasetError(f"clip {clip.audio_path!r} has no transcript")
    if clip.duration is not None:
        if min_seconds is not None and clip.duration < min_seconds:
            raise DatasetError(
                f"clip {clip.audio_path!r} too short ({clip.duration:.1f}s; min {min_seconds:.0f}s)"
            )
        if max_seconds is not None and clip.duration > max_seconds:
            raise DatasetError(
                f"clip {clip.audio_path!r} too long ({clip.duration:.1f}s; "
                f"max {max_seconds:.0f}s) — split it"
            )


def validate_dataset(
    clips: Sequence[SpeakerClip],
    *,
    min_clips: int = _MIN_CLIPS,
    min_total_seconds: float = _MIN_TOTAL_SECONDS,
    min_clip_seconds: float | None = _MIN_CLIP_SECONDS,
    max_clip_seconds: float | None = _MAX_CLIP_SECONDS,
) -> DatasetStats:
    """Validate every clip and the dataset as a whole; return its stats.

    Raises `DatasetError` if any clip is unusable, or (when durations are known for the whole
    set) if there are too few clips / too little audio for a usable fine-tune.
    """
    if not clips:
        raise DatasetError("empty dataset — record some target-speaker clips first")
    for clip in clips:
        validate_clip(clip, min_seconds=min_clip_seconds, max_seconds=max_clip_seconds)

    durations = [c.duration for c in clips if c.duration is not None]
    total = float(sum(durations))
    stats = DatasetStats(
        num_clips=len(clips),
        total_seconds=total,
        known_durations=len(durations),
        min_seconds=min(durations) if durations else None,
        max_seconds=max(durations) if durations else None,
    )
    if len(clips) < min_clips:
        raise DatasetError(
            f"too few clips ({len(clips)}; want at least {min_clips}) for a usable fine-tune"
        )
    # Only enforce the audio-minutes floor when we actually measured every clip.
    if len(durations) == len(clips) and total < min_total_seconds:
        raise DatasetError(
            f"only {total:.0f}s of audio across {len(clips)} clips; want at least "
            f"{min_total_seconds:.0f}s of the target speaker for a usable fine-tune"
        )
    return stats


def probe_durations(
    clips: Sequence[SpeakerClip], *, root: str | Path | None = None
) -> list[SpeakerClip]:
    """Return copies of `clips` with `duration` filled in by decoding each WAV.

    Needs an audio extra (numpy/soundfile). `root` is prepended to relative audio paths.
    """
    from ...audio import decode_wav

    root_path = Path(root) if root is not None else None
    out: list[SpeakerClip] = []
    for clip in clips:
        path = Path(clip.audio_path)
        if root_path is not None and not path.is_absolute():
            path = root_path / path
        try:
            samples, sr = decode_wav(path.read_bytes())
        except Exception as exc:  # missing/corrupt clip — surface which one
            raise DatasetError(f"could not read clip {path}: {exc}") from exc
        duration = len(samples) / sr if sr else 0.0
        out.append(clip.model_copy(update={"duration": duration}))
    return out


async def transcribe_clips(
    stt: object, audio_paths: Sequence[str | Path], *, root: str | Path | None = None
) -> list[SpeakerClip]:
    """Auto-transcribe each WAV with the cascade's STT into labeled clips (best effort).

    `stt` is an `STTAdapter` (typed loosely to avoid a heavy import). Clips that fail to
    transcribe to non-empty text are dropped, so the caller should check the result size.
    """
    from ...audio import read_wav_file

    root_path = Path(root) if root is not None else None
    clips: list[SpeakerClip] = []
    for raw in audio_paths:
        path = Path(raw)
        full = root_path / path if (root_path is not None and not path.is_absolute()) else path
        audio = read_wav_file(full)
        transcript = await stt.transcribe(audio)  # type: ignore[attr-defined]
        text = (getattr(transcript, "text", "") or "").strip()
        if text:
            clips.append(SpeakerClip(audio_path=str(raw), text=text))
    return clips

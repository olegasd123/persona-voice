"""Audio helpers shared by the file-based pipeline and the adapters.

Audio flows between stages as **WAV-encoded bytes** (self-describing: sample rate and
channels live in the header). The STT adapter decodes wav → mono float32 and resamples to
its model rate; the TTS adapter encodes its float32 output back to wav bytes. Keeping the
codec here means adapters share one decode/encode/resample implementation.

`numpy` and `soundfile` are imported lazily so the package (and `server --check`) work on a
machine without the backend extras installed; calling these helpers without them raises a
clear `AudioDependencyError` naming the extra to install.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


class AudioDependencyError(RuntimeError):
    """Raised when an audio helper needs numpy/soundfile but they aren't installed."""


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise AudioDependencyError(
            "numpy is required for audio I/O; install a backend extra, e.g. "
            "`pip install -e '.[mac]'`"
        ) from exc
    return np


def _require_soundfile() -> Any:
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise AudioDependencyError(
            "soundfile is required for audio I/O; install a backend extra, e.g. "
            "`pip install -e '.[mac]'`"
        ) from exc
    return sf


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV bytes into (mono float32 in [-1, 1], sample_rate)."""
    np = _require_numpy()
    sf = _require_soundfile()
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if audio.ndim > 1:  # mixdown to mono
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32), int(sr)


def encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode a mono float32 array to 16-bit PCM WAV bytes (afplay/whisper friendly)."""
    np = _require_numpy()
    sf = _require_soundfile()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = np.clip(audio, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample mono float32 audio with linear interpolation (no ffmpeg needed).

    Linear interpolation is good enough for feeding STT; high-fidelity resampling for
    playback is not needed in the offline dev loop.
    """
    np = _require_numpy()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sr_in == sr_out or audio.size == 0:
        return audio
    n_out = round(audio.size * sr_out / sr_in)
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    src_idx = np.arange(audio.size, dtype=np.float64)
    dst_idx = np.linspace(0.0, audio.size - 1, n_out)
    return np.interp(dst_idx, src_idx, audio).astype(np.float32)


def wav_to_pcm16(data: bytes, target_sr: int) -> bytes:
    """Decode WAV bytes to mono 16-bit little-endian PCM at `target_sr`.

    This is the raw frame format LiveKit's `rtc.AudioFrame` carries, so the streaming
    agent uses it to push TTS audio onto the WebRTC track.
    """
    np = _require_numpy()
    samples, sr = decode_wav(data)
    samples = resample(samples, sr, target_sr)
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767.0).round().astype("<i2").tobytes()


def pcm16_to_wav(data: bytes, sample_rate: int) -> bytes:
    """Encode mono 16-bit little-endian PCM to WAV bytes (e.g. a VAD-buffered utterance)."""
    np = _require_numpy()
    samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    return encode_wav(samples, sample_rate)


def read_wav_file(path: str | Path) -> bytes:
    return Path(path).read_bytes()


def write_wav_file(path: str | Path, data: bytes) -> None:
    Path(path).write_bytes(data)

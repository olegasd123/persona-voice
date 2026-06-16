"""Audio helpers. Skipped unless numpy/soundfile (a backend extra) are installed."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("soundfile")

from personavoice.audio import (  # noqa: E402
    decode_wav,
    encode_wav,
    pcm16_to_wav,
    resample,
    wav_to_pcm16,
)


def test_wav_roundtrip_preserves_rate_and_length() -> None:
    tone = (0.5 * np.sin(np.linspace(0, 2 * np.pi * 220, 16000))).astype(np.float32)
    wav = encode_wav(tone, 16000)
    decoded, sr = decode_wav(wav)
    assert sr == 16000
    assert decoded.shape[0] == tone.shape[0]
    # 16-bit PCM round-trip: close but not exact.
    assert np.max(np.abs(decoded - tone)) < 1e-3


def test_resample_changes_length_proportionally() -> None:
    x = np.zeros(16000, dtype=np.float32)
    down = resample(x, 16000, 8000)
    assert abs(down.shape[0] - 8000) <= 1
    same = resample(x, 16000, 16000)
    assert same.shape[0] == 16000


def test_wav_to_pcm16_roundtrips_through_wav() -> None:
    tone = (0.5 * np.sin(np.linspace(0, 2 * np.pi * 220, 24000))).astype(np.float32)
    wav = encode_wav(tone, 24000)
    pcm = wav_to_pcm16(wav, 24000)
    # mono int16 -> 2 bytes/sample, no resample at matching rate.
    assert len(pcm) == tone.shape[0] * 2
    decoded, sr = decode_wav(pcm16_to_wav(pcm, 24000))
    assert sr == 24000
    assert np.max(np.abs(decoded - tone)) < 1e-3


def test_wav_to_pcm16_resamples_to_target_rate() -> None:
    tone = np.zeros(24000, dtype=np.float32)
    pcm = wav_to_pcm16(encode_wav(tone, 24000), 16000)
    assert abs(len(pcm) // 2 - 16000) <= 1


def test_decode_mixes_stereo_to_mono() -> None:
    import io

    import soundfile as sf

    stereo = np.zeros((1000, 2), dtype=np.float32)
    stereo[:, 0] = 1.0  # L
    stereo[:, 1] = -1.0  # R -> mono mean ~0
    buf = io.BytesIO()
    sf.write(buf, stereo, 16000, format="WAV", subtype="FLOAT")
    decoded, _ = decode_wav(buf.getvalue())
    assert decoded.ndim == 1
    assert np.max(np.abs(decoded)) < 1e-6

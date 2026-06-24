"""CUDA TTS adapter helpers."""

from __future__ import annotations

import pytest

from personavoice.adapters.tts.chatterbox import ChatterboxTTS, _waveform_to_wav


def test_chatterbox_flags() -> None:
    adapter = ChatterboxTTS()
    assert adapter.name == "chatterbox"
    assert adapter.implemented is True
    assert adapter.supports_cloning is True


def test_waveform_to_wav_flattens_2d() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    from personavoice.audio import decode_wav

    waveform = np.zeros((1, 8), dtype=np.float32)  # [1, N] like Chatterbox output
    wav = _waveform_to_wav(waveform, 24000)
    samples, sr = decode_wav(wav)
    assert sr == 24000
    assert samples.shape == (8,)

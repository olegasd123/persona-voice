"""CUDA TTS adapter helpers."""

from __future__ import annotations

import pytest

from personavoice.adapters.tts.chatterbox import (
    _DEFAULT_EXAGGERATION,
    ChatterboxTTS,
    _resolve_exaggeration,
    _waveform_to_wav,
)


def test_chatterbox_flags() -> None:
    adapter = ChatterboxTTS()
    assert adapter.name == "chatterbox"
    assert adapter.implemented is True
    assert adapter.supports_cloning is True


def test_resolve_exaggeration_maps_emotion_words() -> None:
    # Known descriptive words map to their configured intensity; case/space-insensitive.
    assert _resolve_exaggeration("neutral", _DEFAULT_EXAGGERATION) == 0.5
    assert _resolve_exaggeration("  Excited ", _DEFAULT_EXAGGERATION) == 0.8
    # The dynamic-emotion vocabulary (Feature F) is covered too.
    assert _resolve_exaggeration("sad", _DEFAULT_EXAGGERATION) == 0.4
    assert _resolve_exaggeration("happy", _DEFAULT_EXAGGERATION) == 0.7


def test_resolve_exaggeration_accepts_literal_number_and_clamps() -> None:
    assert _resolve_exaggeration("0.7", _DEFAULT_EXAGGERATION) == 0.7
    assert _resolve_exaggeration("9.0", _DEFAULT_EXAGGERATION) == 2.0  # clamped to range max


def test_resolve_exaggeration_falls_back_to_default() -> None:
    # None and unknown words both fall back to the supplied default, never raise.
    assert _resolve_exaggeration(None, 0.55) == 0.55
    assert _resolve_exaggeration("sproingy", 0.55) == 0.55


def test_waveform_to_wav_flattens_2d() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    from personavoice.audio import decode_wav

    waveform = np.zeros((1, 8), dtype=np.float32)  # [1, N] like Chatterbox output
    wav = _waveform_to_wav(waveform, 24000)
    samples, sr = decode_wav(wav)
    assert sr == 24000
    assert samples.shape == (8,)

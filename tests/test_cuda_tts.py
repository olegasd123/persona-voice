"""CUDA TTS adapters (Orpheus, Chatterbox): voice resolution + PCM/waveform wrapping."""

from __future__ import annotations

import pytest

from personavoice.adapters.tts.chatterbox import ChatterboxTTS, _waveform_to_wav
from personavoice.adapters.tts.orpheus import OrpheusTTS, _pcm16_to_wav, _resolve_voice
from personavoice.models import VoiceRef


def test_orpheus_flags() -> None:
    adapter = OrpheusTTS(model="canopylabs/orpheus-3b-0.1-ft")
    assert adapter.name == "orpheus"
    assert adapter.implemented is True
    # Orpheus exposes preset voices only (no reference-sample input); cloning is Chatterbox.
    assert adapter.supports_cloning is False
    assert adapter.sample_rate == 24000


def test_chatterbox_flags() -> None:
    adapter = ChatterboxTTS()
    assert adapter.name == "chatterbox"
    assert adapter.implemented is True
    assert adapter.supports_cloning is True


def test_resolve_voice_preset_vs_clone_ref() -> None:
    # A bare preset name is honored as-is.
    assert _resolve_voice(VoiceRef(id="leo"), default="tara") == "leo"
    # A clone-style ref falls back to the default preset (cloning is a separate feature).
    assert _resolve_voice(VoiceRef(id="voices/hr_warm"), default="tara") == "tara"
    assert _resolve_voice(VoiceRef(id=""), default="tara") == "tara"


def test_pcm16_to_wav_roundtrips_through_decode() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    from personavoice.audio import decode_wav

    # Two full-scale samples (max positive, max negative) at 24 kHz.
    raw = b"\xff\x7f\x00\x80"  # int16 LE: 32767, -32768
    wav = _pcm16_to_wav(raw, 24000)
    samples, sr = decode_wav(wav)
    assert sr == 24000
    assert samples.shape == (2,)
    assert samples[0] == pytest.approx(1.0, abs=1e-3)
    assert samples[1] == pytest.approx(-1.0, abs=1e-3)
    assert np.issubdtype(samples.dtype, np.floating)


def test_waveform_to_wav_flattens_2d() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    from personavoice.audio import decode_wav

    waveform = np.zeros((1, 8), dtype=np.float32)  # [1, N] like Chatterbox output
    wav = _waveform_to_wav(waveform, 24000)
    samples, sr = decode_wav(wav)
    assert sr == 24000
    assert samples.shape == (8,)

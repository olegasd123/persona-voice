"""The voice registry: logical voice ref -> distinct backend-native preset (M4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from personavoice.models import VoiceDef
from personavoice.persona import load_personas
from personavoice.voice import VoiceError, VoiceRegistry

PERSONA_REFS = {
    "companion": "voices/companion_soft",
    "hr_interviewer": "voices/hr_warm",
    "language_teacher": "voices/teacher_clear",
    "pm_interviewer": "voices/pm_calm",
}


def _registry(config_dir: Path) -> VoiceRegistry:
    return VoiceRegistry.load(config_dir / "voices.yaml")


def test_shipped_registry_covers_every_persona_voice(config_dir: Path) -> None:
    voices = _registry(config_dir)
    personas = load_personas(config_dir / "personas")
    for persona in personas.values():
        assert persona.voice.ref in voices, f"{persona.id} voice unregistered"


@pytest.mark.parametrize("tts", ["kokoro", "orpheus"])
def test_personas_resolve_to_distinct_presets(config_dir: Path, tts: str) -> None:
    voices = _registry(config_dir)
    presets = [voices.resolve(ref, tts).id for ref in PERSONA_REFS.values()]
    # The whole point of M4: each persona gets its own concrete preset on each backend.
    assert len(set(presets)) == len(presets)
    assert all("/" not in p for p in presets)  # concrete preset, not a clone-style ref


def test_voices_prefix_is_optional(config_dir: Path) -> None:
    voices = _registry(config_dir)
    with_prefix = voices.resolve("voices/companion_soft", "kokoro")
    without_prefix = voices.resolve("companion_soft", "kokoro")
    assert with_prefix.id == without_prefix.id
    assert with_prefix.id == "af_heart"


def test_resolve_sets_emotion_and_backend(config_dir: Path) -> None:
    voice = _registry(config_dir).resolve("voices/companion_soft", "kokoro")
    assert voice.emotion == "warm"  # from the registry entry
    assert voice.backend == "kokoro"
    assert voice.name == "companion_soft"


def test_backend_without_a_preset_passes_the_ref_through() -> None:
    # Chatterbox/F5 are clone-only: no preset entry, so the raw ref is returned and the
    # adapter falls back to its default (until M5 uses `sample`).
    reg = VoiceRegistry(
        {"hr_warm": VoiceDef(emotion="warm", presets={"kokoro": "af_sarah"}, sample="s.wav")}
    )
    voice = reg.resolve("voices/hr_warm", "chatterbox")
    assert voice.id == "voices/hr_warm"
    assert voice.sample_path == "s.wav"  # carried for the cloning backend (M5)
    assert voice.emotion == "warm"
    assert reg.has_preset("voices/hr_warm", "kokoro") is True
    assert reg.has_preset("voices/hr_warm", "chatterbox") is False


def test_unknown_voice_falls_back_to_the_ref() -> None:
    reg = VoiceRegistry({})
    voice = reg.resolve("voices/nope", "kokoro", default_emotion="sad")
    assert voice.id == "voices/nope"
    assert voice.emotion == "sad"  # default carried through for an unknown voice
    assert "voices/nope" not in reg


def test_entry_emotion_overrides_default() -> None:
    reg = VoiceRegistry({"v": VoiceDef(emotion="warm", presets={"kokoro": "af_heart"})})
    assert reg.resolve("v", "kokoro", default_emotion="neutral").emotion == "warm"


def test_entry_without_emotion_uses_default() -> None:
    reg = VoiceRegistry({"v": VoiceDef(presets={"kokoro": "af_heart"})})
    assert reg.resolve("v", "kokoro", default_emotion="neutral").emotion == "neutral"


def test_missing_file_yields_empty_registry(tmp_path: Path) -> None:
    voices = VoiceRegistry.load(tmp_path / "nope.yaml")
    assert len(voices) == 0
    assert voices.resolve("voices/x", "kokoro").id == "voices/x"  # passthrough


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "voices.yaml"
    bad.write_text("- not\n- a mapping\n")
    with pytest.raises(VoiceError, match="mapping"):
        VoiceRegistry.load(bad)


def test_unknown_field_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "voices.yaml"
    bad.write_text("v:\n  bogus: 1\n")
    with pytest.raises(VoiceError, match="invalid voice registry"):
        VoiceRegistry.load(bad)


async def test_pipeline_speaks_each_persona_in_its_own_voice(config_dir: Path) -> None:
    """End-to-end: the registry threads through to the TTS call, distinctly per persona."""
    from personavoice.orchestrator.pipeline import Pipeline
    from personavoice.persona import load_personas

    from .fakes import make_backend

    voices = _registry(config_dir)
    personas = load_personas(config_dir / "personas")

    spoken: dict[str, str] = {}
    for pid in PERSONA_REFS:
        backend = make_backend()
        backend.tts.name = "kokoro"  # match the registry's preset keys
        pipe = Pipeline(backend, personas[pid], voices)
        await pipe.run_turn(b"x")
        spoken[pid] = backend.tts.last_voice.id  # type: ignore[union-attr]

    assert spoken["companion"] == "af_heart"
    assert spoken["pm_interviewer"] == "am_michael"
    assert len(set(spoken.values())) == len(spoken)  # all distinct

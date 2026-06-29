"""The voice registry: logical voice ref -> distinct backend-native preset."""

from __future__ import annotations

from pathlib import Path

import pytest

from personavoice.models import VoiceDef
from personavoice.persona import load_personas
from personavoice.voice import (
    ClonedVoice,
    ClonesStore,
    FinetunedVoice,
    FinetunedVoicesStore,
    VoiceError,
    VoiceRegistry,
)

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


def test_personas_resolve_to_distinct_kokoro_presets(config_dir: Path) -> None:
    voices = _registry(config_dir)
    presets = [voices.resolve(ref, "kokoro").id for ref in PERSONA_REFS.values()]
    # The whole point: each persona gets its own concrete preset on preset-capable backends.
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
    # Chatterbox is clone-only: no preset entry, so the raw ref is returned and the
    # adapter falls back to its default (until a clone uses `sample`).
    reg = VoiceRegistry(
        {"hr_warm": VoiceDef(emotion="warm", presets={"kokoro": "af_sarah"}, sample="s.wav")}
    )
    voice = reg.resolve("voices/hr_warm", "chatterbox")
    assert voice.id == "voices/hr_warm"
    assert voice.sample_path == "s.wav"  # carried for the cloning backend
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


# --- catalog + resolve_choice ---------------------------------------------


def _registry_with_stores(tmp_path: Path) -> VoiceRegistry:
    """A registry with one preset, one clone, and one fine-tuned voice for catalog tests."""
    clones = ClonesStore(tmp_path / "clones")
    clones.record(ClonedVoice(name="my_clone", sample_path="s.wav"))
    finetuned = FinetunedVoicesStore(tmp_path / "ft")
    finetuned.record(FinetunedVoice(name="my_ft", checkpoint_path="ckpt"))
    voices = {"companion_soft": VoiceDef(description="warm", presets={"kokoro": "af_heart"})}
    return VoiceRegistry(voices, clones=clones, finetuned=finetuned)


def test_catalog_order_and_availability_non_cloning(tmp_path: Path) -> None:
    reg = _registry_with_stores(tmp_path)
    catalog = reg.catalog("kokoro", supports_cloning=False)
    # Stable order: finetuned, clones, presets.
    assert [o.kind for o in catalog] == ["finetuned", "clone", "preset"]
    by_id = {o.id: o for o in catalog}
    # On a non-cloning backend clones/fine-tunes are listed but unavailable, with a reason.
    assert by_id["my_ft"].available is False and by_id["my_ft"].reason
    assert by_id["my_clone"].available is False and by_id["my_clone"].reason
    # The preset is available where it maps for this backend.
    assert by_id["companion_soft"].available is True
    assert by_id["companion_soft"].name == "warm"  # description used as the label


def test_catalog_availability_cloning_backend(tmp_path: Path) -> None:
    reg = _registry_with_stores(tmp_path)
    catalog = reg.catalog("chatterbox", supports_cloning=True)
    by_id = {o.id: o for o in catalog}
    assert by_id["my_clone"].available is True and by_id["my_clone"].reason is None
    assert by_id["my_ft"].available is True
    # Chatterbox has no preset mapping, so the unmapped preset is omitted entirely.
    assert "companion_soft" not in by_id


def test_catalog_marks_only_unprotected_clones_removable(tmp_path: Path) -> None:
    reg = _registry_with_stores(tmp_path)
    reg.clones.record(ClonedVoice(name="Feminine", sample_path="f.wav"))  # a bundled seed voice
    by_id = {
        o.id: o for o in reg.catalog("chatterbox", supports_cloning=True, protected={"Feminine"})
    }
    # A user clone is removable; the protected "inbox" voice and the fine-tune are not.
    assert by_id["my_clone"].removable is True
    assert by_id["Feminine"].removable is False
    assert by_id["my_ft"].removable is False


def test_resolve_choice_precedence_finetuned_over_clone(tmp_path: Path) -> None:
    reg = _registry_with_stores(tmp_path)
    ref = reg.resolve_choice("my_ft", "chatterbox", supports_cloning=True)
    assert ref is not None and ref.model_path == "ckpt"


def test_resolve_choice_clone_on_cloning_backend(tmp_path: Path) -> None:
    reg = _registry_with_stores(tmp_path)
    ref = reg.resolve_choice(
        "my_clone", "chatterbox", supports_cloning=True, default_emotion="warm"
    )
    assert ref is not None
    assert ref.sample_path == "s.wav"
    assert ref.emotion == "warm"


def test_resolve_choice_clone_unavailable_on_non_cloning_backend(tmp_path: Path) -> None:
    reg = _registry_with_stores(tmp_path)
    # The clone isn't speakable on Kokoro -> None, so the caller falls back to the default.
    assert reg.resolve_choice("my_clone", "kokoro", supports_cloning=False) is None


def test_resolve_choice_preset(tmp_path: Path) -> None:
    reg = _registry_with_stores(tmp_path)
    ref = reg.resolve_choice("companion_soft", "kokoro", supports_cloning=False)
    assert ref is not None and ref.id == "af_heart"


def test_resolve_choice_preset_without_mapping_is_none(tmp_path: Path) -> None:
    reg = _registry_with_stores(tmp_path)
    # Known preset, but no mapping for chatterbox -> not speakable as that choice -> None.
    assert reg.resolve_choice("companion_soft", "chatterbox", supports_cloning=True) is None


def test_resolve_choice_unknown_is_none(tmp_path: Path) -> None:
    reg = _registry_with_stores(tmp_path)
    assert reg.resolve_choice("nope", "kokoro", supports_cloning=True) is None


def test_choice_ids_spans_all_kinds(tmp_path: Path) -> None:
    reg = _registry_with_stores(tmp_path)
    assert reg.choice_ids() == ["companion_soft", "my_clone", "my_ft"]


# --- fine-tune engine gating ---------------------------------------------------------


def _ft_registry(tmp_path: Path, *, engine: str | None, name: str = "trained") -> VoiceRegistry:
    """A registry whose single fine-tuned voice carries `engine` (None = legacy/unknown)."""
    finetuned = FinetunedVoicesStore(tmp_path / "ft")
    finetuned.record(FinetunedVoice(name=name, checkpoint_path="ckpt", engine=engine))
    return VoiceRegistry({}, finetuned=finetuned)


def test_catalog_marks_engine_mismatched_finetune_unavailable(tmp_path: Path) -> None:
    # A fine-tune from another engine is listed, but greyed-out with an engine reason.
    reg = _ft_registry(tmp_path, engine="other")
    opt = {o.id: o for o in reg.catalog("chatterbox", supports_cloning=True)}["trained"]
    assert opt.available is False
    assert opt.reason and "other" in opt.reason


def test_catalog_lists_matching_engine_finetune_available(tmp_path: Path) -> None:
    reg = _ft_registry(tmp_path, engine="chatterbox")
    opt = {o.id: o for o in reg.catalog("chatterbox", supports_cloning=True)}["trained"]
    assert opt.available is True and opt.reason is None


def test_catalog_legacy_finetune_without_engine_stays_available(tmp_path: Path) -> None:
    # No recorded engine -> can't prove a mismatch, so it stays available (don't hide it).
    reg = _ft_registry(tmp_path, engine=None)
    opt = {o.id: o for o in reg.catalog("chatterbox", supports_cloning=True)}["trained"]
    assert opt.available is True


def test_resolve_choice_engine_mismatched_finetune_is_none(tmp_path: Path) -> None:
    reg = _ft_registry(tmp_path, engine="other")
    # Mismatched engine -> not speakable here -> None (caller falls back to the persona default).
    assert reg.resolve_choice("trained", "chatterbox", supports_cloning=True) is None


def test_resolve_for_persona_skips_engine_mismatched_finetune(tmp_path: Path) -> None:
    from .fakes import make_persona

    finetuned = FinetunedVoicesStore(tmp_path / "ft")
    finetuned.record(FinetunedVoice(name="trained", checkpoint_path="ckpt", engine="other"))
    finetuned.assign("p", "trained")
    # The persona's ref maps to a chatterbox preset (with a sample) — what it falls back to.
    voices = {"test": VoiceDef(presets={"chatterbox": "cb_voice"}, sample="s.wav")}
    reg = VoiceRegistry(voices, finetuned=finetuned)
    ref = reg.resolve_for_persona(make_persona("p"), "chatterbox", supports_cloning=True)
    # Not the mismatched checkpoint — fell through to the chatterbox preset's sample.
    assert ref.model_path is None
    assert ref.sample_path == "s.wav"


def test_preset_sample_stems_only_counts_sample_backed_presets() -> None:
    reg = VoiceRegistry(
        {
            "Feminine": VoiceDef(
                sample="assets/seed_voices/Feminine.wav", presets={"chatterbox": "Feminine"}
            ),
            "companion_soft": VoiceDef(presets={"kokoro": "af_heart"}),  # no sample
        }
    )
    assert reg.preset_sample_stems() == {"Feminine"}


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

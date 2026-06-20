"""Fine-tuned voices: store persistence, registry precedence, adapter wiring, --check."""

from __future__ import annotations

from pathlib import Path

import pytest

from personavoice.adapters.tts.f5_mlx import _f5_kwargs
from personavoice.models import VoiceRef
from personavoice.persona import load_personas
from personavoice.voice import (
    ClonedVoice,
    ClonesStore,
    FinetunedVoice,
    FinetunedVoiceError,
    FinetunedVoicesStore,
    VoiceRegistry,
)

# --------------------------------------------------------------------------------------
# FinetunedVoicesStore
# --------------------------------------------------------------------------------------


def test_store_missing_manifest_is_empty(tmp_path: Path) -> None:
    store = FinetunedVoicesStore.load(tmp_path)
    assert len(store) == 0 and store.names() == []
    assert store.voice_ref("nope", "chatterbox") is None


def test_store_record_assign_and_reload(tmp_path: Path) -> None:
    store = FinetunedVoicesStore.load(tmp_path)
    store.record(FinetunedVoice(name="my_voice", checkpoint_path="/ckpt/my_voice", engine="f5"))
    store.assign("companion", "my_voice")

    reloaded = FinetunedVoicesStore.load(tmp_path)
    assert reloaded.names() == ["my_voice"]
    assert "my_voice" in reloaded
    assert reloaded.assignment_for("companion") == "my_voice"
    assert reloaded.get("my_voice").engine == "f5"


def test_store_voice_ref_carries_model_path(tmp_path: Path) -> None:
    store = FinetunedVoicesStore.load(tmp_path)
    store.record(FinetunedVoice(name="v", checkpoint_path="/ckpt/v"))
    ref = store.voice_ref("v", "chatterbox", emotion="warm")
    assert ref is not None
    assert ref.model_path == "/ckpt/v"
    assert ref.emotion == "warm"
    assert ref.backend == "chatterbox"


def test_store_record_rejects_bad_name(tmp_path: Path) -> None:
    store = FinetunedVoicesStore.load(tmp_path)
    with pytest.raises(FinetunedVoiceError, match="invalid voice name"):
        store.record(FinetunedVoice(name="bad name!", checkpoint_path="/c"))


def test_store_assign_unknown_raises(tmp_path: Path) -> None:
    store = FinetunedVoicesStore.load(tmp_path)
    with pytest.raises(FinetunedVoiceError, match="unknown fine-tuned voice"):
        store.assign("companion", "ghost")


def test_store_unassign(tmp_path: Path) -> None:
    store = FinetunedVoicesStore.load(tmp_path)
    store.record(FinetunedVoice(name="v", checkpoint_path="/c"))
    store.assign("companion", "v")
    assert store.unassign("companion") is True
    assert store.unassign("companion") is False


def test_store_corrupt_manifest_raises(tmp_path: Path) -> None:
    (tmp_path / "finetuned.json").write_text("{ not json")
    with pytest.raises(FinetunedVoiceError, match="invalid finetuned manifest"):
        FinetunedVoicesStore.load(tmp_path)


# --------------------------------------------------------------------------------------
# adapter wiring: VoiceRef.model_path
# --------------------------------------------------------------------------------------


def test_f5_kwargs_model_path_overrides_base() -> None:
    kwargs = _f5_kwargs("hi", VoiceRef(id="v", model_path="/ckpt/v"), model="base-model")
    assert kwargs["model_name"] == "/ckpt/v"  # the fine-tuned checkpoint, not the base


def test_f5_kwargs_without_model_path_uses_base() -> None:
    kwargs = _f5_kwargs("hi", VoiceRef(id="v"), model="base-model")
    assert kwargs["model_name"] == "base-model"


# --------------------------------------------------------------------------------------
# registry precedence: fine-tuned ▶ clone ▶ preset
# --------------------------------------------------------------------------------------


def _registry_with(
    config_dir: Path, tmp_path: Path, *, with_clone: bool, with_finetuned: bool
) -> VoiceRegistry:
    clones = ClonesStore.load(tmp_path / "clones")
    finetuned = FinetunedVoicesStore.load(tmp_path / "finetuned")
    if with_clone:
        clones.record(ClonedVoice(name="cloned", sample_path="/s/cloned.wav"))
        clones.assign("companion", "cloned")
    if with_finetuned:
        finetuned.record(FinetunedVoice(name="trained", checkpoint_path="/ckpt/trained"))
        finetuned.assign("companion", "trained")
    return VoiceRegistry.load(config_dir / "voices.yaml", clones=clones, finetuned=finetuned)


def test_finetuned_outranks_clone_on_cloning_backend(config_dir: Path, tmp_path: Path) -> None:
    registry = _registry_with(config_dir, tmp_path, with_clone=True, with_finetuned=True)
    companion = load_personas(config_dir / "personas")["companion"]
    ref = registry.resolve_for_persona(companion, "chatterbox", supports_cloning=True)
    assert ref.model_path == "/ckpt/trained"  # the fine-tune wins over the clone
    assert ref.sample_path is None
    assert ref.emotion == companion.voice.emotion


def test_clone_used_when_no_finetuned(config_dir: Path, tmp_path: Path) -> None:
    registry = _registry_with(config_dir, tmp_path, with_clone=True, with_finetuned=False)
    companion = load_personas(config_dir / "personas")["companion"]
    ref = registry.resolve_for_persona(companion, "chatterbox", supports_cloning=True)
    assert ref.model_path is None and ref.sample_path == "/s/cloned.wav"


def test_finetuned_ignored_on_non_cloning_backend(config_dir: Path, tmp_path: Path) -> None:
    registry = _registry_with(config_dir, tmp_path, with_clone=False, with_finetuned=True)
    companion = load_personas(config_dir / "personas")["companion"]
    # Kokoro can't load a checkpoint → the persona keeps its distinct preset.
    ref = registry.resolve_for_persona(companion, "kokoro", supports_cloning=False)
    assert ref.id == "af_heart" and ref.model_path is None


def test_finetuned_for_persona_accessor(config_dir: Path, tmp_path: Path) -> None:
    registry = _registry_with(config_dir, tmp_path, with_clone=False, with_finetuned=True)
    assert registry.finetuned_for_persona("companion") == "trained"
    assert registry.finetuned_for_persona("pm_interviewer") is None


async def test_pipeline_speaks_the_finetuned_checkpoint(config_dir: Path, tmp_path: Path) -> None:
    import dataclasses

    from personavoice.orchestrator.pipeline import Pipeline

    from .fakes import FakeTTS, make_backend

    class CloningTTS(FakeTTS):
        name = "chatterbox"
        supports_cloning = True

    registry = _registry_with(config_dir, tmp_path, with_clone=True, with_finetuned=True)
    companion = load_personas(config_dir / "personas")["companion"]
    backend = dataclasses.replace(make_backend(), tts=CloningTTS())
    pipe = Pipeline(backend, companion, registry)
    await pipe.run_turn(b"x")

    assert backend.tts.last_voice.model_path == "/ckpt/trained"  # type: ignore[union-attr]


# --------------------------------------------------------------------------------------
# server --check integration
# --------------------------------------------------------------------------------------


def test_check_lists_finetuned_voices(config_dir: Path, tmp_path: Path) -> None:
    from personavoice.server.check import run_check
    from personavoice.server.config import Settings

    store = FinetunedVoicesStore.load(tmp_path / "finetuned")
    store.record(FinetunedVoice(name="trained", checkpoint_path="/ckpt/trained"))
    store.assign("companion", "trained")

    settings = Settings(
        backend="mac",
        config_dir=config_dir,
        models_dir=tmp_path / "models",
        finetuned_dir=tmp_path / "finetuned",
    )
    report = run_check(settings)
    assert report.ok
    assert "trained" in report.finetuned
    assert report.finetuned_assignments.get("companion") == "trained"

"""Zero-shot voice cloning: sample validation, clone store, cloner, and resolution."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from personavoice.adapters.factory import Backend
from personavoice.adapters.tts.base import TTSAdapter
from personavoice.adapters.tts.f5_mlx import _f5_kwargs
from personavoice.models import VoiceRef
from personavoice.persona import load_personas
from personavoice.voice import (
    ClonedVoice,
    CloneError,
    ClonesStore,
    VoiceCloner,
    VoiceRegistry,
    validate_sample,
)

from .fakes import FakeTTS, make_backend


class CloningTTS(FakeTTS):
    """A fake TTS that can clone (uses the base `clone_voice`: persist sample → VoiceRef)."""

    name = "chatterbox"
    supports_cloning = True


def _cloning_backend(stt_text: str = "this is my voice") -> Backend:
    be = make_backend(stt_text=stt_text)
    return dataclasses.replace(be, tts=CloningTTS())


# --------------------------------------------------------------------------------------
# validate_sample
# --------------------------------------------------------------------------------------


def _wav(seconds: float, sr: int = 16000) -> bytes:
    np = pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    from personavoice.audio import encode_wav

    return encode_wav(np.zeros(int(seconds * sr), dtype=np.float32), sr)


def test_validate_sample_returns_duration() -> None:
    assert validate_sample(_wav(5.0)) == pytest.approx(5.0, abs=0.05)


def test_validate_sample_rejects_too_short() -> None:
    with pytest.raises(CloneError, match="too short"):
        validate_sample(_wav(1.0))


def test_validate_sample_rejects_too_long() -> None:
    with pytest.raises(CloneError, match="too long"):
        validate_sample(_wav(5.0), max_seconds=2.0)


def test_validate_sample_rejects_empty() -> None:
    with pytest.raises(CloneError, match="empty"):
        validate_sample(b"")


def test_validate_sample_rejects_non_wav() -> None:
    pytest.importorskip("soundfile")
    with pytest.raises(CloneError, match="could not decode"):
        validate_sample(b"not a wav at all")


# --------------------------------------------------------------------------------------
# ClonesStore
# --------------------------------------------------------------------------------------


def test_store_missing_manifest_is_empty(tmp_path: Path) -> None:
    store = ClonesStore.load(tmp_path)
    assert len(store) == 0
    assert store.names() == []
    assert store.voice_ref("nope", "chatterbox") is None


def test_store_record_assign_and_reload(tmp_path: Path) -> None:
    store = ClonesStore.load(tmp_path)
    store.record(ClonedVoice(name="my_voice", sample_path=str(tmp_path / "my_voice.wav")))
    store.assign("companion", "my_voice")

    # Persisted: a fresh load sees the clone and the assignment.
    reloaded = ClonesStore.load(tmp_path)
    assert reloaded.names() == ["my_voice"]
    assert "my_voice" in reloaded
    assert reloaded.assignment_for("companion") == "my_voice"
    assert reloaded.assignments == {"companion": "my_voice"}


def test_store_voice_ref_carries_sample_and_ref_text(tmp_path: Path) -> None:
    store = ClonesStore.load(tmp_path)
    store.record(
        ClonedVoice(name="v", sample_path="/s/v.wav", ref_text="hello", backend="chatterbox")
    )
    ref = store.voice_ref("v", "chatterbox", emotion="warm")
    assert ref is not None
    assert ref.id == "v"
    assert ref.sample_path == "/s/v.wav"
    assert ref.ref_text == "hello"
    assert ref.emotion == "warm"
    assert ref.backend == "chatterbox"


def test_store_assign_unknown_raises(tmp_path: Path) -> None:
    store = ClonesStore.load(tmp_path)
    with pytest.raises(CloneError, match="unknown clone"):
        store.assign("companion", "ghost")


def test_store_unassign(tmp_path: Path) -> None:
    store = ClonesStore.load(tmp_path)
    store.record(ClonedVoice(name="v", sample_path="/s/v.wav"))
    store.assign("companion", "v")
    assert store.unassign("companion") is True
    assert store.unassign("companion") is False  # already gone
    assert store.assignment_for("companion") is None


def test_store_corrupt_manifest_raises(tmp_path: Path) -> None:
    (tmp_path / "clones.json").write_text("{ not json")
    with pytest.raises(CloneError, match="invalid clones manifest"):
        ClonesStore.load(tmp_path)


# --------------------------------------------------------------------------------------
# TTSAdapter.clone_voice (base) + F5 kwargs
# --------------------------------------------------------------------------------------


async def test_base_clone_voice_persists_sample(tmp_path: Path) -> None:
    tts = CloningTTS()
    tts.options["clones_dir"] = str(tmp_path)
    voice = await tts.clone_voice(b"RIFFsample", "my_voice")
    assert voice.id == "my_voice"
    assert voice.backend == "chatterbox"
    written = Path(voice.sample_path)
    assert written == tmp_path / "my_voice.wav"
    assert written.read_bytes() == b"RIFFsample"


async def test_base_clone_voice_rejected_when_not_supported() -> None:
    class NoClone(TTSAdapter):
        name = "kokoro"

    with pytest.raises(NotImplementedError, match="does not support voice cloning"):
        await NoClone().clone_voice(b"x", "v")


def test_f5_kwargs_with_and_without_sample() -> None:
    bare = _f5_kwargs("hi", VoiceRef(id="x"), model="m")
    assert bare == {"generation_text": "hi", "model_name": "m"}  # F5's built-in default voice

    cloned = _f5_kwargs("hi", VoiceRef(id="v", sample_path="/s/v.wav", ref_text="ref"), model="m")
    assert cloned["ref_audio_path"] == "/s/v.wav"
    assert cloned["ref_audio_text"] == "ref"  # f5_tts_mlx names it ref_audio_text


def test_f5_ref_at_24k_resamples_when_needed(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    from personavoice.adapters.tts.f5_mlx import _F5_REF_RATE, F5MLXTTS
    from personavoice.audio import decode_wav, encode_wav

    adapter = F5MLXTTS()
    # 44.1 kHz reference → resampled to a 24 kHz temp file f5 will accept.
    ref = tmp_path / "ref.wav"
    ref.write_bytes(encode_wav(np.zeros(44100, dtype=np.float32), 44100))
    out, tmp = adapter._ref_at_24k(str(ref))
    assert tmp is not None and out == tmp
    _, sr = decode_wav(Path(out).read_bytes())
    assert sr == _F5_REF_RATE

    # Already 24 kHz → passed through unchanged (no temp file to clean up).
    ref24 = tmp_path / "ref24.wav"
    ref24.write_bytes(encode_wav(np.zeros(24000, dtype=np.float32), 24000))
    assert adapter._ref_at_24k(str(ref24)) == (str(ref24), None)
    # No sample → nothing to do.
    assert adapter._ref_at_24k(None) == (None, None)


# --------------------------------------------------------------------------------------
# VoiceCloner
# --------------------------------------------------------------------------------------


async def test_cloner_clones_records_and_assigns(tmp_path: Path) -> None:
    store = ClonesStore.load(tmp_path)
    cloner = VoiceCloner(_cloning_backend(stt_text="this is my voice"), store)

    # Skip audio-decode validation: the fake sample isn't a real wav.
    voice = await cloner.clone(
        b"RIFFsample", "my_voice", assign_to="companion", min_seconds=None, max_seconds=None
    )

    assert voice.sample_path == str(tmp_path / "my_voice.wav")
    assert voice.ref_text == "this is my voice"  # filled by the cascade's STT
    # Recorded + assigned, and it survives a reload.
    reloaded = ClonesStore.load(tmp_path)
    assert reloaded.get("my_voice").ref_text == "this is my voice"
    assert reloaded.assignment_for("companion") == "my_voice"


async def test_cloner_rejects_non_cloning_backend(tmp_path: Path) -> None:
    store = ClonesStore.load(tmp_path)
    cloner = VoiceCloner(make_backend(), store)  # FakeTTS: supports_cloning is False
    with pytest.raises(CloneError, match="can't clone voices"):
        await cloner.clone(b"x", "v", min_seconds=None, max_seconds=None)


async def test_cloner_rejects_bad_name(tmp_path: Path) -> None:
    cloner = VoiceCloner(_cloning_backend(), ClonesStore.load(tmp_path))
    with pytest.raises(CloneError, match="invalid clone name"):
        await cloner.clone(b"x", "bad name!", min_seconds=None, max_seconds=None)


async def test_cloner_explicit_ref_text_skips_transcription(tmp_path: Path) -> None:
    cloner = VoiceCloner(_cloning_backend(stt_text="WRONG"), ClonesStore.load(tmp_path))
    voice = await cloner.clone(b"x", "v", ref_text="given text", min_seconds=None, max_seconds=None)
    assert voice.ref_text == "given text"


# --------------------------------------------------------------------------------------
# Registry / pipeline resolution
# --------------------------------------------------------------------------------------


def _assigned_registry(config_dir: Path, tmp_path: Path) -> VoiceRegistry:
    store = ClonesStore.load(tmp_path)
    store.record(ClonedVoice(name="my_voice", sample_path="/s/my_voice.wav", ref_text="hi"))
    store.assign("companion", "my_voice")
    return VoiceRegistry.load(config_dir / "voices.yaml", clones=store)


def test_assigned_clone_wins_on_cloning_backend(config_dir: Path, tmp_path: Path) -> None:
    registry = _assigned_registry(config_dir, tmp_path)
    companion = load_personas(config_dir / "personas")["companion"]
    ref = registry.resolve_for_persona(companion, "chatterbox", supports_cloning=True)
    assert ref.sample_path == "/s/my_voice.wav"
    assert ref.ref_text == "hi"
    assert ref.emotion == companion.voice.emotion  # carried from the persona


def test_assigned_clone_ignored_on_non_cloning_backend(config_dir: Path, tmp_path: Path) -> None:
    registry = _assigned_registry(config_dir, tmp_path)
    companion = load_personas(config_dir / "personas")["companion"]
    # Kokoro can't clone → the persona keeps its distinct preset, not the clone.
    ref = registry.resolve_for_persona(companion, "kokoro", supports_cloning=False)
    assert ref.id == "af_heart"
    assert ref.sample_path is None


def test_unassigned_persona_uses_preset(config_dir: Path, tmp_path: Path) -> None:
    registry = _assigned_registry(config_dir, tmp_path)
    pm = load_personas(config_dir / "personas")["pm_interviewer"]  # no clone assigned
    ref = registry.resolve_for_persona(pm, "chatterbox", supports_cloning=True)
    assert ref.sample_path is None  # falls back to the static registry resolution


async def test_pipeline_speaks_the_assigned_clone(config_dir: Path, tmp_path: Path) -> None:
    from personavoice.orchestrator.pipeline import Pipeline

    registry = _assigned_registry(config_dir, tmp_path)
    companion = load_personas(config_dir / "personas")["companion"]
    backend = _cloning_backend()
    pipe = Pipeline(backend, companion, registry)
    await pipe.run_turn(b"x")

    assert backend.tts.last_voice.sample_path == "/s/my_voice.wav"  # type: ignore[union-attr]


def test_registry_without_clones_resolves_preset(config_dir: Path) -> None:
    registry = VoiceRegistry.load(config_dir / "voices.yaml")  # no clones attached
    companion = load_personas(config_dir / "personas")["companion"]
    ref = registry.resolve_for_persona(companion, "chatterbox", supports_cloning=True)
    assert ref.sample_path is None
    assert registry.clone_for_persona("companion") is None


# --------------------------------------------------------------------------------------
# server --check integration
# --------------------------------------------------------------------------------------


def test_check_lists_clones(config_dir: Path, tmp_path: Path) -> None:
    from personavoice.server.check import run_check
    from personavoice.server.config import Settings

    store = ClonesStore.load(tmp_path)
    store.record(ClonedVoice(name="my_voice", sample_path="/s/my_voice.wav"))
    store.assign("companion", "my_voice")

    settings = Settings(
        backend="mac", config_dir=config_dir, models_dir=tmp_path / "models", clones_dir=tmp_path
    )
    report = run_check(settings)
    assert report.ok
    assert "my_voice" in report.clones
    assert report.clone_assignments.get("companion") == "my_voice"


def test_validate_personas_clone_suppresses_distinctness_warning(
    config_dir: Path, tmp_path: Path
) -> None:
    from personavoice.adapters.factory import build_backend
    from personavoice.models import BackendConfig, StageConfig
    from personavoice.server.check import _validate_personas

    # Chatterbox can clone but the registry has no chatterbox *presets*, so without an assigned clone, every
    # persona would warn; an assigned clone (and cloning support) clears those warnings.
    backend = build_backend(
        BackendConfig(
            backend="cuda",
            stt=StageConfig(adapter="faster_whisper"),
            llm=StageConfig(adapter="vllm"),
            tts=StageConfig(adapter="chatterbox"),
        )
    )
    store = ClonesStore.load(tmp_path)
    store.record(ClonedVoice(name="v", sample_path="/s/v.wav"))
    store.assign("companion", "v")
    registry = VoiceRegistry.load(config_dir / "voices.yaml", clones=store)
    personas = load_personas(config_dir / "personas")

    assert _validate_personas(personas, backend, registry) == []

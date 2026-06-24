"""Seed voices: the idempotent enroll loop, sample discovery, and per-backend enrollers."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from personavoice.voice.clone import ClonedVoice, CloneError, ClonesStore
from personavoice.voice.seed import (
    SeedResult,
    discover_samples,
    make_enroller,
    seed_clones,
    seed_voice_names,
)

from .fakes import make_backend
from .test_clone import CloningTTS, _wav


def _recording_enroller(store: ClonesStore):
    """An enroll fn that records a bare clone into `store` (no backend / models)."""

    async def _enroll(name: str, wav: bytes) -> None:
        store.record(ClonedVoice(name=name, sample_path=f"{name}.wav"))

    return _enroll


# --- seed_clones (the pure loop) ------------------------------------------------------


async def test_seed_clones_enrolls_all(tmp_path: Path) -> None:
    store = ClonesStore(tmp_path)
    result = await seed_clones(_recording_enroller(store), store, {"female": b"f", "male": b"m"})
    assert result.seeded == ["female", "male"]
    assert result.skipped == [] and result.ok
    assert "female" in store and "male" in store


async def test_seed_clones_is_idempotent(tmp_path: Path) -> None:
    store = ClonesStore(tmp_path)
    samples = {"female": b"f", "male": b"m"}
    first = await seed_clones(_recording_enroller(store), store, samples)
    assert first.seeded == ["female", "male"]
    # A second run skips what's already present rather than re-enrolling.
    second = await seed_clones(_recording_enroller(store), store, samples)
    assert second.seeded == [] and second.skipped == ["female", "male"]


async def test_seed_clones_excludes_presets(tmp_path: Path) -> None:
    store = ClonesStore(tmp_path)
    samples = {"Feminine": b"f", "userclone": b"u"}
    # "Feminine" ships as a voices.yaml preset -> never enrolled as a clone, even with force.
    result = await seed_clones(
        _recording_enroller(store), store, samples, exclude={"Feminine"}, force=True
    )
    assert result.seeded == ["userclone"]
    assert result.excluded == ["Feminine"]
    assert "Feminine" not in store


async def test_seed_clones_force_re_enrolls(tmp_path: Path) -> None:
    store = ClonesStore(tmp_path)
    store.record(ClonedVoice(name="female", sample_path="old.wav"))
    result = await seed_clones(_recording_enroller(store), store, {"female": b"f"}, force=True)
    assert result.seeded == ["female"] and result.skipped == []


async def test_seed_clones_captures_failures(tmp_path: Path) -> None:
    store = ClonesStore(tmp_path)

    async def _bad(name: str, wav: bytes) -> None:
        raise CloneError("sample too short (1.0s)")

    result = await seed_clones(_bad, store, {"female": b"f"})
    assert result.seeded == [] and result.failed == [("female", "sample too short (1.0s)")]
    assert not result.ok


# --- discover_samples -----------------------------------------------------------------


def test_discover_samples_reads_wavs_keyed_by_stem(tmp_path: Path) -> None:
    (tmp_path / "female.wav").write_bytes(b"RIFFfemale")
    (tmp_path / "male.wav").write_bytes(b"RIFFmale")
    (tmp_path / "notes.txt").write_text("ignore me")
    samples = discover_samples(tmp_path)
    assert set(samples) == {"female", "male"}
    assert samples["female"] == b"RIFFfemale"


def test_discover_samples_missing_dir_is_empty(tmp_path: Path) -> None:
    assert discover_samples(tmp_path / "nope") == {}


def test_bundled_seed_dir_has_demo_voices() -> None:
    # The repo ships two real voices so a fresh checkout demos with both. They're also wired
    # as presets in config/voices.yaml (the `sample` field), so a cloning backend can speak
    # them out of the box.
    repo_root = Path(__file__).resolve().parents[1]
    samples = discover_samples(repo_root / "assets" / "seed_voices")
    assert {"Feminine", "Masculine"} <= set(samples)


# --- seed_voice_names (protected/"inbox" set) -----------------------------------------


def test_seed_voice_names_returns_stems(tmp_path: Path) -> None:
    (tmp_path / "female.wav").write_bytes(b"RIFF")
    (tmp_path / "male.wav").write_bytes(b"RIFF")
    (tmp_path / "notes.txt").write_text("ignore me")
    assert seed_voice_names(tmp_path) == {"female", "male"}


def test_seed_voice_names_missing_dir_is_empty(tmp_path: Path) -> None:
    assert seed_voice_names(tmp_path / "nope") == set()


def test_seed_voice_names_uses_env_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "narrator.wav").write_bytes(b"RIFF")
    monkeypatch.setenv("PERSONAVOICE_SEED_VOICES_DIR", str(tmp_path))
    assert seed_voice_names() == {"narrator"}


# --- make_enroller (per-backend) ------------------------------------------------------


async def test_enroller_cloning_backend_uses_cloner(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    store = ClonesStore(tmp_path / "clones")
    backend = dataclasses.replace(make_backend(stt_text="my voice"), tts=CloningTTS())
    enroll = make_enroller(backend, store, supports_cloning=True)
    await enroll("female", _wav(5.0))
    # The full cloner path persisted the clone with a reference transcript.
    cv = store.get("female")
    assert cv is not None and cv.ref_text == "my voice"


async def test_enroller_preset_only_records_catalog_entry(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    store = ClonesStore(tmp_path / "clones")
    backend = make_backend()  # FakeTTS.supports_cloning is False
    enroll = make_enroller(backend, store, supports_cloning=False)
    await enroll("male", _wav(5.0))
    cv = store.get("male")
    assert cv is not None
    # Sample was persisted under the clones dir so the catalog can list it (available=false).
    assert Path(cv.sample_path).is_file()


async def test_enroller_preset_only_rejects_bad_sample(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    store = ClonesStore(tmp_path / "clones")
    enroll = make_enroller(make_backend(), store, supports_cloning=False)
    with pytest.raises(CloneError, match="too short"):
        await enroll("female", _wav(1.0))


def test_seed_result_ok_default() -> None:
    assert SeedResult().ok is True

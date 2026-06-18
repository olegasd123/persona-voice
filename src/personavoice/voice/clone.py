"""Zero-shot voice cloning: clone a voice from a short sample and assign it to a persona.

Cloning here is **reference conditioning**, not training: the cloning TTS backends
(Chatterbox on CUDA, F5 on Mac) synthesize *in the voice of* a short reference WAV passed at
generation time. So a "clone" is just a stored ~10 s sample plus, for reference-text models
(F5), its transcript. The pieces:

- `validate_sample` — decode the WAV and bound its duration (clear-speech sanity check).
- `ClonesStore` — persists clones (sample path + transcript + provenance) and **per-persona
  assignments** to `<clones_dir>/clones.json`, so a clone survives a restart and the live
  agent picks it up automatically (`VoiceRegistry.resolve_for_persona`).
- `VoiceCloner` — orchestrates one clone: validate → `TTSAdapter.clone_voice` (persist the
  sample, get a `VoiceRef`) → transcribe the sample with the cascade's STT for the reference
  text → record in the store (and optionally assign to a persona).

The store is intentionally decoupled from the persona/voices YAML: assigning a clone never
rewrites a hand-authored config file; it's a non-destructive overlay consulted at resolve
time, and only honored on a backend that can actually clone (else the persona keeps its
registry preset).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..models import VoiceRef

logger = logging.getLogger("personavoice.clone")

_MANIFEST_NAME = "clones.json"
# A clone name (and persona id) is a bare identifier — keeps it usable as a filename and a
# `voices/<name>` ref without escaping.
_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")

# Clear-speech sample bounds. Too short → the clone has no timbre to work with; too long →
# wasted conditioning latency (a clean ~10 s clip is the sweet spot for zero-shot).
_MIN_SECONDS = 2.0
_MAX_SECONDS = 60.0


class CloneError(ValueError):
    """A sample is unusable, a name/assignment is invalid, or the manifest is corrupt."""


def validate_sample(
    sample_wav: bytes,
    *,
    min_seconds: float | None = _MIN_SECONDS,
    max_seconds: float | None = _MAX_SECONDS,
) -> float:
    """Decode `sample_wav` and return its duration (s), raising `CloneError` if unusable.

    Pass `min_seconds=None`/`max_seconds=None` to skip a bound (the duration is still
    returned). Needs an audio extra (numpy/soundfile); that dependency error propagates.
    """
    from ..audio import AudioDependencyError, decode_wav

    if not sample_wav:
        raise CloneError("empty audio sample")
    try:
        samples, sr = decode_wav(sample_wav)
    except AudioDependencyError:
        raise
    except Exception as exc:  # malformed/non-WAV bytes
        raise CloneError(f"could not decode the audio sample as WAV: {exc}") from exc

    duration = len(samples) / sr if sr else 0.0
    if min_seconds is not None and duration < min_seconds:
        raise CloneError(
            f"sample too short ({duration:.1f}s); record at least {min_seconds:.0f}s "
            "of clear speech"
        )
    if max_seconds is not None and duration > max_seconds:
        raise CloneError(
            f"sample too long ({duration:.1f}s); trim it to at most {max_seconds:.0f}s"
        )
    return duration


class ClonedVoice(BaseModel):
    """One cloned voice: a stored reference sample plus what backends need to use it."""

    model_config = ConfigDict(extra="ignore")

    name: str
    sample_path: str
    ref_text: str | None = None  # transcript of the sample (reference-text backends)
    backend: str | None = None  # which TTS produced it (provenance; samples are portable)
    created_at: str | None = None


class _Manifest(BaseModel):
    """On-disk shape of `clones.json`."""

    model_config = ConfigDict(extra="ignore")

    voices: dict[str, ClonedVoice] = Field(default_factory=dict)
    assignments: dict[str, str] = Field(default_factory=dict)  # persona id -> clone name


class ClonesStore:
    """Persisted catalog of cloned voices and their per-persona assignments."""

    def __init__(
        self,
        directory: str | Path,
        voices: dict[str, ClonedVoice] | None = None,
        assignments: dict[str, str] | None = None,
    ) -> None:
        self._dir = Path(directory).expanduser()
        self._voices = voices or {}
        self._assignments = assignments or {}

    # -- loading / saving --------------------------------------------------------------
    @classmethod
    def load(cls, directory: str | Path) -> ClonesStore:
        """Load the store from `<directory>/clones.json` (a missing file → empty store)."""
        directory = Path(directory).expanduser()
        path = directory / _MANIFEST_NAME
        if not path.is_file():
            return cls(directory)
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise CloneError(f"{path}: invalid clones manifest: {exc}") from exc
        try:
            manifest = _Manifest.model_validate(raw)
        except ValidationError as exc:
            raise CloneError(f"{path}: invalid clones manifest: {exc}") from exc
        return cls(directory, dict(manifest.voices), dict(manifest.assignments))

    def save(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        manifest = _Manifest(voices=self._voices, assignments=self._assignments)
        (self._dir / _MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2))

    # -- voices ------------------------------------------------------------------------
    @property
    def dir(self) -> Path:
        return self._dir

    def record(self, voice: ClonedVoice) -> None:
        """Add or replace a clone, then persist."""
        self._voices[voice.name] = voice
        self.save()

    def get(self, name: str) -> ClonedVoice | None:
        return self._voices.get(name)

    def names(self) -> list[str]:
        return sorted(self._voices)

    def voice_ref(self, name: str, tts_name: str, *, emotion: str | None = None) -> VoiceRef | None:
        """Build the `VoiceRef` a clone speaks with on `tts_name`, or None if unknown."""
        cv = self._voices.get(name)
        if cv is None:
            return None
        return VoiceRef(
            id=cv.name,
            name=cv.name,
            sample_path=cv.sample_path or None,
            ref_text=cv.ref_text,
            emotion=emotion,
            backend=tts_name,
        )

    # -- assignments -------------------------------------------------------------------
    def assign(self, persona_id: str, name: str) -> None:
        """Assign clone `name` to persona `persona_id` (must already be cloned), then save."""
        if name not in self._voices:
            known = ", ".join(self.names()) or "(none)"
            raise CloneError(f"unknown clone {name!r}; cloned voices: {known}")
        self._assignments[persona_id] = name
        self.save()

    def unassign(self, persona_id: str) -> bool:
        """Drop any clone assigned to `persona_id`. Returns True if one was removed."""
        removed = self._assignments.pop(persona_id, None) is not None
        if removed:
            self.save()
        return removed

    def assignment_for(self, persona_id: str) -> str | None:
        return self._assignments.get(persona_id)

    @property
    def assignments(self) -> dict[str, str]:
        return dict(self._assignments)

    def __len__(self) -> int:
        return len(self._voices)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._voices


class VoiceCloner:
    """Clone a voice through the active backend and record it in a `ClonesStore`."""

    def __init__(self, backend: object, store: ClonesStore) -> None:
        # `backend` is an adapters.factory.Backend; typed loosely to avoid a hard import here.
        self._backend = backend
        self._store = store

    async def clone(
        self,
        sample_wav: bytes,
        name: str,
        *,
        ref_text: str | None = None,
        transcribe: bool = True,
        assign_to: str | None = None,
        min_seconds: float | None = _MIN_SECONDS,
        max_seconds: float | None = _MAX_SECONDS,
    ) -> VoiceRef:
        """Clone `sample_wav` as `name`, persist it, and optionally assign it to a persona.

        Validates the sample, asks the TTS backend to persist it and produce a `VoiceRef`,
        fills the reference transcript (given, or transcribed via the cascade's STT for
        reference-text backends), records the clone, and assigns it when `assign_to` is set.
        """
        if not name or not _NAME_RE.fullmatch(name):
            raise CloneError(
                f"invalid clone name {name!r}; use letters, digits, '-' or '_' only"
            )
        tts = self._backend.tts  # type: ignore[attr-defined]
        if not getattr(tts, "supports_cloning", False):
            raise CloneError(
                f"the active TTS backend {tts.name!r} can't clone voices; switch to a cloning "
                "backend (f5_mlx on Mac, chatterbox on CUDA) in config/backends/<backend>.yaml"
            )
        if min_seconds is not None or max_seconds is not None:
            validate_sample(sample_wav, min_seconds=min_seconds, max_seconds=max_seconds)

        # Land the clone sample alongside the manifest so the catalog is self-contained.
        tts.options["clones_dir"] = str(self._store.dir)
        voice = await tts.clone_voice(sample_wav, name)

        if ref_text is None and transcribe:
            ref_text = await self._transcribe(sample_wav)
        if ref_text and not voice.ref_text:
            voice = voice.model_copy(update={"ref_text": ref_text})

        self._store.record(
            ClonedVoice(
                name=name,
                sample_path=voice.sample_path or "",
                ref_text=voice.ref_text,
                backend=getattr(tts, "name", None),
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
        )
        if assign_to:
            self._store.assign(assign_to, name)
        return voice

    async def _transcribe(self, sample_wav: bytes) -> str | None:
        """Best-effort transcript of the sample (a clone still works without one)."""
        try:
            transcript = await self._backend.stt.transcribe(sample_wav)  # type: ignore[attr-defined]
        except Exception as exc:  # STT is optional for cloning; don't fail the clone
            logger.warning("could not transcribe clone sample for reference text: %s", exc)
            return None
        return transcript.text.strip() or None

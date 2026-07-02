"""Zero-shot voice cloning: clone a voice from a short sample and assign it to a persona.

Cloning here is **reference conditioning**, not training: the cloning TTS backend
synthesizes *in the voice of* a short reference WAV passed at generation time. So a "clone"
is just a stored ~10 s sample. The pieces:

- `validate_sample` — decode the WAV and bound its duration (clear-speech sanity check).
- `ClonesStore` — persists clones (sample path + provenance) and **per-persona
  assignments** to `<clones_dir>/clones.json`, so a clone survives a restart and the live
  agent picks it up automatically (`VoiceRegistry.resolve_for_persona`).
- `VoiceCloner` — orchestrates one clone: validate → `TTSAdapter.clone_voice` (persist the
  sample, get a `VoiceRef`) → record in the store (and optionally assign to a persona).

The store is intentionally decoupled from the persona/voices YAML: assigning a clone never
rewrites a hand-authored config file; it's a non-destructive overlay consulted at resolve
time, and only honored on a backend that can actually clone (else the persona keeps its
registry preset).
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .._fsio import atomic_write_text
from ..models import VoiceRef

if TYPE_CHECKING:
    import numpy as np  # annotations only; the runtime import is lazy inside functions

_MANIFEST_NAME = "clones.json"
# A clone name (and persona id) is a bare identifier — keeps it usable as a filename and a
# `voices/<name>` ref without escaping.
_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")

# Clear-speech sample bounds. Too short → the clone has no timbre to work with; too long →
# wasted conditioning latency (a clean ~10 s clip is the sweet spot for zero-shot).
_MIN_SECONDS = 2.0
_MAX_SECONDS = 60.0

# Cap the *stored* reference length. A clean ~5 s clip is the sweet spot for Chatterbox
# conditioning; longer can even hurt. None disables trimming.
_TRIM_REF_SECONDS = 5.0


def trim_wav(
    sample_wav: bytes,
    max_seconds: float,
    *,
    search_seconds: float = 1.2,
    tail_silence_seconds: float = 0.2,
    fade_seconds: float = 0.015,
) -> bytes:
    """Return `sample_wav` trimmed to about `max_seconds`, ending on a clean pause.

    A naive hard cut at an arbitrary point can end the clip mid-word, which gives the model
    a poor reference boundary. So we cut at the quietest point within the last
    `search_seconds` before the cap, fade the tail to kill any click, and append a short
    trailing silence so the reference ends on a pause.

    Unchanged when it's already shorter, or when the bytes can't be decoded — trimming is a
    best-effort latency optimization, so undecodable input is left for the backend to handle.
    """
    import numpy as np

    from ..audio import decode_wav, encode_wav

    try:
        samples, sr = decode_wav(sample_wav)
    except Exception:  # not decodable here → let the backend deal with the raw bytes
        return sample_wav
    target = int(max_seconds * sr)
    if target <= 0 or len(samples) <= target:
        return sample_wav

    cut = _quiet_cut(samples, max(0, target - int(search_seconds * sr)), target, sr)
    clip = samples[:cut].astype(np.float32, copy=True)
    fade = min(int(fade_seconds * sr), len(clip))
    if fade > 0:
        clip[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    pad = int(tail_silence_seconds * sr)
    if pad > 0:
        clip = np.concatenate([clip, np.zeros(pad, dtype=np.float32)])
    return encode_wav(clip, sr)


def _quiet_cut(
    samples: np.ndarray, lo: int, hi: int, sr: int, *, frame_seconds: float = 0.02
) -> int:
    """Index of a low-energy cut point in `samples[lo:hi]`, biased toward `hi`.

    Splits the window into short frames, then picks the *latest* frame whose RMS is within ~1.5x
    of the quietest (so we cut at a pause near the cap rather than the earliest dip). Falls back
    to `hi` when the window is too small to frame.
    """
    import numpy as np

    frame = max(1, int(frame_seconds * sr))
    region = samples[lo:hi]
    n = len(region) // frame
    if n == 0:
        return hi
    energy = np.sqrt((region[: n * frame].reshape(n, frame).astype(np.float32) ** 2).mean(axis=1))
    quiet = np.where(energy <= energy.min() * 1.5 + 1e-6)[0]
    q = int(quiet[-1]) if len(quiet) else int(energy.argmin())
    return lo + (q + 1) * frame


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
        atomic_write_text(self._dir / _MANIFEST_NAME, manifest.model_dump_json(indent=2))

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

    def remove(self, name: str) -> bool:
        """Delete clone `name` and drop any persona assignments to it, then persist.

        Returns True if a clone was removed. Assignments pointing at the removed clone are
        cleared too, so a persona doesn't keep a dangling reference.
        """
        if name not in self._voices:
            return False
        del self._voices[name]
        for persona_id in [pid for pid, n in self._assignments.items() if n == name]:
            del self._assignments[persona_id]
        self.save()
        return True

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
        assign_to: str | None = None,
        min_seconds: float | None = _MIN_SECONDS,
        max_seconds: float | None = _MAX_SECONDS,
        trim_ref_seconds: float | None = _TRIM_REF_SECONDS,
    ) -> VoiceRef:
        """Clone `sample_wav` as `name`, persist it, and optionally assign it to a persona.

        Validates the sample, caps its length (`trim_ref_seconds`, for responsiveness — see
        `_TRIM_REF_SECONDS`), asks the TTS backend to persist it and produce a `VoiceRef`,
        records the clone, and assigns it when `assign_to` is set.
        """
        if not name or not _NAME_RE.fullmatch(name):
            raise CloneError(f"invalid clone name {name!r}; use letters, digits, '-' or '_' only")
        tts = self._backend.tts  # type: ignore[attr-defined]
        if not getattr(tts, "supports_cloning", False):
            raise CloneError(
                f"the active TTS backend {tts.name!r} can't clone voices; switch to a cloning "
                "backend (chatterbox on CUDA) in config/backends/<backend>.yaml"
            )
        if min_seconds is not None or max_seconds is not None:
            validate_sample(sample_wav, min_seconds=min_seconds, max_seconds=max_seconds)
        if trim_ref_seconds is not None:
            sample_wav = trim_wav(sample_wav, trim_ref_seconds)

        # Land the clone sample alongside the manifest so the catalog is self-contained.
        tts.options["clones_dir"] = str(self._store.dir)
        voice = await tts.clone_voice(sample_wav, name)

        self._store.record(
            ClonedVoice(
                name=name,
                sample_path=voice.sample_path or "",
                backend=getattr(tts, "name", None),
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
        )
        if assign_to:
            self._store.assign(assign_to, name)
        return voice

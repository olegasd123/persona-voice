"""Fine-tuned voices: a persisted catalog of high-fidelity, trained voices.

Where a *clone* is reference conditioning — the cloning TTS speaks in the voice of a
stored sample passed at generation time — a **fine-tuned voice** is a trained checkpoint: the
TTS model itself is adapted to a target speaker on a small dataset, for fidelity beyond
zero-shot. At inference the cloning backends load that checkpoint (`VoiceRef.model_path`)
instead of their base weights.

`FinetunedVoicesStore` mirrors `ClonesStore`: it persists fine-tuned voices (checkpoint path
+ engine + provenance) and **per-persona assignments** to `<finetuned_dir>/finetuned.json`,
non-destructively (it never rewrites a hand-authored persona/voices YAML). `VoiceRegistry`
consults it before clones and static presets, but only on a backend that can actually load a
fine-tuned checkpoint (the cloning backends — Chatterbox/F5).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..models import VoiceRef

_MANIFEST_NAME = "finetuned.json"
# A fine-tuned voice name (and persona id) is a bare identifier — usable as a filename and a
# `voices/<name>` ref without escaping.
_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")


class FinetunedVoiceError(ValueError):
    """A name/assignment is invalid, or the manifest is corrupt."""


class FinetunedVoice(BaseModel):
    """One fine-tuned voice: a trained checkpoint a cloning backend loads to speak."""

    model_config = ConfigDict(extra="ignore")

    name: str
    checkpoint_path: str  # the trained model/checkpoint dir the TTS backend loads
    engine: str | None = None  # "f5" | "chatterbox" (which trainer produced it)
    base_model: str | None = None  # the base checkpoint it was fine-tuned from
    speaker: str | None = None  # target speaker label (provenance)
    # A/B verdict vs the zero-shot clone, when an eval was run.
    similarity: float | None = None  # speaker similarity to held-out target clips
    clone_similarity: float | None = None  # the zero-shot clone's similarity, for the delta
    created_at: str | None = None


class _Manifest(BaseModel):
    """On-disk shape of `finetuned.json`."""

    model_config = ConfigDict(extra="ignore")

    voices: dict[str, FinetunedVoice] = Field(default_factory=dict)
    assignments: dict[str, str] = Field(default_factory=dict)  # persona id -> voice name


class FinetunedVoicesStore:
    """Persisted catalog of fine-tuned voices and their per-persona assignments."""

    def __init__(
        self,
        directory: str | Path,
        voices: dict[str, FinetunedVoice] | None = None,
        assignments: dict[str, str] | None = None,
    ) -> None:
        self._dir = Path(directory).expanduser()
        self._voices = voices or {}
        self._assignments = assignments or {}

    # -- loading / saving --------------------------------------------------------------
    @classmethod
    def load(cls, directory: str | Path) -> FinetunedVoicesStore:
        """Load from `<directory>/finetuned.json` (a missing file → empty store)."""
        directory = Path(directory).expanduser()
        path = directory / _MANIFEST_NAME
        if not path.is_file():
            return cls(directory)
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise FinetunedVoiceError(f"{path}: invalid finetuned manifest: {exc}") from exc
        try:
            manifest = _Manifest.model_validate(raw)
        except ValidationError as exc:
            raise FinetunedVoiceError(f"{path}: invalid finetuned manifest: {exc}") from exc
        return cls(directory, dict(manifest.voices), dict(manifest.assignments))

    def save(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        manifest = _Manifest(voices=self._voices, assignments=self._assignments)
        (self._dir / _MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2))

    # -- voices ------------------------------------------------------------------------
    @property
    def dir(self) -> Path:
        return self._dir

    def record(self, voice: FinetunedVoice) -> None:
        """Add or replace a fine-tuned voice, then persist."""
        if not voice.name or not _NAME_RE.fullmatch(voice.name):
            raise FinetunedVoiceError(
                f"invalid voice name {voice.name!r}; use letters, digits, '-' or '_' only"
            )
        self._voices[voice.name] = voice
        self.save()

    def get(self, name: str) -> FinetunedVoice | None:
        return self._voices.get(name)

    def names(self) -> list[str]:
        return sorted(self._voices)

    def voice_ref(self, name: str, tts_name: str, *, emotion: str | None = None) -> VoiceRef | None:
        """Build the `VoiceRef` a fine-tuned voice speaks with on `tts_name`, or None."""
        fv = self._voices.get(name)
        if fv is None:
            return None
        return VoiceRef(
            id=fv.name,
            name=fv.name,
            model_path=fv.checkpoint_path or None,
            emotion=emotion,
            backend=tts_name,
        )

    # -- assignments -------------------------------------------------------------------
    def assign(self, persona_id: str, name: str) -> None:
        """Assign fine-tuned voice `name` to `persona_id` (must exist), then save."""
        if name not in self._voices:
            known = ", ".join(self.names()) or "(none)"
            raise FinetunedVoiceError(f"unknown fine-tuned voice {name!r}; known: {known}")
        self._assignments[persona_id] = name
        self.save()

    def unassign(self, persona_id: str) -> bool:
        """Drop any fine-tuned voice assigned to `persona_id`. True if one was removed."""
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

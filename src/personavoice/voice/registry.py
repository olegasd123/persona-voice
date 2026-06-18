"""The voice registry: resolve a persona's voice ref to a concrete backend voice.

A persona declares a *logical* voice (`voice.ref: voices/companion_soft`). The registry
(`config/voices.yaml`) maps that logical voice to a backend-native preset per TTS adapter,
so the four personas sound **distinct** on whichever backend is active — `af_heart` vs
`am_michael` on Kokoro (Mac), `tara` vs `leo` on Orpheus (CUDA). Clone-only backends
(Chatterbox/F5) have no preset; they fall back to their own default voice until zero-shot
cloning is wired up in M5, at which point the `sample` field feeds the clone.

`resolve()` always returns a usable `VoiceRef`: a concrete preset when one exists for the
backend, otherwise the original ref passed through (the adapter then uses its default). The
registry tolerates a missing file (returns empty) so partial configs and tests still run.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import ValidationError

from ..models import Persona, VoiceDef, VoiceRef

if TYPE_CHECKING:
    from .clone import ClonesStore
    from .finetuned import FinetunedVoicesStore

_VOICE_PREFIX = "voices/"


class VoiceError(ValueError):
    """The voice registry file is missing data or fails validation."""


class VoiceRegistry:
    """Maps logical voice ids to backend-native voices."""

    def __init__(
        self,
        voices: dict[str, VoiceDef],
        *,
        clones: ClonesStore | None = None,
        finetuned: FinetunedVoicesStore | None = None,
    ) -> None:
        self._voices = voices
        # Cloned voices + per-persona assignments (M5). Consulted before the static preset
        # in `resolve_for_persona`, but only on a backend that can actually clone.
        self._clones = clones
        # Fine-tuned voices + assignments (M9). Highest precedence — a trained checkpoint
        # beats a zero-shot clone beats a static preset — again only on a cloning backend.
        self._finetuned = finetuned

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        clones: ClonesStore | None = None,
        finetuned: FinetunedVoicesStore | None = None,
    ) -> VoiceRegistry:
        """Load `config/voices.yaml`. A missing file yields an empty registry."""
        path = Path(path)
        if not path.is_file():
            return cls({}, clones=clones, finetuned=finetuned)
        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise VoiceError(f"{path}: invalid YAML: {exc}") from exc
        if not isinstance(raw, dict):
            raise VoiceError(f"{path}: expected a mapping of voice id -> definition")
        try:
            voices = {vid: VoiceDef.model_validate(body or {}) for vid, body in raw.items()}
        except ValidationError as exc:
            raise VoiceError(f"{path}: invalid voice registry: {exc}") from exc
        return cls(voices, clones=clones, finetuned=finetuned)

    @staticmethod
    def _key(ref: str) -> str:
        """Strip the `voices/` prefix personas use, leaving the registry key."""
        return ref[len(_VOICE_PREFIX) :] if ref.startswith(_VOICE_PREFIX) else ref

    def resolve(
        self, ref: str, tts_name: str, *, default_emotion: str | None = None
    ) -> VoiceRef:
        """Resolve a persona's voice `ref` to the voice it should speak with on `tts_name`.

        Falls back to passing `ref` through unchanged when the voice is unknown or has no
        preset for this backend — the adapter then uses its configured default voice.
        """
        entry = self._voices.get(self._key(ref))
        if entry is None:
            return VoiceRef(id=ref, emotion=default_emotion, backend=tts_name)
        preset = entry.presets.get(tts_name)
        return VoiceRef(
            id=preset or ref,  # no backend preset -> let the adapter default
            name=self._key(ref),
            sample_path=entry.sample,
            emotion=entry.emotion or default_emotion,
            backend=tts_name,
        )

    def resolve_for_persona(
        self, persona: Persona, tts_name: str, *, supports_cloning: bool
    ) -> VoiceRef:
        """Resolve the voice a `persona` speaks with, in precedence order.

        On a backend that can synthesize from a reference/checkpoint (the cloning backends),
        a **fine-tuned voice** assigned to this persona (M9) wins — the persona speaks through
        a trained checkpoint. Failing that, an assigned zero-shot **clone** (M5) wins. Failing
        both, this falls back to the static per-backend **preset** (`resolve`), so a
        non-cloning backend keeps its distinct presets.
        """
        if supports_cloning:
            if self._finetuned is not None:
                assigned = self._finetuned.assignment_for(persona.id)
                if assigned:
                    ref = self._finetuned.voice_ref(
                        assigned, tts_name, emotion=persona.voice.emotion
                    )
                    if ref is not None:
                        return ref
            if self._clones is not None:
                assigned = self._clones.assignment_for(persona.id)
                if assigned:
                    ref = self._clones.voice_ref(
                        assigned, tts_name, emotion=persona.voice.emotion
                    )
                    if ref is not None:
                        return ref
        return self.resolve(
            persona.voice.ref, tts_name, default_emotion=persona.voice.emotion
        )

    def has_preset(self, ref: str, tts_name: str) -> bool:
        """True when this voice maps to a concrete preset for `tts_name`."""
        entry = self._voices.get(self._key(ref))
        return bool(entry and tts_name in entry.presets)

    @property
    def clones(self) -> ClonesStore | None:
        return self._clones

    def clone_for_persona(self, persona_id: str) -> str | None:
        """Name of the clone assigned to `persona_id`, if any."""
        return self._clones.assignment_for(persona_id) if self._clones is not None else None

    @property
    def finetuned(self) -> FinetunedVoicesStore | None:
        return self._finetuned

    def finetuned_for_persona(self, persona_id: str) -> str | None:
        """Name of the fine-tuned voice assigned to `persona_id`, if any (M9)."""
        return (
            self._finetuned.assignment_for(persona_id) if self._finetuned is not None else None
        )

    def ids(self) -> list[str]:
        return sorted(self._voices)

    def __len__(self) -> int:
        return len(self._voices)

    def __contains__(self, ref: object) -> bool:
        return isinstance(ref, str) and self._key(ref) in self._voices

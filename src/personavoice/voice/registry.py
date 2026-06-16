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

import yaml
from pydantic import ValidationError

from ..models import VoiceDef, VoiceRef

_VOICE_PREFIX = "voices/"


class VoiceError(ValueError):
    """The voice registry file is missing data or fails validation."""


class VoiceRegistry:
    """Maps logical voice ids to backend-native voices."""

    def __init__(self, voices: dict[str, VoiceDef]) -> None:
        self._voices = voices

    @classmethod
    def load(cls, path: str | Path) -> VoiceRegistry:
        """Load `config/voices.yaml`. A missing file yields an empty registry."""
        path = Path(path)
        if not path.is_file():
            return cls({})
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
        return cls(voices)

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

    def has_preset(self, ref: str, tts_name: str) -> bool:
        """True when this voice maps to a concrete preset for `tts_name`."""
        entry = self._voices.get(self._key(ref))
        return bool(entry and tts_name in entry.presets)

    def ids(self) -> list[str]:
        return sorted(self._voices)

    def __len__(self) -> int:
        return len(self._voices)

    def __contains__(self, ref: object) -> bool:
        return isinstance(ref, str) and self._key(ref) in self._voices

"""The voice registry: resolve a persona's voice ref to a concrete backend voice.

A persona declares a *logical* voice (`voice.ref: voices/companion_soft`). The registry
(`config/voices.yaml`) maps that logical voice to a backend-native preset per TTS adapter,
so the four personas sound **distinct** on whichever backend is active — `af_heart` vs
`am_michael` on Kokoro (Mac). Clone-only backends
(Chatterbox/F5) have no preset; they fall back to their own default voice until zero-shot
cloning is wired up separately, at which point the `sample` field feeds the clone.

`resolve()` always returns a usable `VoiceRef`: a concrete preset when one exists for the
backend, otherwise the original ref passed through (the adapter then uses its default). The
registry tolerates a missing file (returns empty) so partial configs and tests still run.
"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, ValidationError

from ..models import Persona, VoiceDef, VoiceRef

if TYPE_CHECKING:
    from .clone import ClonesStore
    from .finetuned import FinetunedVoicesStore

_VOICE_PREFIX = "voices/"

# Reason shown for a clone/fine-tune the active backend can't speak (no cloning capability).
_NO_CLONING_REASON = "requires a cloning backend (f5_mlx on Mac, chatterbox on CUDA)"

# Which fine-tune engine each cloning TTS backend can load a checkpoint for. A fine-tuned voice
# trained with a different engine isn't loadable (an F5 checkpoint isn't a Chatterbox one) — left
# selectable it crashes the synth, so it's gated out of resolution and shown unavailable.
_TTS_FINETUNE_ENGINE = {"chatterbox": "chatterbox", "f5_mlx": "f5"}


def _finetune_speakable(engine: str | None, tts_name: str) -> bool:
    """True when a fine-tuned voice's `engine` can be loaded by the `tts_name` backend.

    A voice with no recorded engine (legacy) is treated as loadable — we can't prove a mismatch,
    so we don't hide it. A known engine must match the active backend's engine.
    """
    backend_engine = _TTS_FINETUNE_ENGINE.get(tts_name)
    if backend_engine is None:
        return False  # backend can't load any fine-tune checkpoint
    return engine is None or engine == backend_engine


def _engine_mismatch_reason(engine: str | None, tts_name: str) -> str:
    """Why an engine-mismatched fine-tuned voice can't be spoken on the active backend."""
    return f"trained with the {engine} engine; the active {tts_name} backend can't load it"


class VoiceError(ValueError):
    """The voice registry file is missing data or fails validation."""


class VoiceOption(BaseModel):
    """One selectable voice in the catalog the client shows in a picker.

    `available` reflects whether this voice is actually speakable on the *active* TTS backend
    (presets need a per-backend mapping; clones/fine-tunes need a cloning backend). When it's
    not, `reason` explains why so the UI can grey it out with a hint rather than silently drop
    it.
    """

    id: str
    name: str
    kind: str  # "preset" | "clone" | "finetuned"
    emotion: str | None = None
    available: bool = True
    reason: str | None = None
    # Whether the user may delete this voice from the library. Only user-enrolled clones are
    # removable; presets/fine-tunes and the bundled "inbox" seed clones are not (the UI hides
    # their delete control and the server rejects a delete).
    removable: bool = False


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
        # Cloned voices + per-persona assignments. Consulted before the static preset
        # in `resolve_for_persona`, but only on a backend that can actually clone.
        self._clones = clones
        # Fine-tuned voices + assignments. Highest precedence — a trained checkpoint
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

    def resolve(self, ref: str, tts_name: str, *, default_emotion: str | None = None) -> VoiceRef:
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
        a **fine-tuned voice** assigned to this persona wins — the persona speaks through
        a trained checkpoint. Failing that, an assigned zero-shot **clone** wins. Failing
        both, this falls back to the static per-backend **preset** (`resolve`), so a
        non-cloning backend keeps its distinct presets.
        """
        if supports_cloning:
            if self._finetuned is not None:
                assigned = self._finetuned.assignment_for(persona.id)
                # Skip an assigned fine-tune the active backend can't load (engine mismatch) so
                # it falls through to a clone / static preset instead of crashing at synth time.
                if assigned and _finetune_speakable(self._finetuned_engine(assigned), tts_name):
                    ref = self._finetuned.voice_ref(
                        assigned, tts_name, emotion=persona.voice.emotion
                    )
                    if ref is not None:
                        return ref
            if self._clones is not None:
                assigned = self._clones.assignment_for(persona.id)
                if assigned:
                    ref = self._clones.voice_ref(assigned, tts_name, emotion=persona.voice.emotion)
                    if ref is not None:
                        return ref
        return self.resolve(persona.voice.ref, tts_name, default_emotion=persona.voice.emotion)

    def resolve_choice(
        self,
        voice_id: str,
        tts_name: str,
        *,
        supports_cloning: bool,
        default_emotion: str | None = None,
    ) -> VoiceRef | None:
        """Resolve an explicitly chosen voice id, independent of any persona assignment.

        Precedence mirrors `resolve_for_persona`: fine-tuned store → clones store → preset.
        Returns `None` when the choice is unknown or **not speakable** on this backend (a
        clone/fine-tune on a non-cloning backend, or a preset with no mapping for `tts_name`),
        so the caller falls back to the persona's default voice.
        """
        key = self._key(voice_id)
        if supports_cloning:
            if (
                self._finetuned is not None
                and key in self._finetuned
                and _finetune_speakable(self._finetuned_engine(key), tts_name)
            ):
                ref = self._finetuned.voice_ref(key, tts_name, emotion=default_emotion)
                if ref is not None:
                    return ref
            if self._clones is not None and key in self._clones:
                ref = self._clones.voice_ref(key, tts_name, emotion=default_emotion)
                if ref is not None:
                    return ref
        if key in self._voices and self.has_preset(key, tts_name):
            return self.resolve(key, tts_name, default_emotion=default_emotion)
        return None

    def catalog(
        self, tts_name: str, *, supports_cloning: bool, protected: Collection[str] = ()
    ) -> list[VoiceOption]:
        """The unified, selectable voice list for a picker on the active backend.

        Stable order: fine-tuned, then clones, then presets. Clones/fine-tunes are listed even
        on a non-cloning backend but marked `available=False` with a `reason` (so the UI can
        hint "switch to a cloning backend"). Presets, by contrast, are **omitted entirely**
        when they have no mapping for `tts_name`: an unmapped preset (e.g. a Kokoro preset on
        F5) is nothing the user can act on, so it's dropped rather than shown greyed-out.

        `protected` names the clones the user may **not** delete (the bundled "inbox" seed
        voices); those are marked `removable=False` like presets/fine-tunes.
        """
        options: list[VoiceOption] = []
        cloning_reason = None if supports_cloning else _NO_CLONING_REASON
        if self._finetuned is not None:
            for name in self._finetuned.names():
                engine = self._finetuned_engine(name)
                speakable = supports_cloning and _finetune_speakable(engine, tts_name)
                if not supports_cloning:
                    reason = cloning_reason
                elif not speakable:
                    reason = _engine_mismatch_reason(engine, tts_name)
                else:
                    reason = None
                options.append(
                    VoiceOption(
                        id=name,
                        name=name,
                        kind="finetuned",
                        available=speakable,
                        reason=reason,
                    )
                )
        if self._clones is not None:
            for name in self._clones.names():
                options.append(
                    VoiceOption(
                        id=name,
                        name=name,
                        kind="clone",
                        available=supports_cloning,
                        reason=cloning_reason,
                        removable=name not in protected,
                    )
                )
        for vid in self.ids():
            entry = self._voices[vid]
            # Drop presets with no mapping for the active backend instead of listing them
            # unavailable — see the docstring. Clones/fine-tunes above stay listed-but-greyed.
            if tts_name not in entry.presets:
                continue
            options.append(
                VoiceOption(
                    id=vid,
                    name=entry.description or vid,
                    kind="preset",
                    emotion=entry.emotion,
                )
            )
        return options

    def has_preset(self, ref: str, tts_name: str) -> bool:
        """True when this voice maps to a concrete preset for `tts_name`."""
        entry = self._voices.get(self._key(ref))
        return bool(entry and tts_name in entry.presets)

    def describe(self, ref: str) -> str:
        """Human description of a voice `ref` (`config/voices.yaml` `description`).

        Backend-independent — the character of the voice ("warm, soft, feminine"), not the
        per-backend preset. Empty string when the voice is unknown, so callers can treat it
        as "no detail" without special-casing.
        """
        entry = self._voices.get(self._key(ref))
        return entry.description if entry else ""

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
        """Name of the fine-tuned voice assigned to `persona_id`, if any."""
        return self._finetuned.assignment_for(persona_id) if self._finetuned is not None else None

    def _finetuned_engine(self, name: str) -> str | None:
        """The training engine of fine-tuned voice `name` (None if unknown/absent)."""
        fv = self._finetuned.get(name) if self._finetuned is not None else None
        return fv.engine if fv is not None else None

    def preset_sample_stems(self) -> set[str]:
        """File stems of preset sample wavs (bundled voices shipped via `voices.yaml`).

        The seeder excludes these so a bundled preset wav under the seed dir isn't *also*
        enrolled as a duplicate clone; the token server excludes them from the protected-clone
        set (a preset is already non-removable). Presets with no `sample` (Kokoro presets) and
        clones/fine-tunes don't contribute.
        """
        return {Path(v.sample).stem for v in self._voices.values() if v.sample}

    def ids(self) -> list[str]:
        return sorted(self._voices)

    def choice_ids(self) -> list[str]:
        """Every selectable voice id (presets + clones + fine-tunes), for choice validation.

        Backend-independent — a clone listed here may still be unspeakable on a non-cloning
        backend (that's gated at resolve time / surfaced by `catalog`). It answers "is this a
        known voice the user could pick?", which is what the token server validates against.
        """
        ids = set(self._voices)
        if self._clones is not None:
            ids.update(self._clones.names())
        if self._finetuned is not None:
            ids.update(self._finetuned.names())
        return sorted(ids)

    def __len__(self) -> int:
        return len(self._voices)

    def __contains__(self, ref: object) -> bool:
        return isinstance(ref, str) and self._key(ref) in self._voices

"""`server --check`: validate config and load (stub) adapters on this machine.

Produces a structured `CheckReport` so the same logic is exercised by tests and by the
CLI. This is the M0 acceptance test: it must pass on both the Mac and the 4080.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..adapters.factory import Backend, build_backend
from ..memory import MemoryStoreError
from ..models import CheckResult, Persona
from ..voice.registry import VoiceRegistry
from .config import (
    ConfigError,
    Settings,
    load_all_personas,
    load_backend_config,
    load_memory_store,
    load_voice_registry,
)


@dataclass
class CheckReport:
    backend_name: str = ""
    adapter_results: list[CheckResult] = field(default_factory=list)
    personas: list[str] = field(default_factory=list)
    voices: list[str] = field(default_factory=list)
    clones: list[str] = field(default_factory=list)
    clone_assignments: dict[str, str] = field(default_factory=dict)
    memory_dir: str = ""
    memory_encrypted: bool = False
    memory_users: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and all(r.ok for r in self.adapter_results)


def _validate_personas(
    personas: dict[str, Persona], backend: Backend, voices: VoiceRegistry
) -> list[str]:
    """Cross-check personas' voices against the active backend (warnings, not errors)."""
    warnings: list[str] = []
    tts = backend.tts.name
    cloning = getattr(backend.tts, "supports_cloning", False)
    for persona in personas.values():
        ref = persona.voice.ref
        # An assigned clone on a cloning backend (M5) → the persona speaks in that voice.
        if cloning and voices.clone_for_persona(persona.id):
            continue
        # A concrete preset for this backend → the persona will sound distinct.
        if voices.has_preset(ref, tts):
            continue
        # No preset and no usable clone: it falls back to the default voice, so distinct
        # personas won't sound distinct on this backend.
        if not cloning:
            warnings.append(
                f"persona {persona.id!r} voice {ref!r} has no {tts!r} preset and {tts!r} "
                f"can't clone — it will use the default voice (won't sound distinct)"
            )
    return warnings


def run_check(settings: Settings) -> CheckReport:
    report = CheckReport()

    # 1. Backend config + adapters.
    try:
        backend_config = load_backend_config(settings)
    except ConfigError as exc:
        report.errors.append(str(exc))
        return report

    report.backend_name = backend_config.backend
    try:
        backend = build_backend(backend_config)
    except ValueError as exc:
        report.errors.append(str(exc))
        return report

    for adapter in (backend.stt, backend.llm, backend.tts):
        try:
            result = adapter.check()
        except Exception as exc:  # surface any adapter check failure as a non-ok result
            result = CheckResult(
                stage=adapter.stage, adapter=adapter.name, ok=False, detail=str(exc)
            )
        report.adapter_results.append(result)
        report.warnings.extend(f"[{result.stage}:{result.adapter}] {w}" for w in result.warnings)

    # 2. Personas.
    try:
        personas = load_all_personas(settings)
    except ValueError as exc:
        report.errors.append(str(exc))
        return report

    report.personas = sorted(personas)
    if not personas:
        report.warnings.append(f"no personas found in {settings.personas_dir}")

    # 3. Voice registry (distinct per-persona voices).
    try:
        voices = load_voice_registry(settings)
    except ValueError as exc:
        report.errors.append(str(exc))
        return report

    report.voices = voices.ids()
    if voices.clones is not None:
        report.clones = voices.clones.names()
        report.clone_assignments = voices.clones.assignments
    report.warnings.extend(_validate_personas(personas, backend, voices))

    # 4. Memory (M8). Surfaces the store location, at-rest encryption, and #users so a
    # misconfigured PERSONAVOICE_MEMORY_KEY (cryptography missing / bad key) fails the check.
    report.memory_dir = str(settings.memory_dir)
    memory_enabled = any(p.memory.enabled for p in personas.values())
    try:
        store = load_memory_store(settings)
        report.memory_encrypted = store.encrypted
        report.memory_users = len(store.users())
    except MemoryStoreError as exc:
        report.errors.append(str(exc))
        return report
    if memory_enabled and not report.memory_encrypted:
        report.warnings.append(
            "a persona has memory enabled but PERSONAVOICE_MEMORY_KEY is unset — stored "
            "conversations are unencrypted at rest (fine for dev; set a key for real users)"
        )

    return report

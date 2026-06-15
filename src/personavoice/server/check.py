"""`server --check`: validate config and load (stub) adapters on this machine.

Produces a structured `CheckReport` so the same logic is exercised by tests and by the
CLI. This is the M0 acceptance test: it must pass on both the Mac and the 4080.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..adapters.factory import Backend, build_backend
from ..models import CheckResult, Persona
from .config import (
    ConfigError,
    Settings,
    load_all_personas,
    load_backend_config,
)


@dataclass
class CheckReport:
    backend_name: str = ""
    adapter_results: list[CheckResult] = field(default_factory=list)
    personas: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and all(r.ok for r in self.adapter_results)


def _validate_personas(personas: dict[str, Persona], backend: Backend) -> list[str]:
    """Cross-check personas against the active backend (warnings, not hard errors)."""
    warnings: list[str] = []
    cloning = getattr(backend.tts, "supports_cloning", False)
    for persona in personas.values():
        ref = persona.voice.ref
        # A clone-style ref (sample under voices/) needs a cloning-capable TTS backend.
        if ref.startswith("voices/") and not cloning:
            warnings.append(
                f"persona {persona.id!r} uses cloned voice {ref!r} but backend TTS "
                f"{backend.tts.name!r} has no cloning support (ok until M5)"
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
    report.warnings.extend(_validate_personas(personas, backend))

    return report

"""The M0 acceptance test: `server --check` passes on both backends with stub adapters."""

from __future__ import annotations

from personavoice.server.check import run_check
from personavoice.server.config import Settings


def test_check_passes_on_both_backends(settings: Settings) -> None:
    report = run_check(settings)
    assert report.ok, f"check failed: errors={report.errors}"
    assert report.backend_name == settings.backend
    # All three stages constructed and self-checked.
    assert {r.stage for r in report.adapter_results} == {"stt", "llm", "tts"}
    assert all(r.ok for r in report.adapter_results)
    # The four shipped personas are discovered.
    assert report.personas == [
        "companion",
        "hr_interviewer",
        "language_teacher",
        "pm_interviewer",
    ]


def test_check_reports_unknown_adapter(tmp_path) -> None:
    backends = tmp_path / "backends"
    backends.mkdir()
    (backends / "mac.yaml").write_text(
        "backend: mac\n"
        "stt: {adapter: nope, model: x}\n"
        "llm: {adapter: ollama, model: y}\n"
        "tts: {adapter: kokoro, model: z}\n"
    )
    (tmp_path / "personas").mkdir()
    settings = Settings(backend="mac", config_dir=tmp_path, models_dir=tmp_path / "models")

    report = run_check(settings)
    assert not report.ok
    assert any("unknown stt adapter" in e for e in report.errors)


def test_check_warns_on_backend_mismatch(tmp_path) -> None:
    backends = tmp_path / "backends"
    backends.mkdir()
    # File is mac.yaml but declares cuda.
    (backends / "mac.yaml").write_text(
        "backend: cuda\n"
        "stt: {adapter: whisper_mlx}\n"
        "llm: {adapter: ollama}\n"
        "tts: {adapter: kokoro}\n"
    )
    (tmp_path / "personas").mkdir()
    settings = Settings(backend="mac", config_dir=tmp_path, models_dir=tmp_path / "models")

    report = run_check(settings)
    assert not report.ok
    assert any("declares backend" in e for e in report.errors)

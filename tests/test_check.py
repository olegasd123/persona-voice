"""The acceptance check: `server --check` passes on both backends with stub adapters."""

from __future__ import annotations

import pytest

from personavoice.obs import prometheus as prom
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


def test_check_warns_when_clones_present_but_tts_cannot_speak_them(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from personavoice.voice.clone import ClonedVoice, ClonesStore

    backends = tmp_path / "backends"
    backends.mkdir()
    (backends / "mac.yaml").write_text(
        "backend: mac\n"
        "stt: {adapter: whisper_mlx}\n"
        "llm: {adapter: ollama}\n"
        "tts: {adapter: kokoro}\n"  # preset-only: can't speak clones
    )
    (tmp_path / "personas").mkdir()
    clones_dir = tmp_path / "clones"
    store = ClonesStore(clones_dir)
    store.record(ClonedVoice(name="my_voice", sample_path="s.wav"))
    settings = Settings(
        backend="mac",
        config_dir=tmp_path,
        models_dir=tmp_path / "models",
        clones_dir=clones_dir,
    )

    report = run_check(settings)
    assert any("can't speak them" in w for w in report.warnings)


def test_check_lists_tools_and_companion_tool_routes(settings: Settings) -> None:
    # The default config: the registry is reported and the companion's get_current_time is known
    # and routable on the OpenAI-compatible backends — so no tool warnings.
    report = run_check(settings)
    assert report.tools == ["get_current_time"]
    assert not any("tool" in w.lower() for w in report.warnings)


def test_check_warns_on_unknown_persona_tool(tmp_path) -> None:  # type: ignore[no-untyped-def]
    backends = tmp_path / "backends"
    backends.mkdir()
    (backends / "mac.yaml").write_text(
        "backend: mac\n"
        "stt: {adapter: whisper_mlx}\n"
        "llm: {adapter: lmstudio}\n"  # supports tools, so only the unknown-tool warning fires
        "tts: {adapter: kokoro}\n"
    )
    personas = tmp_path / "personas"
    personas.mkdir()
    (personas / "toolpersona.yaml").write_text(
        "id: toolpersona\n"
        "name: Tool Persona\n"
        "system_prompt: hi\n"
        "llm: {base_model: m}\n"
        "voice: {ref: voices/x}\n"
        "tools: [ghost]\n"
    )
    settings = Settings(backend="mac", config_dir=tmp_path, models_dir=tmp_path / "models")

    report = run_check(settings)
    assert any("unknown tool" in w and "ghost" in w for w in report.warnings)


def test_check_warns_when_backend_cannot_route_tools(tmp_path) -> None:  # type: ignore[no-untyped-def]
    backends = tmp_path / "backends"
    backends.mkdir()
    (backends / "mac.yaml").write_text(
        "backend: mac\n"
        "stt: {adapter: whisper_mlx}\n"
        "llm: {adapter: ollama}\n"  # can't route function calls
        "tts: {adapter: kokoro}\n"
    )
    personas = tmp_path / "personas"
    personas.mkdir()
    (personas / "toolpersona.yaml").write_text(
        "id: toolpersona\n"
        "name: Tool Persona\n"
        "system_prompt: hi\n"
        "llm: {base_model: m}\n"
        "voice: {ref: voices/x}\n"
        "tools: [get_current_time]\n"  # a real tool, but the backend can't route it
    )
    settings = Settings(backend="mac", config_dir=tmp_path, models_dir=tmp_path / "models")

    report = run_check(settings)
    assert any("can't route function calls" in w for w in report.warnings)


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


def test_check_reports_metrics_status(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    # The metrics exporter availability is surfaced and no PROMETHEUS_MULTIPROC_DIR warning fires
    # when the var is unset (the common case).
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    report = run_check(settings)
    assert report.metrics_enabled == prom.AVAILABLE
    assert report.metrics_multiproc_dir == ""
    assert not any("PROMETHEUS_MULTIPROC_DIR" in w for w in report.warnings)


def test_check_warns_on_missing_multiproc_dir(
    settings: Settings, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not prom.AVAILABLE:
        pytest.skip("prometheus-client not installed")
    missing = tmp_path / "does-not-exist"
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(missing))
    report = run_check(settings)
    assert report.metrics_multiproc_dir == str(missing)
    assert any(
        "PROMETHEUS_MULTIPROC_DIR" in w and "existing directory" in w for w in report.warnings
    )


def test_check_reports_admission_control(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The default cap (1) is reported without a warning; raising it flags the OOM risk.
    monkeypatch.delenv("PERSONAVOICE_MAX_SESSIONS", raising=False)
    report = run_check(settings)
    assert report.max_sessions == 1
    assert not any("PERSONAVOICE_MAX_SESSIONS" in w for w in report.warnings)

    monkeypatch.setenv("PERSONAVOICE_MAX_SESSIONS", "4")
    report = run_check(settings)
    assert report.max_sessions == 4
    assert any("PERSONAVOICE_MAX_SESSIONS" in w for w in report.warnings)

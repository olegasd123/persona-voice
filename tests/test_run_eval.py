"""run_eval: pure gate-parsing / artifact-extraction helpers (no backend)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from personavoice.eval.dashboard import CEILING, FLOOR

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_eval.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_eval", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_eval = _load()


def test_parse_gate() -> None:
    gate = run_eval.parse_gate("wer:ceiling:0.1")
    assert gate.metric == "wer"
    assert gate.bound == CEILING
    assert gate.threshold == pytest.approx(0.1)


def test_parse_gate_rejects_bad_spec() -> None:
    with pytest.raises(ValueError):
        run_eval.parse_gate("wer:0.1")
    with pytest.raises(ValueError):
        run_eval.parse_gate("wer:sideways:0.1")
    with pytest.raises(ValueError):
        run_eval.parse_gate("wer:ceiling:nope")


def test_default_gates_cover_headline_metrics() -> None:
    metrics = {g.metric for g in run_eval.default_gates()}
    assert {"e2e_audio_ms", "wer", "persona_adherence", "mos"} <= metrics


def test_latency_ms_from_bench() -> None:
    payload = {"stats": {"e2e_audio": {"median": 0.58}}}
    assert run_eval.latency_ms_from_bench(payload) == pytest.approx(580.0)
    assert run_eval.latency_ms_from_bench({"stats": {}}) is None


def test_merge_gates_overrides_by_metric() -> None:
    defaults = run_eval.default_gates()
    merged = run_eval.merge_gates(defaults, [run_eval.parse_gate("wer:ceiling:0.05")])
    wer_gate = next(g for g in merged if g.metric == "wer")
    assert wer_gate.threshold == pytest.approx(0.05)
    # New metric appends rather than replacing.
    merged2 = run_eval.merge_gates(defaults, [run_eval.parse_gate("xtra:floor:1.0")])
    assert any(g.metric == "xtra" for g in merged2)


def test_merge_gates_new_metric_when_no_defaults() -> None:
    merged = run_eval.merge_gates([], [run_eval.parse_gate("only:floor:0.5")])
    assert len(merged) == 1
    assert merged[0].metric == "only"
    assert merged[0].bound == FLOOR

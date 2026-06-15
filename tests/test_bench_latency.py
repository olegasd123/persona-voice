"""bench_latency: pure aggregation/formatting helpers (no backend, no models)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench_latency.py"


def _load_bench() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bench_latency", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses can resolve the module by `__module__`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_bench()


def test_summarize_basic_stats() -> None:
    s = bench.summarize([1.0, 2.0, 3.0])
    assert s.n == 3
    assert s.min == 1.0
    assert s.max == 3.0
    assert s.median == 2.0
    assert s.mean == pytest.approx(2.0)


def test_summarize_runs_per_stage() -> None:
    timings = [
        {"stt": 1.0, "llm": 0.5, "tts": 2.0, "total": 3.5},
        {"stt": 1.2, "llm": 0.7, "tts": 2.2, "total": 4.1},
    ]
    stats = bench.summarize_runs(timings)
    assert set(stats) == {"stt", "llm", "tts", "total"}
    assert stats["stt"].min == 1.0
    assert stats["stt"].max == 1.2
    assert stats["total"].mean == pytest.approx(3.8)


def test_summarize_runs_empty() -> None:
    assert bench.summarize_runs([]) == {}


def test_summarize_runs_skips_missing_stage() -> None:
    # A run missing a stage key (e.g. an error path) shouldn't crash aggregation.
    stats = bench.summarize_runs([{"stt": 1.0}, {"stt": 2.0, "total": 5.0}])
    assert stats["stt"].n == 2
    assert stats["total"].n == 1
    assert "llm" not in stats


def test_format_report_contains_stages() -> None:
    stats = bench.summarize_runs([{"stt": 1.0, "llm": 0.5, "tts": 2.0, "total": 3.5}])
    report = bench.format_report(stats, backend="mac", runs=1)
    assert "backend=mac" in report
    for stage in ("stt", "llm", "tts", "total"):
        assert stage in report

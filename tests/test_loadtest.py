"""loadtest: pure aggregation helpers (no network)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "loadtest.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("loadtest", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


loadtest = _load()


def test_summarize_load_counts_and_rate() -> None:
    stats = loadtest.summarize_load([10.0, 20.0, 30.0, 40.0], errors=1, wall_s=2.0)
    assert stats.requests == 5  # 4 successes + 1 error
    assert stats.errors == 1
    assert stats.error_rate == 0.2
    assert stats.rps == 2.5  # 5 requests / 2s
    assert stats.max_ms == 40.0


def test_summarize_load_percentiles_monotonic() -> None:
    stats = loadtest.summarize_load([float(i) for i in range(1, 101)], errors=0, wall_s=1.0)
    assert stats.p50_ms <= stats.p95_ms <= stats.p99_ms <= stats.max_ms


def test_summarize_load_all_errors() -> None:
    stats = loadtest.summarize_load([], errors=3, wall_s=1.0)
    assert stats.requests == 3
    assert stats.error_rate == 1.0
    assert stats.p50_ms == 0.0
    assert stats.max_ms == 0.0


def test_format_load_contains_headline() -> None:
    stats = loadtest.summarize_load([5.0, 6.0], errors=0, wall_s=1.0)
    out = loadtest.format_load(stats, url="http://x", concurrency=4)
    assert "Load test" in out
    assert "throughput" in out
    assert "concurrency=4" in out

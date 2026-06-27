"""Automated quality eval: STT word-error-rate, voice MOS spot-checks, and a
dashboard that gates the headline metrics against thresholds.

The scoring primitives here are pure (text/number in, number out) so they're unit-tested
offline; the runnable harnesses that exercise a real backend live in `scripts/eval_stt.py`
and `scripts/run_eval.py`. Persona-adherence scoring already lives in
`personavoice.training.eval`; the dashboard composes all of them.
"""

from __future__ import annotations

from .dashboard import (
    Gate,
    MetricResult,
    evaluate_gates,
    format_dashboard,
)
from .mos import MosSummary, summarize_mos
from .wer import normalize_text, wer, word_errors

__all__ = [
    "Gate",
    "MetricResult",
    "MosSummary",
    "evaluate_gates",
    "format_dashboard",
    "normalize_text",
    "summarize_mos",
    "wer",
    "word_errors",
]

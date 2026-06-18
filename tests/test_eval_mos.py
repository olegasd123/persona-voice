"""MOS spot-check aggregation (M10)."""

from __future__ import annotations

import pytest

from personavoice.eval.mos import summarize_mos, summarize_mos_by_voice


def test_mean_of_ratings() -> None:
    s = summarize_mos("companion", [4.0, 5.0, 3.0])
    assert s.n == 3
    assert s.mean == pytest.approx(4.0)
    assert s.stdev == pytest.approx(1.0)
    assert s.ci95 > 0.0


def test_single_rating_has_no_spread() -> None:
    s = summarize_mos("pm", [4.0])
    assert s.n == 1
    assert s.mean == 4.0
    assert s.stdev == 0.0
    assert s.ci95 == 0.0
    assert s.low == 4.0 and s.high == 4.0


def test_empty_ratings() -> None:
    s = summarize_mos("hr", [])
    assert s.n == 0
    assert s.mean == 0.0


def test_out_of_range_rejected() -> None:
    with pytest.raises(ValueError):
        summarize_mos("teacher", [4.0, 6.0])
    with pytest.raises(ValueError):
        summarize_mos("teacher", [0.0])


def test_by_voice_sorted() -> None:
    out = summarize_mos_by_voice({"zoe": [5.0], "amy": [4.0]})
    assert [s.voice for s in out] == ["amy", "zoe"]

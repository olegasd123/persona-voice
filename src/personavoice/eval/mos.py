"""MOS (Mean Opinion Score) spot-check aggregation (automated eval).

There's no reliable, license-clean objective MOS model we want to depend on, so voice quality
is spot-checked the standard way: synthesize a handful of probe lines per voice, have a human
rate each 1-5, and aggregate. `scripts/run_eval.py` generates the probe wavs and a ratings
sheet (one row per clip); this module is the pure aggregation — read the filled-in ratings
back and report mean / spread / a rough 95% confidence interval per voice.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

MOS_MIN = 1.0
MOS_MAX = 5.0


@dataclass(frozen=True)
class MosSummary:
    """Aggregate MOS for one voice's rated clips."""

    voice: str
    n: int
    mean: float
    stdev: float
    ci95: float  # half-width of the 95% confidence interval around the mean

    @property
    def low(self) -> float:
        return self.mean - self.ci95

    @property
    def high(self) -> float:
        return self.mean + self.ci95


def _validate(ratings: Sequence[float]) -> list[float]:
    valid: list[float] = []
    for r in ratings:
        if not (MOS_MIN <= r <= MOS_MAX):
            raise ValueError(f"MOS rating {r} out of range [{MOS_MIN}, {MOS_MAX}]")
        valid.append(float(r))
    return valid


def summarize_mos(voice: str, ratings: Sequence[float]) -> MosSummary:
    """Mean/stdev/95%-CI for one voice's ratings (CI half-width = 1.96 * SE)."""
    valid = _validate(ratings)
    n = len(valid)
    if n == 0:
        return MosSummary(voice=voice, n=0, mean=0.0, stdev=0.0, ci95=0.0)
    mean = sum(valid) / n
    if n == 1:
        return MosSummary(voice=voice, n=1, mean=mean, stdev=0.0, ci95=0.0)
    variance = sum((r - mean) ** 2 for r in valid) / (n - 1)
    stdev = math.sqrt(variance)
    ci95 = 1.96 * stdev / math.sqrt(n)
    return MosSummary(voice=voice, n=n, mean=mean, stdev=stdev, ci95=ci95)


def summarize_mos_by_voice(ratings_by_voice: Mapping[str, Sequence[float]]) -> list[MosSummary]:
    """Summarize each voice's ratings, sorted by voice id for stable reporting."""
    return [summarize_mos(v, ratings_by_voice[v]) for v in sorted(ratings_by_voice)]

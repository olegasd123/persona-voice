"""Token-bucket rate limiting for the token server (M10 security pass).

The token endpoint mints LiveKit credentials, so it's the one place worth protecting from a
runaway/abusive client. A classic per-key token bucket: each key (client IP by default)
refills `rate` tokens/second up to `burst`; a request is allowed if a token is available.
Disabled (allow-all) when `rate <= 0`, which is the dev default.

Pure and time-injectable (`now=`), so the throttle behavior is unit-tested without sleeping.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field


@dataclass
class RateLimiter:
    """Per-key token bucket. `allow(key)` consumes a token and returns whether it was free."""

    rate: float  # tokens refilled per second (<= 0 disables limiting)
    burst: float  # bucket capacity (max tokens)
    _buckets: dict[str, tuple[float, float]] = field(default_factory=dict)  # key -> (tokens, ts)

    @property
    def enabled(self) -> bool:
        return self.rate > 0 and self.burst > 0

    def allow(self, key: str, *, now: float | None = None) -> bool:
        if not self.enabled:
            return True
        ts = time.monotonic() if now is None else now
        tokens, last = self._buckets.get(key, (self.burst, ts))
        # Refill since the last request, capped at the bucket size.
        tokens = min(self.burst, tokens + (ts - last) * self.rate)
        if tokens >= 1.0:
            self._buckets[key] = (tokens - 1.0, ts)
            return True
        self._buckets[key] = (tokens, ts)
        return False


def rate_limiter_from_env(env: dict[str, str] | None = None) -> RateLimiter:
    """Build a `RateLimiter` from `PERSONAVOICE_RATE_LIMIT_RPS` / `..._BURST` (disabled by default)."""
    env = os.environ if env is None else env

    def _f(name: str, default: float) -> float:
        raw = (env.get(name) or "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    rate = _f("PERSONAVOICE_RATE_LIMIT_RPS", 0.0)
    # Default the burst to one second of rate (min 1) so a fresh client gets at least one token.
    burst = _f("PERSONAVOICE_RATE_LIMIT_BURST", max(1.0, rate))
    return RateLimiter(rate=rate, burst=burst)

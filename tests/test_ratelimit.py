"""Token-bucket rate limiter (M10 security pass)."""

from __future__ import annotations

from personavoice.server.ratelimit import RateLimiter, rate_limiter_from_env


def test_disabled_limiter_allows_everything() -> None:
    limiter = RateLimiter(rate=0.0, burst=0.0)
    assert limiter.enabled is False
    assert all(limiter.allow("ip", now=float(i)) for i in range(100))


def test_burst_then_throttle() -> None:
    # burst=2 -> two immediate requests pass, the third (same instant) is denied.
    limiter = RateLimiter(rate=1.0, burst=2.0)
    assert limiter.allow("ip", now=0.0) is True
    assert limiter.allow("ip", now=0.0) is True
    assert limiter.allow("ip", now=0.0) is False


def test_refill_over_time() -> None:
    limiter = RateLimiter(rate=1.0, burst=1.0)
    assert limiter.allow("ip", now=0.0) is True
    assert limiter.allow("ip", now=0.5) is False  # only 0.5 token refilled
    assert limiter.allow("ip", now=1.0) is True  # a full token back


def test_per_key_isolation() -> None:
    limiter = RateLimiter(rate=1.0, burst=1.0)
    assert limiter.allow("a", now=0.0) is True
    assert limiter.allow("b", now=0.0) is True  # different key, own bucket
    assert limiter.allow("a", now=0.0) is False


def test_from_env_default_disabled() -> None:
    assert rate_limiter_from_env({}).enabled is False


def test_from_env_parses_rps_and_default_burst() -> None:
    limiter = rate_limiter_from_env({"PERSONAVOICE_RATE_LIMIT_RPS": "5"})
    assert limiter.rate == 5.0
    assert limiter.burst == 5.0  # defaults to one second of rate
    custom = rate_limiter_from_env(
        {"PERSONAVOICE_RATE_LIMIT_RPS": "5", "PERSONAVOICE_RATE_LIMIT_BURST": "20"}
    )
    assert custom.burst == 20.0


def test_from_env_ignores_invalid() -> None:
    assert rate_limiter_from_env({"PERSONAVOICE_RATE_LIMIT_RPS": "fast"}).enabled is False

"""Token-bucket rate limiter (security pass)."""

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


def test_idle_buckets_are_evicted() -> None:
    # Many one-shot keys leave a bucket each; once they've refilled, a sweep forgets them.
    limiter = RateLimiter(rate=1.0, burst=1.0, sweep_interval=10.0)
    for i in range(50):
        assert limiter.allow(f"ip-{i}", now=0.0) is True
    assert len(limiter._buckets) == 50
    # Advance past the sweep interval: every bucket has fully refilled, so all are dropped and
    # only the triggering call's own (still-draining) bucket remains.
    assert limiter.allow("trigger", now=20.0) is True
    assert set(limiter._buckets) == {"trigger"}


def test_active_bucket_survives_sweep() -> None:
    # Slow refill relative to the sweep interval, so a recently-busy key isn't full at sweep time.
    limiter = RateLimiter(rate=1.0, burst=100.0, sweep_interval=10.0)
    for _ in range(100):
        limiter.allow("busy", now=0.0)  # drain to 0 tokens
    assert limiter.allow("busy", now=0.0) is False  # throttled
    # A sweep fires while another key is checked at t=10; "busy" has refilled only ~10/100 tokens,
    # so it's still throttled and must be kept — evicting it would reset the abuser to a full bucket.
    limiter.allow("other", now=10.0)
    assert "busy" in limiter._buckets
    assert limiter.allow("busy", now=10.0) is True  # its preserved ~10 tokens still let one through


def test_memory_bounded_under_key_churn() -> None:
    # Regression: per-key state must not grow with the *total* number of keys ever seen, only
    # with those active within one sweep interval. Without eviction this map would reach 2000.
    limiter = RateLimiter(rate=1.0, burst=1.0, sweep_interval=10.0)
    peak = 0
    for window in range(100):
        base = window * 100.0  # each window is well past the previous one's refill
        for k in range(20):
            limiter.allow(f"ip-{window}-{k}", now=base)
        peak = max(peak, len(limiter._buckets))
    assert peak <= 40  # ~one window's worth, not the 2000 distinct keys seen over time
    assert len(limiter._buckets) <= 40


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

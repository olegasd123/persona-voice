#!/usr/bin/env python3
"""Load-test the token server (observability / robustness).

    python scripts/loadtest.py --url http://localhost:8080 --requests 500 --concurrency 20
    python scripts/loadtest.py --url http://localhost:8080 --token $PERSONAVOICE_API_TOKEN

Fires `--requests` token mints across `--concurrency` workers and reports throughput, latency
percentiles, and the error rate — the "load test" and a quick way to
confirm the server stays healthy (and that rate limiting kicks in) under concurrency. Uses
only the stdlib (`urllib`, threads). The aggregation (`summarize_load`) is pure + unit-tested.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any


@dataclass
class LoadStats:
    requests: int
    errors: int
    wall_s: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    rps: float

    @property
    def error_rate(self) -> float:
        return self.errors / self.requests if self.requests else 0.0


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Nearest-rank percentile (pct in [0, 100]) of an already-sorted list."""
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, round(pct / 100.0 * len(sorted_vals) + 0.5) - 1))
    return sorted_vals[k]


def summarize_load(latencies_ms: list[float], errors: int, wall_s: float) -> LoadStats:
    """Aggregate per-request latencies (successes) + error count into headline stats."""
    n = len(latencies_ms) + errors
    ordered = sorted(latencies_ms)
    return LoadStats(
        requests=n,
        errors=errors,
        wall_s=wall_s,
        p50_ms=_percentile(ordered, 50),
        p95_ms=_percentile(ordered, 95),
        p99_ms=_percentile(ordered, 99),
        max_ms=ordered[-1] if ordered else 0.0,
        rps=(n / wall_s) if wall_s > 0 else 0.0,
    )


def format_load(stats: LoadStats, *, url: str, concurrency: int) -> str:
    return "\n".join(
        [
            f"Load test — {url}  requests={stats.requests}  concurrency={concurrency}",
            f"  throughput : {stats.rps:.1f} req/s over {stats.wall_s:.2f} s",
            f"  errors     : {stats.errors}  ({stats.error_rate * 100:.1f}%)",
            f"  latency ms : p50={stats.p50_ms:.0f}  p95={stats.p95_ms:.0f}  "
            f"p99={stats.p99_ms:.0f}  max={stats.max_ms:.0f}",
        ]
    )


def _one_request(url: str, token: str | None, persona: str | None) -> float:
    """Mint one token; return latency in ms or raise on a non-2xx / transport error."""
    body = json.dumps({"persona": persona} if persona else {}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{url.rstrip('/')}/token", data=body, headers=headers)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()
    return (time.perf_counter() - t0) * 1000.0


def run_load(
    url: str, *, total: int, concurrency: int, token: str | None, persona: str | None
) -> LoadStats:
    latencies: list[float] = []
    errors = 0
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_one_request, url, token, persona) for _ in range(total)]
        for fut in futures:
            try:
                latencies.append(fut.result())
            except Exception:
                errors += 1
    wall = time.perf_counter() - start
    return summarize_load(latencies, errors, wall)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Concurrency load test for the token server.")
    parser.add_argument("--url", default="http://localhost:8080", help="token server base URL")
    parser.add_argument("--requests", type=int, default=200, help="total requests to send")
    parser.add_argument("--concurrency", type=int, default=10, help="concurrent workers")
    parser.add_argument("--token", default=None, help="bearer token if the server requires auth")
    parser.add_argument("--persona", default=None, help="persona id to request")
    parser.add_argument("--json", default=None, help="also write the report as JSON")
    args = parser.parse_args(argv)

    stats = run_load(
        args.url,
        total=args.requests,
        concurrency=args.concurrency,
        token=args.token,
        persona=args.persona,
    )
    print(format_load(stats, url=args.url, concurrency=args.concurrency))

    if args.json is not None:
        from pathlib import Path

        payload: dict[str, Any] = {
            "url": args.url,
            "concurrency": args.concurrency,
            "requests": stats.requests,
            "errors": stats.errors,
            "error_rate": stats.error_rate,
            "wall_s": stats.wall_s,
            "rps": stats.rps,
            "p50_ms": stats.p50_ms,
            "p95_ms": stats.p95_ms,
            "p99_ms": stats.p99_ms,
            "max_ms": stats.max_ms,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote JSON report -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

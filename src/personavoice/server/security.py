"""Security posture audit for the token server (M10 security pass).

A small, pure checklist over the server's effective configuration — auth, transport (TLS),
rate limiting, secret strength — that turns "is this safe to expose?" into concrete findings.
The token server prints these at startup; in strict mode (`PERSONAVOICE_REQUIRE_AUTH=1`, for
prod) the `error`-level findings make startup refuse rather than silently serve wide open.

Pure (plain args in, findings out) so the policy is unit-tested without sockets.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

ERROR = "error"
WARNING = "warning"
OK = "ok"

# A bearer token shorter than this is almost certainly not a real secret.
_MIN_TOKEN_LEN = 16
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", ""}


@dataclass(frozen=True)
class SecurityFinding:
    level: str  # ERROR | WARNING | OK
    message: str


def _is_local(host: str) -> bool:
    return host in _LOCAL_HOSTS


def audit_security(
    *,
    api_token: str | None,
    livekit_url: str,
    tls_enabled: bool,
    rate_limited: bool,
    require_auth: bool = False,
    bind_host: str = "0.0.0.0",
) -> list[SecurityFinding]:
    """Evaluate the server's effective config and return findings (worst-first by level)."""
    findings: list[SecurityFinding] = []
    exposed = not _is_local(bind_host)  # bound to a non-loopback address → reachable off-box

    # Auth on the token endpoint.
    if not api_token:
        msg = "no API token set — /token and /personas are open (set PERSONAVOICE_API_TOKEN)"
        findings.append(SecurityFinding(ERROR if require_auth else WARNING, msg))
    elif len(api_token) < _MIN_TOKEN_LEN:
        findings.append(
            SecurityFinding(
                WARNING, f"API token is short (<{_MIN_TOKEN_LEN} chars) — use a strong random token"
            )
        )
    else:
        findings.append(SecurityFinding(OK, "API token auth is enabled"))

    # Transport security for the HTTP server.
    if not tls_enabled:
        level = WARNING if (exposed or require_auth) else OK
        findings.append(
            SecurityFinding(
                level,
                "token server is HTTP (no TLS) — terminate TLS at a reverse proxy or set "
                "PERSONAVOICE_TLS_CERT/PERSONAVOICE_TLS_KEY before exposing it",
            )
        )

    # LiveKit signaling transport.
    parsed = urlparse(livekit_url) if livekit_url else None
    if parsed and parsed.scheme == "ws" and not _is_local(parsed.hostname or ""):
        findings.append(
            SecurityFinding(WARNING, f"LiveKit URL uses ws:// (not wss://): {livekit_url}")
        )

    # Rate limiting on the mint endpoint.
    if not rate_limited:
        level = WARNING if (exposed or require_auth) else OK
        findings.append(
            SecurityFinding(level, "no rate limiting — set PERSONAVOICE_RATE_LIMIT_RPS to throttle")
        )

    return findings


def has_errors(findings: list[SecurityFinding]) -> bool:
    return any(f.level == ERROR for f in findings)

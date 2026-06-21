"""Token-server security audit (security pass)."""

from __future__ import annotations

from personavoice.server.security import (
    ERROR,
    OK,
    WARNING,
    audit_security,
    has_errors,
)

STRONG_TOKEN = "a" * 32


def _levels(findings) -> set[str]:
    return {f.level for f in findings}


def test_dev_defaults_warn_but_do_not_error() -> None:
    # No token, HTTP, no rate limit, loopback bind: warnings only, never blocking.
    findings = audit_security(
        api_token=None,
        livekit_url="ws://localhost:7880",
        tls_enabled=False,
        rate_limited=False,
        require_auth=False,
        bind_host="127.0.0.1",
    )
    assert has_errors(findings) is False
    assert WARNING in _levels(findings)


def test_strict_mode_without_token_is_an_error() -> None:
    findings = audit_security(
        api_token=None,
        livekit_url="wss://lk:7880",
        tls_enabled=True,
        rate_limited=True,
        require_auth=True,
    )
    assert has_errors(findings) is True


def test_short_token_warns() -> None:
    findings = audit_security(
        api_token="short",
        livekit_url="wss://lk:7880",
        tls_enabled=True,
        rate_limited=True,
    )
    assert any("short" in f.message for f in findings)
    assert has_errors(findings) is False


def test_well_configured_prod_is_clean() -> None:
    findings = audit_security(
        api_token=STRONG_TOKEN,
        livekit_url="wss://lk.example:7880",
        tls_enabled=True,
        rate_limited=True,
        require_auth=True,
        bind_host="0.0.0.0",
    )
    assert has_errors(findings) is False
    assert all(f.level == OK for f in findings)


def test_plaintext_livekit_url_warns_when_remote() -> None:
    findings = audit_security(
        api_token=STRONG_TOKEN,
        livekit_url="ws://lk.example:7880",
        tls_enabled=True,
        rate_limited=True,
    )
    assert any("ws://" in f.message for f in findings)


def test_local_ws_livekit_does_not_warn() -> None:
    findings = audit_security(
        api_token=STRONG_TOKEN,
        livekit_url="ws://localhost:7880",
        tls_enabled=True,
        rate_limited=True,
    )
    assert not any("ws://" in f.message for f in findings)


def test_exposed_http_without_tls_warns() -> None:
    findings = audit_security(
        api_token=STRONG_TOKEN,
        livekit_url="wss://lk:7880",
        tls_enabled=False,
        rate_limited=True,
        bind_host="0.0.0.0",
    )
    assert any("TLS" in f.message for f in findings)
    assert ERROR not in _levels(findings)

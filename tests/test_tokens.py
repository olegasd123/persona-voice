"""LiveKit JWT minting/verification (`personavoice.server.tokens`)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json

import pytest

from personavoice.server.tokens import (
    AccessToken,
    TokenInvalid,
    VideoGrant,
    decode_token,
    mint_access_token,
)

KEY = "APIabc123"
SECRET = "s3cret-shhh"


def _fixed_clock(t: float):
    return lambda: t


def test_minted_token_has_three_segments_and_verifies() -> None:
    token = mint_access_token(
        api_key=KEY, api_secret=SECRET, identity="user-1", room="room-1", now=_fixed_clock(1000)
    )
    assert token.count(".") == 2  # header.payload.signature
    claims = decode_token(token, SECRET, now=_fixed_clock(1000))
    assert claims["iss"] == KEY
    assert claims["sub"] == "user-1"
    assert claims["nbf"] == 1000
    assert claims["exp"] == 1000 + 3600


def test_video_grant_claim_shape() -> None:
    token = mint_access_token(
        api_key=KEY, api_secret=SECRET, identity="u", room="myroom", now=_fixed_clock(0)
    )
    grant = decode_token(token, SECRET, now=_fixed_clock(0))["video"]
    assert grant == {
        "room": "myroom",
        "roomJoin": True,
        "canPublish": True,
        "canSubscribe": True,
        "canPublishData": True,
    }


def test_header_is_hs256_jwt() -> None:
    token = mint_access_token(api_key=KEY, api_secret=SECRET, identity="u", room="r")
    header_seg = token.split(".")[0]
    pad = "=" * (-len(header_seg) % 4)
    header = json.loads(base64.urlsafe_b64decode(header_seg + pad))
    assert header == {"alg": "HS256", "typ": "JWT"}


def test_name_and_metadata_round_trip() -> None:
    meta = json.dumps({"persona": "hr_interviewer"})
    token = mint_access_token(
        api_key=KEY,
        api_secret=SECRET,
        identity="u",
        room="r",
        name="Alice",
        metadata=meta,
        now=_fixed_clock(0),
    )
    claims = decode_token(token, SECRET, now=_fixed_clock(0))
    assert claims["name"] == "Alice"
    assert json.loads(claims["metadata"]) == {"persona": "hr_interviewer"}


def test_optional_claims_omitted_when_absent() -> None:
    token = mint_access_token(api_key=KEY, api_secret=SECRET, identity="u", room="r")
    claims = decode_token(token, SECRET)
    assert "name" not in claims
    assert "metadata" not in claims


def test_signature_must_match_secret() -> None:
    token = mint_access_token(api_key=KEY, api_secret=SECRET, identity="u", room="r")
    with pytest.raises(TokenInvalid, match="signature"):
        decode_token(token, "wrong-secret")


def test_tampered_payload_is_rejected() -> None:
    token = mint_access_token(
        api_key=KEY, api_secret=SECRET, identity="u", room="r", now=_fixed_clock(0)
    )
    header_seg, _payload_seg, sig_seg = token.split(".")
    forged_payload = (
        base64.urlsafe_b64encode(
            json.dumps({"iss": KEY, "sub": "admin", "video": {"room": "r"}}).encode()
        )
        .rstrip(b"=")
        .decode()
    )
    forged = f"{header_seg}.{forged_payload}.{sig_seg}"
    with pytest.raises(TokenInvalid):
        decode_token(forged, SECRET, now=_fixed_clock(0))


def test_expired_token_is_rejected() -> None:
    token = AccessToken(
        api_key=KEY,
        api_secret=SECRET,
        identity="u",
        grant=VideoGrant(room="r"),
        ttl_seconds=60,
        _now=_fixed_clock(1000),
    ).to_jwt()
    # 61 s later the 60 s token has expired.
    with pytest.raises(TokenInvalid, match="expired"):
        decode_token(token, SECRET, now=_fixed_clock(1061))


def test_not_yet_valid_token_is_rejected() -> None:
    token = mint_access_token(
        api_key=KEY, api_secret=SECRET, identity="u", room="r", now=_fixed_clock(1000)
    )
    with pytest.raises(TokenInvalid, match="not yet valid"):
        decode_token(token, SECRET, now=_fixed_clock(500))


def test_malformed_tokens_are_rejected() -> None:
    with pytest.raises(TokenInvalid, match="3 segments"):
        decode_token("not.a.jwt.token", SECRET)
    with pytest.raises(TokenInvalid):
        decode_token("only-one-segment", SECRET)


def test_custom_grant_permissions_survive() -> None:
    token = AccessToken(
        api_key=KEY,
        api_secret=SECRET,
        identity="listener",
        grant=VideoGrant(room="r", can_publish=False, can_publish_data=False),
        _now=_fixed_clock(0),
    ).to_jwt()
    grant = decode_token(token, SECRET, now=_fixed_clock(0))["video"]
    assert grant["canPublish"] is False
    assert grant["canPublishData"] is False
    assert grant["canSubscribe"] is True


def test_missing_credentials_raise() -> None:
    with pytest.raises(ValueError, match="api_key and api_secret"):
        mint_access_token(api_key="", api_secret=SECRET, identity="u", room="r")
    with pytest.raises(ValueError, match="identity is required"):
        mint_access_token(api_key=KEY, api_secret=SECRET, identity="", room="r")


def test_signature_is_hmac_sha256_over_signing_input() -> None:
    # Lock in the exact construction LiveKit validates against.
    token = mint_access_token(
        api_key=KEY, api_secret=SECRET, identity="u", room="r", now=_fixed_clock(0)
    )
    signing_input, sig_seg = token.rsplit(".", 1)
    expected = hmac.new(SECRET.encode(), signing_input.encode("ascii"), hashlib.sha256).digest()
    pad = "=" * (-len(sig_seg) % 4)
    assert base64.urlsafe_b64decode(sig_seg + pad) == expected

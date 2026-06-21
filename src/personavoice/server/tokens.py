"""Mint LiveKit access tokens (JWTs) so clients can join rooms.

A LiveKit access token is a standard **HS256 JWT** signed with the project's API secret,
carrying a `video` grant (room name + publish/subscribe permissions). We mint it with the
stdlib (`hmac`/`hashlib`/`base64`/`json`) rather than pulling in the LiveKit server SDK, so
the token endpoint stays dependency-light and fully unit-testable without the `livekit`
extra. The claim shape is LiveKit's documented, stable one:

    header   {"alg": "HS256", "typ": "JWT"}
    payload  {"iss": <api-key>, "sub": <identity>, "nbf": ..., "exp": ...,
              "name": <display>, "metadata": <json string>,
              "video": {"room": ..., "roomJoin": true, "canPublish": true,
                        "canSubscribe": true, "canPublishData": true}}

`decode_token` verifies the signature and `exp`/`nbf`; it's used by the tests (and is handy
for debugging) so we never have to trust the encoder blindly.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Any

# We sign with HMAC-SHA256 ("HS256"), the algorithm every LiveKit SDK validates against.
_ALG = "HS256"


def _b64url_encode(data: bytes) -> str:
    """Base64url without padding, per the JWT spec (RFC 7515)."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(segment: str) -> bytes:
    """Inverse of `_b64url_encode`; re-pads to a multiple of 4 before decoding."""
    pad = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + pad)


def _encode_segment(obj: dict[str, Any]) -> str:
    # Compact separators keep the token small; key order is irrelevant to JWT validators.
    return _b64url_encode(json.dumps(obj, separators=(",", ":")).encode("utf-8"))


@dataclass
class VideoGrant:
    """The `video` grant in a LiveKit token: which room, and what the holder may do.

    Defaults are what an interactive voice client needs — join one room, publish its mic,
    subscribe to the agent's audio, and exchange data messages (used to switch persona
    mid-call).
    """

    room: str
    room_join: bool = True
    can_publish: bool = True
    can_subscribe: bool = True
    can_publish_data: bool = True

    def to_claim(self) -> dict[str, Any]:
        return {
            "room": self.room,
            "roomJoin": self.room_join,
            "canPublish": self.can_publish,
            "canSubscribe": self.can_subscribe,
            "canPublishData": self.can_publish_data,
        }


@dataclass
class AccessToken:
    """A LiveKit access token, built up then signed into a JWT with `to_jwt`."""

    api_key: str
    api_secret: str
    identity: str
    grant: VideoGrant
    name: str | None = None
    metadata: str | None = None
    ttl_seconds: int = 3600
    # Injectable clock so tests are deterministic (defaults to wall-clock seconds).
    _now: Any = field(default=time.time, repr=False)

    def to_jwt(self) -> str:
        if not self.api_key or not self.api_secret:
            raise ValueError("api_key and api_secret are required to mint a token")
        if not self.identity:
            raise ValueError("identity is required to mint a token")

        issued = int(self._now())
        payload: dict[str, Any] = {
            "iss": self.api_key,
            "sub": self.identity,
            "nbf": issued,
            "exp": issued + int(self.ttl_seconds),
            "video": self.grant.to_claim(),
        }
        if self.name:
            payload["name"] = self.name
        if self.metadata is not None:
            payload["metadata"] = self.metadata

        signing_input = f"{_encode_segment({'alg': _ALG, 'typ': 'JWT'})}.{_encode_segment(payload)}"
        signature = hmac.new(
            self.api_secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256
        ).digest()
        return f"{signing_input}.{_b64url_encode(signature)}"


def mint_access_token(
    *,
    api_key: str,
    api_secret: str,
    identity: str,
    room: str,
    name: str | None = None,
    metadata: str | None = None,
    ttl_seconds: int = 3600,
    now: Any = time.time,
) -> str:
    """Convenience wrapper: build a default `VideoGrant` for `room` and sign the token."""
    token = AccessToken(
        api_key=api_key,
        api_secret=api_secret,
        identity=identity,
        grant=VideoGrant(room=room),
        name=name,
        metadata=metadata,
        ttl_seconds=ttl_seconds,
        _now=now,
    )
    return token.to_jwt()


class TokenInvalid(ValueError):
    """A JWT failed signature, structure, or expiry/not-before validation."""


def decode_token(token: str, api_secret: str, *, now: Any = time.time) -> dict[str, Any]:
    """Verify a LiveKit JWT's HS256 signature and `exp`/`nbf`, returning its claims.

    Raises `TokenInvalid` on a bad signature, malformed token, or expired/not-yet-valid
    window. Used by the tests (and useful for debugging an integration); the server itself
    only mints, it doesn't verify — LiveKit does that.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise TokenInvalid("token is not a well-formed JWT (expected 3 segments)")
    header_seg, payload_seg, sig_seg = parts

    try:
        header = json.loads(_b64url_decode(header_seg))
        payload = json.loads(_b64url_decode(payload_seg))
    except (ValueError, json.JSONDecodeError) as exc:
        raise TokenInvalid(f"token segments are not valid base64url JSON: {exc}") from exc

    if header.get("alg") != _ALG:
        raise TokenInvalid(f"unexpected JWT alg {header.get('alg')!r}; expected {_ALG}")

    expected = hmac.new(
        api_secret.encode("utf-8"), f"{header_seg}.{payload_seg}".encode("ascii"), hashlib.sha256
    ).digest()
    if not hmac.compare_digest(expected, _b64url_decode(sig_seg)):
        raise TokenInvalid("token signature does not match the API secret")

    current = int(now())
    if "nbf" in payload and current < int(payload["nbf"]):
        raise TokenInvalid("token is not yet valid (nbf in the future)")
    if "exp" in payload and current >= int(payload["exp"]):
        raise TokenInvalid("token has expired")
    return payload

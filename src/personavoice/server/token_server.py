"""A tiny HTTP token server so thin clients can join the LiveKit room.

The Flutter client (and the LiveKit Agents Playground, and any other LiveKit client) needs
two things to connect: the LiveKit server URL and a signed access token scoped to a room.
The agent worker (`orchestrator/agent.py`) already auto-joins new rooms; what was missing
is something to *hand a client a token*. This module is that piece — "fat server, thin
client": the secret never leaves the server, the client just asks for a token.

It's deliberately stdlib-only (`http.server`): low-traffic, dev/self-hosted, and no new core
dependency. The request handling lives in `TokenService` (pure, unit-tested); the
`BaseHTTPRequestHandler` is a thin shell that maps HTTP ↔ `TokenService` calls.

Endpoints (all JSON, permissive CORS so a browser client / Playground can call them):

    GET    /healthz          -> {"status": "ok", "backend": ...}
    GET    /personas         -> {"personas": [{"id","name","description","voice"}], "default": <id>}
    GET    /voices           -> {"voices": [VoiceOption...], "tts", "supports_cloning"}
    POST   /token            -> mint a token; body: {"room"?, "identity"?, "persona"?,
                                "voice"?, "cefr"?, "demeanor"?}
    GET    /token?room=&identity=&persona=&voice=&cefr=&demeanor=  (same, for manual testing)
    POST   /voices/clone?name=&text=&authorized=  -> enroll a clone; raw wav as the body
    DELETE /voices/clone/{name}                   -> remove a cloned voice

`/token` returns `{"url","token","room","identity","persona","voice","cefr","demeanor"}`.
Persona and session-options selection on the live agent ride the existing data-message path:
the client connects, then publishes a `{"persona": <id>, "voice": ..., "cefr": ...,
"demeanor": ...}` data message which the agent's `on("data_received")` handler applies (see
`orchestrator/agent.py`). We echo the resolved values back so the client knows what to send.
If `PERSONAVOICE_API_TOKEN` is set, requests must carry `Authorization: Bearer …`.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import re
import ssl
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..models import CEFRLevel, Demeanor, SessionOptions
from ..persona.registry import PersonaRegistry
from ..voice.clone import CloneError, VoiceCloner
from ..voice.registry import VoiceRegistry
from .config import Settings, load_voice_registry
from .ratelimit import RateLimiter, rate_limiter_from_env
from .security import audit_security, has_errors
from .tokens import mint_access_token

logger = logging.getLogger("personavoice.token_server")

# A clone name is a bare identifier (usable as a filename and a `voices/<name>` ref).
_VOICE_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")
# Cap an uploaded clone sample (bytes) so an enroll can't exhaust memory/disk. ~10 s of
# 24 kHz mono PCM wav is well under this; override with PERSONAVOICE_MAX_CLONE_BYTES.
_DEFAULT_MAX_CLONE_BYTES = 10 * 1024 * 1024
# Cap clones per deployment (abuse surface); override with PERSONAVOICE_MAX_CLONES.
_DEFAULT_MAX_CLONES = 50

# Lazily builds a `VoiceCloner` bound to the active backend + the shared clones store.
ClonerFactory = Callable[[], VoiceCloner]


class TokenServiceError(Exception):
    """A client-facing error with an HTTP status code."""

    status = 400

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        if status is not None:
            self.status = status


class Unauthorized(TokenServiceError):
    status = 401


class BadRequest(TokenServiceError):
    status = 400


class ServerMisconfigured(TokenServiceError):
    status = 500


class TooManyRequests(TokenServiceError):
    status = 429


class PayloadTooLarge(TokenServiceError):
    status = 413


@dataclass
class TokenServiceConfig:
    """LiveKit credentials + token policy, resolved from the environment."""

    livekit_url: str
    api_key: str
    api_secret: str
    # If set, requests must present `Authorization: Bearer <api_token>`. Empty = open (dev).
    api_token: str | None = None
    token_ttl: int = 3600
    # Prod hardening: refuse to start wide-open, and optional TLS for the HTTP server.
    require_auth: bool = False
    tls_cert: str | None = None
    tls_key: str | None = None

    @property
    def tls_enabled(self) -> bool:
        return bool(self.tls_cert and self.tls_key)

    @classmethod
    def from_env(cls) -> TokenServiceConfig:
        ttl_raw = os.getenv("PERSONAVOICE_TOKEN_TTL", "3600").strip()
        try:
            ttl = int(ttl_raw) if ttl_raw else 3600
        except ValueError as exc:
            raise ServerMisconfigured(
                f"PERSONAVOICE_TOKEN_TTL must be an integer, got {ttl_raw!r}"
            ) from exc
        api_token = (os.getenv("PERSONAVOICE_API_TOKEN") or "").strip() or None
        require_auth = (os.getenv("PERSONAVOICE_REQUIRE_AUTH") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        return cls(
            livekit_url=(os.getenv("LIVEKIT_URL") or "").strip(),
            api_key=(os.getenv("LIVEKIT_API_KEY") or "").strip(),
            api_secret=(os.getenv("LIVEKIT_API_SECRET") or "").strip(),
            api_token=api_token,
            token_ttl=ttl,
            require_auth=require_auth,
            tls_cert=(os.getenv("PERSONAVOICE_TLS_CERT") or "").strip() or None,
            tls_key=(os.getenv("PERSONAVOICE_TLS_KEY") or "").strip() or None,
        )


def _new_room() -> str:
    return f"pv-{uuid.uuid4().hex[:12]}"


def _new_identity() -> str:
    return f"user-{uuid.uuid4().hex[:8]}"


class TokenService:
    """Resolves personas and mints room tokens. Pure (no sockets) so it's unit-testable."""

    def __init__(
        self,
        config: TokenServiceConfig,
        registry: PersonaRegistry,
        *,
        default_persona: str,
        backend: str = "",
        voices: VoiceRegistry | None = None,
        tts_name: str = "",
        supports_cloning: bool = False,
        cloner_factory: ClonerFactory | None = None,
        max_clones: int = _DEFAULT_MAX_CLONES,
    ) -> None:
        self._config = config
        self._registry = registry
        self._default_persona = default_persona
        self._backend = backend
        # Optional: resolves a persona's voice ref to a human description for the picker,
        # backs the /voices catalog, and validates a chosen voice. None (e.g. in unit tests)
        # just omits the voice blurb and disables voice validation/catalog.
        self._voices = voices
        # The active TTS adapter name + whether it can clone — drives catalog availability and
        # gates enrollment. `cloner_factory` lazily builds a cloner for enrollment (None on a
        # non-cloning backend or when no factory was wired).
        self._tts_name = tts_name
        self._supports_cloning = supports_cloning
        self._cloner_factory = cloner_factory
        self._max_clones = max_clones

    def check_auth(self, authorization: str | None) -> None:
        """Enforce the optional bearer token. No-op when `api_token` is unset (dev mode)."""
        expected = self._config.api_token
        if not expected:
            return
        prefix = "Bearer "
        if not authorization or not authorization.startswith(prefix):
            raise Unauthorized("missing or malformed Authorization header")
        # Constant-time comparison so a timing side-channel can't leak the token.
        if not hmac.compare_digest(authorization[len(prefix) :].strip(), expected):
            raise Unauthorized("invalid API token")

    def personas(self) -> dict[str, Any]:
        """List selectable personas (hot-reloaded from disk) and the default id.

        Each entry carries enough for a rich picker: a one-line `description` and a
        human `voice` blurb (resolved via the voice registry, "" when unavailable).
        """
        self._registry.reload()
        return {
            "personas": [
                {
                    "id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "voice": self._voices.describe(p.voice.ref) if self._voices else "",
                }
                for p in self._registry.all()
            ],
            "default": self._default_persona,
        }

    def _validate_session_options(
        self, *, voice: str | None, cefr: str | None, demeanor: str | None
    ) -> SessionOptions:
        """Validate the per-session overrides, raising `BadRequest` on an unknown value.

        Voice is checked against the selectable catalog when a registry is available; CEFR and
        demeanor are checked against their enums (case-insensitively). Returns canonicalized
        `SessionOptions` (empty fields left None).
        """
        opts = SessionOptions()
        voice = (voice or "").strip()
        if voice:
            if self._voices is not None and voice not in self._voices.choice_ids():
                known = ", ".join(self._voices.choice_ids()) or "(none)"
                raise BadRequest(f"unknown voice {voice!r}; known: {known}")
            opts.voice = voice
        cefr = (cefr or "").strip()
        if cefr:
            try:
                opts.cefr = CEFRLevel(cefr.upper())
            except ValueError as exc:
                allowed = ", ".join(level.value for level in CEFRLevel)
                raise BadRequest(f"unknown cefr {cefr!r}; expected one of {allowed}") from exc
        demeanor = (demeanor or "").strip()
        if demeanor:
            try:
                opts.demeanor = Demeanor(demeanor.lower())
            except ValueError as exc:
                allowed = ", ".join(d.value for d in Demeanor)
                raise BadRequest(
                    f"unknown demeanor {demeanor!r}; expected one of {allowed}"
                ) from exc
        return opts

    def issue(
        self,
        *,
        room: str | None = None,
        identity: str | None = None,
        persona: str | None = None,
        voice: str | None = None,
        cefr: str | None = None,
        demeanor: str | None = None,
    ) -> dict[str, Any]:
        """Mint a LiveKit token for a room, validating the persona and session options.

        Missing `room`/`identity` are generated. A requested persona must exist (else
        `BadRequest`); none requested falls back to the default. Per-session overrides
        (`voice`/`cefr`/`demeanor`) are validated the same way. Persona + options are embedded
        in the token metadata for observability and echoed back so the client can apply them
        via a data message after connecting.
        """
        if not (self._config.livekit_url and self._config.api_key and self._config.api_secret):
            raise ServerMisconfigured(
                "LiveKit is not configured; set LIVEKIT_URL, LIVEKIT_API_KEY and "
                "LIVEKIT_API_SECRET in the environment"
            )

        room = (room or "").strip() or _new_room()
        identity = (identity or "").strip() or _new_identity()

        persona_id = (persona or "").strip() or self._default_persona
        if persona_id and persona_id not in self._registry:
            self._registry.reload()  # tolerate a persona added since startup
        if persona_id and persona_id not in self._registry:
            known = ", ".join(self._registry.ids()) or "(none)"
            raise BadRequest(f"unknown persona {persona_id!r}; known: {known}")

        options = self._validate_session_options(voice=voice, cefr=cefr, demeanor=demeanor)

        meta: dict[str, Any] = {}
        if persona_id:
            meta["persona"] = persona_id
        meta.update(options.model_dump(exclude_none=True, mode="json"))

        token = mint_access_token(
            api_key=self._config.api_key,
            api_secret=self._config.api_secret,
            identity=identity,
            room=room,
            name=identity,
            metadata=json.dumps(meta) if meta else None,
            ttl_seconds=self._config.token_ttl,
        )
        return {
            "url": self._config.livekit_url,
            "token": token,
            "room": room,
            "identity": identity,
            "persona": persona_id,
            "voice": options.voice,
            "cefr": options.cefr.value if options.cefr else None,
            "demeanor": options.demeanor.value if options.demeanor else None,
        }

    # -- voice library -----------------------------------------------------------------

    def voices_catalog(self) -> dict[str, Any]:
        """The selectable voice catalog for the active backend (for a client picker)."""
        options = (
            self._voices.catalog(self._tts_name, supports_cloning=self._supports_cloning)
            if self._voices is not None
            else []
        )
        return {
            "voices": [o.model_dump() for o in options],
            "backend": self._backend,
            "tts": self._tts_name,
            "supports_cloning": self._supports_cloning,
        }

    async def enroll_voice(
        self, *, audio: bytes, name: str, ref_text: str | None = None, authorized: bool = False
    ) -> dict[str, Any]:
        """Clone a voice from an uploaded sample and add it to the library.

        Gated like the rest of the server (auth at the HTTP layer). Requires a cloning backend
        and an explicit `authorized` acknowledgement (the caller affirms they may use the
        voice). Validates the sample, enforces the per-deployment quota, then runs the cascade
        cloner (which transcribes the reference text for reference-text backends and persists
        the clone). Returns the new voice's catalog entry.
        """
        if not self._supports_cloning or self._cloner_factory is None:
            raise BadRequest(
                f"the active TTS backend {self._tts_name or '(unknown)'!r} can't clone voices; "
                "switch to a cloning backend (f5_mlx on Mac, chatterbox on CUDA)"
            )
        if not authorized:
            raise BadRequest(
                "voice enrollment requires acknowledging you're authorized to use this voice "
                "(set authorized=true)"
            )
        name = (name or "").strip()
        if not name or not _VOICE_NAME_RE.fullmatch(name):
            raise BadRequest(f"invalid voice name {name!r}; use letters, digits, '-' or '_' only")
        if not audio:
            raise BadRequest("no audio sample provided")
        store = self._voices.clones if self._voices is not None else None
        if store is not None and name not in store and len(store) >= self._max_clones:
            raise BadRequest(f"clone quota reached ({self._max_clones}); delete a voice first")
        cloner = self._cloner_factory()
        try:
            await cloner.clone(audio, name, ref_text=ref_text)
        except CloneError as exc:
            raise BadRequest(str(exc)) from exc
        emotion = None
        ref = (
            self._voices.resolve_choice(
                name, self._tts_name, supports_cloning=self._supports_cloning
            )
            if self._voices is not None
            else None
        )
        if ref is not None:
            emotion = ref.emotion
        return {
            "id": name,
            "name": name,
            "kind": "clone",
            "emotion": emotion,
            "available": self._supports_cloning,
            "reason": None,
        }

    def delete_voice(self, name: str) -> dict[str, Any]:
        """Remove a cloned voice from the library (and any persona assignments to it)."""
        store = self._voices.clones if self._voices is not None else None
        if store is None:
            raise ServerMisconfigured("voice library is not configured")
        name = (name or "").strip()
        if not store.remove(name):
            raise BadRequest(f"unknown clone {name!r}")
        return {"deleted": name}


def _active_tts(settings: Settings) -> tuple[str, bool]:
    """Resolve the active TTS adapter name + whether it can clone, without loading models.

    Reads the backend YAML and consults the adapter table's class attribute. Tolerates a
    missing/invalid backend config (returns unknown / no-cloning) so /voices still answers.
    """
    from ..adapters.factory import TTS_ADAPTERS
    from .config import ConfigError, load_backend_config

    try:
        config = load_backend_config(settings)
    except ConfigError:
        return "", False
    tts_name = config.tts.adapter
    cls = TTS_ADAPTERS.get(tts_name)
    return tts_name, bool(getattr(cls, "supports_cloning", False))


def build_service(settings: Settings | None = None) -> TokenService:
    """Assemble a `TokenService` from process settings + env LiveKit credentials."""
    settings = settings or Settings.load()
    registry = PersonaRegistry(settings.personas_dir)
    default_persona = os.getenv("PERSONAVOICE_PERSONA", "").strip()
    if default_persona not in registry:
        default_persona = registry.ids()[0] if len(registry) else ""
    # Voice registry for picker blurbs, the /voices catalog, and choice validation. Attaches
    # the clone + fine-tuned stores so those voices are selectable; tolerates missing files.
    voices = load_voice_registry(settings)
    tts_name, supports_cloning = _active_tts(settings)

    cloner_factory: ClonerFactory | None = None
    if supports_cloning and voices.clones is not None:
        # Lazily build a cloner bound to the active backend + the *shared* clones store, so a
        # newly enrolled clone shows up in the catalog immediately. Backend construction is
        # cheap (models load lazily inside clone()).
        from ..adapters.factory import build_backend
        from .config import load_backend_config

        store = voices.clones

        def cloner_factory() -> VoiceCloner:  # type: ignore[misc]
            backend = build_backend(load_backend_config(settings))
            return VoiceCloner(backend, store)

    max_clones = int(os.getenv("PERSONAVOICE_MAX_CLONES", str(_DEFAULT_MAX_CLONES)).strip() or 0)
    return TokenService(
        TokenServiceConfig.from_env(),
        registry,
        default_persona=default_persona,
        backend=settings.backend,
        voices=voices,
        tts_name=tts_name,
        supports_cloning=supports_cloning,
        cloner_factory=cloner_factory,
        max_clones=max_clones,
    )


# --------------------------------------------------------------------------------------
# HTTP shell
# --------------------------------------------------------------------------------------


def _env_max_clone_bytes() -> int:
    raw = os.getenv("PERSONAVOICE_MAX_CLONE_BYTES", str(_DEFAULT_MAX_CLONE_BYTES)).strip()
    try:
        return int(raw) if raw else _DEFAULT_MAX_CLONE_BYTES
    except ValueError:
        return _DEFAULT_MAX_CLONE_BYTES


def _as_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _make_handler(
    service: TokenService, limiter: RateLimiter | None = None
) -> type[BaseHTTPRequestHandler]:
    rl: RateLimiter = limiter or RateLimiter(rate=0.0, burst=0.0)  # disabled by default
    max_clone_bytes = _env_max_clone_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "personavoice-token/0.1"

        # Quiet by default; route access logs through our logger at debug level.
        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug("%s - %s", self.address_string(), fmt % args)

        def _send_json(self, status: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            # Permissive CORS: the token endpoint is meant to be called from clients.
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.end_headers()
            self.wfile.write(payload)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise BadRequest(f"request body is not valid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise BadRequest("request body must be a JSON object")
            return obj

        def _read_raw_body(self, max_bytes: int) -> bytes:
            """Read the raw request body, rejecting an oversize upload with 413."""
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return b""
            if length > max_bytes:
                raise PayloadTooLarge(f"upload too large ({length} bytes); max {max_bytes}")
            return self.rfile.read(length)

        def _dispatch(self, method: str) -> None:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            try:
                if path == "/healthz" and method == "GET":
                    self._send_json(200, {"status": "ok", "backend": service._backend})
                    return
                # Rate-limit everything else (before auth, to throttle unauthenticated floods)
                # keyed by client IP.
                if not rl.allow(self.client_address[0]):
                    raise TooManyRequests("rate limit exceeded")
                # Everything below is protected by the optional bearer token.
                service.check_auth(self.headers.get("Authorization"))
                if path == "/personas" and method == "GET":
                    self._send_json(200, service.personas())
                elif path == "/voices" and method == "GET":
                    self._send_json(200, service.voices_catalog())
                elif path == "/voices/clone" and method == "POST":
                    self._handle_clone(query)
                elif path.startswith("/voices/clone/") and method == "DELETE":
                    name = path[len("/voices/clone/") :]
                    self._send_json(200, service.delete_voice(name))
                elif path == "/token" and method in ("GET", "POST"):
                    body = self._read_json_body() if method == "POST" else {}
                    result = service.issue(
                        room=body.get("room") or query.get("room"),
                        identity=body.get("identity") or query.get("identity"),
                        persona=body.get("persona") or query.get("persona"),
                        voice=body.get("voice") or query.get("voice"),
                        cefr=body.get("cefr") or query.get("cefr"),
                        demeanor=body.get("demeanor") or query.get("demeanor"),
                    )
                    self._send_json(200, result)
                else:
                    self._send_json(404, {"error": f"no route for {method} {path}"})
            except TokenServiceError as exc:
                self._send_json(exc.status, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover - defensive 500
                logger.exception("token server error")
                self._send_json(500, {"error": f"internal error: {exc}"})

        def _handle_clone(self, query: dict[str, str]) -> None:
            """POST /voices/clone: raw wav body + `name`/`text`/`authorized` query params.

            The sample rides as the raw request body (Content-Type audio/wav) rather than
            multipart — zero-dependency and stdlib-only (Python 3.13 dropped `cgi`). Cloning
            loads models, so it runs in this request's worker thread via `asyncio.run`.
            """
            audio = self._read_raw_body(max_clone_bytes)
            result = asyncio.run(
                service.enroll_voice(
                    audio=audio,
                    name=query.get("name", ""),
                    ref_text=query.get("text") or None,
                    authorized=_as_bool(query.get("authorized")),
                )
            )
            self._send_json(201, result)

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def do_DELETE(self) -> None:
            self._dispatch("DELETE")

        def do_OPTIONS(self) -> None:
            self._send_json(204, {})

    return Handler


def make_server(
    service: TokenService,
    host: str = "0.0.0.0",
    port: int = 8080,
    *,
    limiter: RateLimiter | None = None,
) -> ThreadingHTTPServer:
    """Create (but don't start) a threaded HTTP server, with TLS if the config provides certs."""
    httpd = ThreadingHTTPServer((host, port), _make_handler(service, limiter))
    cfg = service._config
    if cfg.tls_enabled:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=cfg.tls_cert, keyfile=cfg.tls_key)  # type: ignore[arg-type]
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    return httpd


def run(settings: Settings | None = None) -> None:
    """Build the service from env and serve forever (blocking). Used by `--token-server`.

    Runs a security audit first: in strict mode (`PERSONAVOICE_REQUIRE_AUTH=1`) an
    `error`-level finding — e.g. no API token — refuses to start rather than serve wide open.
    """
    settings = settings or Settings.load()
    service = build_service(settings)
    cfg = service._config
    limiter = rate_limiter_from_env()
    host = os.getenv("PERSONAVOICE_HOST", "0.0.0.0").strip() or "0.0.0.0"
    port = int(os.getenv("PERSONAVOICE_PORT", "8080").strip() or "8080")

    findings = audit_security(
        api_token=cfg.api_token,
        livekit_url=cfg.livekit_url,
        tls_enabled=cfg.tls_enabled,
        rate_limited=limiter.enabled,
        require_auth=cfg.require_auth,
        bind_host=host,
    )
    for f in findings:
        if f.level == "error":
            logger.error("security: %s", f.message)
        elif f.level == "warning":
            logger.warning("security: %s", f.message)
    if cfg.require_auth and has_errors(findings):
        raise ServerMisconfigured(
            "refusing to start: PERSONAVOICE_REQUIRE_AUTH is set but the security audit found "
            "blocking issues (see the security errors above)"
        )

    if not (cfg.livekit_url and cfg.api_key and cfg.api_secret):
        logger.warning(
            "LiveKit credentials are not fully set; /token will 500 until LIVEKIT_URL, "
            "LIVEKIT_API_KEY and LIVEKIT_API_SECRET are provided"
        )

    httpd = make_server(service, host, port, limiter=limiter)
    scheme = "https" if cfg.tls_enabled else "http"
    auth = "on" if cfg.api_token else "off (open)"
    rl = f"{limiter.rate:g} rps" if limiter.enabled else "off"
    print(f"Token server listening on {scheme}://{host}:{port}  (auth: {auth}, rate-limit: {rl})")
    print(f"  LiveKit URL: {cfg.livekit_url or '(unset)'}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI shim
    from ..obs import configure_logging

    configure_logging()
    run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

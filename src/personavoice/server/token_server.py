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
    GET    /metrics          -> Prometheus exposition (per-turn latency/outcome counters);
                                empty when prometheus_client isn't installed
    GET    /personas[?user=] -> {"personas": [{"id","name","description","voice","cefr",
                                "demeanor","custom"}], "default": <id>}
                                (curated + the user's custom personas)
    POST   /personas?user=   -> create a custom persona; body = a persona draft
    POST   /personas/draft?user= -> draft a persona from {"description": ...} (LLM); not persisted
    GET    /personas/{id}?user=   -> one persona's full body (for an edit form)
    PUT    /personas/{id}?user=   -> replace one of the user's own personas
    DELETE /personas/{id}?user=   -> delete one of the user's own personas
    GET    /loras            -> {"loras": [LoraOption...], "llm", "supports_lora"}
    GET    /voices           -> {"voices": [VoiceOption...], "tts", "supports_cloning"}
    GET    /consent?user=    -> {"user","granted","allow_training","updated_at"}
    POST   /consent?user=    -> set it; body: {"granted": bool, "allow_training"?: bool}
    POST   /token            -> mint a token; body: {"room"?, "identity"?, "name"?, "persona"?,
                                "voice"?, "cefr"?, "demeanor"?, "user"?}
    GET    /token?room=&identity=&name=&persona=&voice=&cefr=&demeanor=&user=  (manual testing)
    POST   /voices/clone?name=&text=&authorized=  -> enroll a clone; raw wav as the body
    DELETE /voices/clone/{name}                   -> remove a cloned voice

`identity` is the unique LiveKit participant id (`sub`); `name` is the cosmetic display name
(falls back to `identity`). `/token` returns
`{"url","token","room","identity","name","persona","voice","cefr","demeanor","user"}`.
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
from collections.abc import Callable, Collection
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from pydantic import ValidationError

from ..memory import MemoryStore
from ..models import CEFRLevel, Demeanor, Persona, SessionOptions
from ..obs import read_sessions, render_metrics
from ..orchestrator.busy import busy_retry_after
from ..persona import author
from ..persona.author import PersonaDraftError
from ..persona.lora import served_loras
from ..persona.registry import PersonaRegistry
from ..persona.store import UserPersonaStore
from ..voice.clone import CloneError, VoiceCloner
from ..voice.registry import VoiceRegistry
from ..voice.seed import seed_voice_names
from .config import (
    Settings,
    load_memory_store,
    load_user_persona_store,
    load_voice_registry,
    max_sessions,
)
from .ratelimit import RateLimiter, rate_limiter_from_env
from .security import audit_security, has_errors
from .tokens import mint_access_token

logger = logging.getLogger("personavoice.token_server")

# A clone name is a bare identifier (usable as a filename and a `voices/<name>` ref).
_VOICE_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")
# A user id scopes custom personas (and clones). Derived from the token identity/metadata;
# permissive enough for the generated `user-xxxx` ids and email-like identities.
_USER_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@-]{0,127}")
# Turn a persona name into a slug id (lowercase, dashes); falls back to "persona" when empty.
_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")
# Cap custom personas per user (abuse surface); override with PERSONAVOICE_MAX_USER_PERSONAS.
_DEFAULT_MAX_USER_PERSONAS = 50
# Cap an uploaded clone sample (bytes) so an enroll can't exhaust memory/disk. ~10 s of
# 24 kHz mono PCM wav is well under this; override with PERSONAVOICE_MAX_CLONE_BYTES.
_DEFAULT_MAX_CLONE_BYTES = 10 * 1024 * 1024
# Cap clones per deployment (abuse surface); override with PERSONAVOICE_MAX_CLONES.
_DEFAULT_MAX_CLONES = 50

# Lazily builds a `VoiceCloner` bound to the active backend + the shared clones store.
ClonerFactory = Callable[[], VoiceCloner]

# Lazily builds the cascade LLM adapter used to draft a persona from a description (Feature K).
# Returns the active backend's `.llm` (a `persona.author._ChatLLM`); None disables /personas/draft.
DraftLLMFactory = Callable[[], Any]

# Reads the live `(active, capacity)` session counts the agent worker publishes (the gauge backing
# `/healthz`), or None when unknown (exporter off / no worker reporting). Injectable so the optional
# early gate is unit-testable without a running worker.
SessionsReader = Callable[[], "tuple[int, int] | None"]


class TokenServiceError(Exception):
    """A client-facing error with an HTTP status code (and optional headers / body fields)."""

    status = 400

    def __init__(
        self,
        message: str,
        status: int | None = None,
        *,
        headers: dict[str, str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        if status is not None:
            self.status = status
        # Extra response headers (e.g. `Retry-After` on a 503) and extra JSON body fields merged
        # alongside `{"error": ...}` (e.g. `retry_after`) — both empty for the common case.
        self.headers = headers
        self.extra = extra or {}


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


class Unavailable(TokenServiceError):
    """All workers are at capacity (admission control's optional token-server early gate)."""

    status = 503


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


def _slugify(name: str) -> str:
    """A bare-identifier id from a persona name (lowercase, dashes), e.g. 'My Tutor' → 'my-tutor'."""
    slug = _SLUG_STRIP_RE.sub("-", name.strip().lower()).strip("-")
    return slug or "persona"


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
        protected_voices: Collection[str] = (),
        user_personas: UserPersonaStore | None = None,
        max_user_personas: int = _DEFAULT_MAX_USER_PERSONAS,
        draft_llm_factory: DraftLLMFactory | None = None,
        llm_name: str = "",
        supports_lora: bool = False,
        lora_modules: str | None = None,
        memory: MemoryStore | None = None,
        admission_gate: bool = False,
        sessions_reader: SessionsReader = read_sessions,
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
        # User-dropped "inbox" seed clones the user can't delete: the catalog marks them
        # `removable=False` and `delete_voice` rejects them. Bundled voices now ship as presets
        # (already non-removable), so the builder excludes preset stems from this set.
        self._protected_voices = frozenset(protected_voices)
        # Multi-user custom personas (N3): None disables the authoring routes. Scoped per
        # user id; curated personas always win on id clash and are never stored/deletable here.
        self._user_personas = user_personas
        self._max_user_personas = max_user_personas
        # Optional persona authoring helper (Feature K): builds the cascade LLM on demand to draft a
        # persona from a description. None disables /personas/draft (e.g. in unit tests, or when no
        # custom-persona store is wired). Drafting is one-shot and runs in the request thread.
        self._draft_llm_factory = draft_llm_factory
        # Active LLM adapter name + whether it can hot-swap LoRA, plus the served `--lora-modules`
        # spec — drives the /loras picker and validates a custom persona's `llm.lora`.
        self._llm_name = llm_name
        self._supports_lora = supports_lora
        self._lora_modules = lora_modules
        # Per-user conversation memory store: backs the /consent routes (the same `consent.json`
        # the cascade checks before recording). None disables them (e.g. in unit tests).
        self._memory = memory
        # Admission control's optional token-server early gate (Feature I, step 4): when on, `issue`
        # returns 503 + Retry-After once the worker reports full instead of minting a token the
        # caller can't use. `sessions_reader` is the live-count source (the shared gauge). It's an
        # *optimization* over the worker's load gate — it has a read→mint→connect TOCTOU race — and
        # fails open (mints) when the count is unknown, so it never blocks on its own.
        self._admission_gate = admission_gate
        self._sessions_reader = sessions_reader

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

    def _persona_summary(self, persona: Persona, *, custom: bool) -> dict[str, Any]:
        """Picker entry for one persona: id/name/description, a voice blurb, the baked-in
        session defaults (CEFR / demeanor), and editability.

        The CEFR/demeanor defaults let the client show what a persona speaks like before any
        per-call override; both are `None` for the curated personas (none set them today) and
        for any custom persona that left them unset.
        """
        defaults = persona.session_defaults
        return {
            "id": persona.id,
            "name": persona.name,
            "description": persona.description,
            "voice": self._voices.describe(persona.voice.ref) if self._voices else "",
            "cefr": defaults.cefr.value if defaults.cefr else None,
            "demeanor": defaults.demeanor.value if defaults.demeanor else None,
            "custom": custom,  # True = user-authored (editable/deletable); False = curated
        }

    def personas(self, user: str | None = None) -> dict[str, Any]:
        """List selectable personas (hot-reloaded from disk) and the default id.

        Curated personas (`config/personas`) are always returned. When `user` is given and the
        custom-persona store is enabled, that user's own personas are merged in too — curated
        win on id clash (so a user can't shadow a built-in). Each entry carries enough for a
        rich picker: a one-line `description`, a human `voice` blurb, and a `custom` flag the
        client uses to show edit/delete only on user-authored personas.
        """
        self._registry.reload()
        curated_ids = set(self._registry.ids())
        out = [self._persona_summary(p, custom=False) for p in self._registry.all()]
        user = (user or "").strip()
        if user and self._user_personas is not None:
            self._user_personas.reload()  # pick up personas created since startup
            for p in self._user_personas.list_for(user):
                if p.id in curated_ids:
                    continue  # curated win on id clash
                out.append(self._persona_summary(p, custom=True))
        return {"personas": out, "default": self._default_persona}

    def _is_user_persona(self, user: str, persona_id: str) -> bool:
        """Whether `persona_id` is one of `user`'s own custom personas (store hot-reloaded)."""
        if not user or self._user_personas is None:
            return False
        self._user_personas.reload()
        return self._user_personas.has(user, persona_id)

    def get_persona(self, user: str | None, persona_id: str) -> dict[str, Any]:
        """Return one persona's full stored body — for prefilling an edit form.

        The list route (`personas`) returns only a picker summary; the form needs the whole
        persona (system prompt, llm, voice, behavior, memory, session defaults). A curated id
        resolves for anyone (read-only); a custom id resolves only for its owning `user`.
        """
        persona_id = (persona_id or "").strip()
        self._registry.reload()
        if persona_id in self._registry:
            curated = self._registry.get(persona_id)
            return {
                **self._persona_summary(curated, custom=False),
                "persona": curated.model_dump(mode="json"),
            }
        user = (user or "").strip()
        if user and self._user_personas is not None:
            self._user_personas.reload()
            persona = self._user_personas.get(user, persona_id)
            if persona is not None:
                return self._stored_persona_payload(persona)
        raise BadRequest(f"unknown persona {persona_id!r}")

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

    def _reject_if_full(self) -> None:
        """Refuse to mint a token when the worker reports full (the optional 503 early gate).

        No-op unless the gate is enabled. Reads the live count from the shared gauge and compares
        against the *config* capacity (`max_sessions()`, the authoritative value `/healthz` also
        reports). When the count is unknown (None — exporter off or no worker has reported) it fails
        open and mints, leaving the worker's load gate as the real rail. Raises `Unavailable` (503)
        with a `Retry-After` header and a `retry_after` body field when at/over capacity.
        """
        if not self._admission_gate:
            return
        sessions = self._sessions_reader()
        if sessions is None:
            return  # can't see the worker's load — fail open; the load gate is the real rail
        active, _gauge_capacity = sessions
        capacity = max_sessions()
        if active >= capacity:
            retry = busy_retry_after()
            logger.info("admission gate: at capacity (%d/%d); returning 503", active, capacity)
            raise Unavailable(
                "all sessions are busy right now; please retry shortly",
                headers={"Retry-After": str(retry)},
                extra={"retry_after": retry},
            )

    def issue(
        self,
        *,
        room: str | None = None,
        identity: str | None = None,
        name: str | None = None,
        persona: str | None = None,
        voice: str | None = None,
        cefr: str | None = None,
        demeanor: str | None = None,
        user: str | None = None,
    ) -> dict[str, Any]:
        """Mint a LiveKit token for a room, validating the persona and session options.

        Missing `room`/`identity` are generated. `identity` is the LiveKit participant id
        (`sub`) — it should be stable and unique; `name` is the human-facing display name (the
        JWT `name` claim), which may change freely and need not be unique. They're separate so
        renaming a participant doesn't change its identity; `name` falls back to `identity`
        when omitted. A requested persona must exist (else
        `BadRequest`); none requested falls back to the default. A `user` scopes custom-persona
        resolution: it lets a caller request one of their *own* personas (curated win on clash)
        and is embedded in the token metadata so the agent resolves it at call time (and keys
        memory to a stable id). Per-session overrides (`voice`/`cefr`/`demeanor`) are validated
        the same way. Persona + options + user are embedded in the token metadata for the agent
        and echoed back so the client can apply them via a data message after connecting.
        """
        if not (self._config.livekit_url and self._config.api_key and self._config.api_secret):
            raise ServerMisconfigured(
                "LiveKit is not configured; set LIVEKIT_URL, LIVEKIT_API_KEY and "
                "LIVEKIT_API_SECRET in the environment"
            )

        self._reject_if_full()

        room = (room or "").strip() or _new_room()
        identity = (identity or "").strip() or _new_identity()
        # The display name (`name` claim) is cosmetic; fall back to the participant id.
        display_name = (name or "").strip() or identity

        user = (user or "").strip()
        if user and not _USER_ID_RE.fullmatch(user):
            raise BadRequest("invalid user id; use letters, digits, '.', '-', '_' or '@'")

        persona_id = (persona or "").strip() or self._default_persona
        if persona_id and persona_id not in self._registry:
            self._registry.reload()  # tolerate a persona added since startup
        # A custom persona isn't in the curated registry — accept it when this user owns one.
        if (
            persona_id
            and persona_id not in self._registry
            and not self._is_user_persona(user, persona_id)
        ):
            known = ", ".join(self._registry.ids()) or "(none)"
            raise BadRequest(f"unknown persona {persona_id!r}; known: {known}")

        options = self._validate_session_options(voice=voice, cefr=cefr, demeanor=demeanor)

        meta: dict[str, Any] = {}
        if persona_id:
            meta["persona"] = persona_id
        if user:
            meta["user"] = user  # scopes custom-persona resolution + memory in the agent
        meta.update(options.model_dump(exclude_none=True, mode="json"))

        token = mint_access_token(
            api_key=self._config.api_key,
            api_secret=self._config.api_secret,
            identity=identity,
            room=room,
            name=display_name,
            metadata=json.dumps(meta) if meta else None,
            ttl_seconds=self._config.token_ttl,
        )
        return {
            "url": self._config.livekit_url,
            "token": token,
            "room": room,
            "identity": identity,
            "name": display_name,
            "persona": persona_id,
            "voice": options.voice,
            "cefr": options.cefr.value if options.cefr else None,
            "demeanor": options.demeanor.value if options.demeanor else None,
            "user": user or None,
        }

    # -- voice library -----------------------------------------------------------------

    def voices_catalog(self) -> dict[str, Any]:
        """The selectable voice catalog for the active backend (for a client picker)."""
        options = (
            self._voices.catalog(
                self._tts_name,
                supports_cloning=self._supports_cloning,
                protected=self._protected_voices,
            )
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
        self, *, audio: bytes, name: str, authorized: bool = False
    ) -> dict[str, Any]:
        """Clone a voice from an uploaded sample and add it to the library.

        Gated like the rest of the server (auth at the HTTP layer). Requires a cloning backend
        and an explicit `authorized` acknowledgement (the caller affirms they may use the
        voice). Validates the sample, enforces the per-deployment quota, then runs the cascade
        cloner (which persists the clone). Returns the new voice's catalog entry.
        """
        if not self._supports_cloning or self._cloner_factory is None:
            raise BadRequest(
                f"the active TTS backend {self._tts_name or '(unknown)'!r} can't clone voices; "
                "switch to a cloning backend (chatterbox on CUDA)"
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
            await cloner.clone(audio, name)
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
        if name in self._protected_voices:
            raise BadRequest(f"{name!r} is a bundled voice and can't be removed")
        if not store.remove(name):
            raise BadRequest(f"unknown clone {name!r}")
        return {"deleted": name}

    # -- custom personas (multi-user) --------------------------------------------------

    def _require_user(self, user: str | None) -> str:
        """Validate the scoping user id (required for the persona CRUD routes)."""
        user = (user or "").strip()
        if not user:
            raise BadRequest("a user id is required (pass ?user=<id>)")
        if not _USER_ID_RE.fullmatch(user):
            raise BadRequest("invalid user id; use letters, digits, '.', '-', '_' or '@'")
        return user

    def _require_user_store(self) -> UserPersonaStore:
        if self._user_personas is None:
            raise ServerMisconfigured("custom personas are not enabled")
        return self._user_personas

    def _curated(self) -> Persona | None:
        """The default curated persona (for sensible draft defaults), or the first, or None."""
        if self._default_persona in self._registry:
            return self._registry.get(self._default_persona)
        ids = self._registry.ids()
        return self._registry.get(ids[0]) if ids else None

    def _unique_persona_id(self, name: str, user: str, store: UserPersonaStore) -> str:
        """A slug id from `name`, unique within the user's set and never a curated id."""
        base = _slugify(name)
        curated = set(self._registry.ids())
        existing = set(store.ids_for(user))
        candidate, n = base, 2
        while candidate in curated or candidate in existing:
            candidate, n = f"{base}-{n}", n + 1
        return candidate

    def _persona_from_draft(self, draft: Any, *, persona_id: str) -> Persona:
        """Validate a client draft into a `Persona`, filling sensible defaults.

        The server assigns the id (a client-sent `id`/`user` is ignored). A minimal draft
        (`name` + `system_prompt`) is enough: the voice ref and LLM base model default to the
        curated default persona's. Unknown top-level fields are rejected (`Persona` forbids
        extras); a set `llm.lora` is checked against the served adapters on a LoRA backend.
        """
        if not isinstance(draft, dict):
            raise BadRequest("persona draft must be a JSON object")
        data = {k: v for k, v in draft.items() if k not in ("id", "user")}
        data["id"] = persona_id

        curated = self._curated()
        voice = dict(data.get("voice") or {}) if isinstance(data.get("voice"), dict) else {}
        if not voice.get("ref"):
            voice["ref"] = curated.voice.ref if curated else "voices/companion_soft"
        data["voice"] = voice
        llm = dict(data.get("llm") or {}) if isinstance(data.get("llm"), dict) else {}
        if not llm.get("base_model"):
            llm["base_model"] = curated.llm.base_model if curated else "qwen2.5-7b-instruct"
        data["llm"] = llm

        lora = llm.get("lora")
        if lora and self._supports_lora:
            served = {
                o.id for o in served_loras(supports_lora=True, lora_modules=self._lora_modules)
            }
            if Path(lora).name not in served:
                known = ", ".join(sorted(served)) or "(none)"
                raise BadRequest(f"unknown lora {lora!r}; served: {known}")

        try:
            return Persona.model_validate(data)
        except ValidationError as exc:
            errors = exc.errors()
            if errors:
                loc = ".".join(str(p) for p in errors[0].get("loc", ())) or "persona"
                raise BadRequest(f"invalid persona draft: {loc}: {errors[0].get('msg')}") from exc
            raise BadRequest(f"invalid persona draft: {exc}") from exc

    def _stored_persona_payload(self, persona: Persona) -> dict[str, Any]:
        """Create/update response: the picker summary plus the full stored persona for editing."""
        return {
            **self._persona_summary(persona, custom=True),
            "persona": persona.model_dump(mode="json"),
        }

    def create_persona(self, user: str | None, draft: Any) -> dict[str, Any]:
        """Create a custom persona for `user` from a draft, returning the stored persona."""
        user = self._require_user(user)
        store = self._require_user_store()
        store.reload()
        name = draft.get("name") if isinstance(draft, dict) else None
        if not isinstance(name, str) or not name.strip():
            raise BadRequest("a persona needs a non-empty name")
        if len(store.ids_for(user)) >= self._max_user_personas:
            raise BadRequest(f"persona quota reached ({self._max_user_personas}); delete one first")
        persona_id = self._unique_persona_id(name, user, store)
        persona = self._persona_from_draft(draft, persona_id=persona_id)
        store.record(user, persona)
        return self._stored_persona_payload(persona)

    def update_persona(self, user: str | None, persona_id: str, draft: Any) -> dict[str, Any]:
        """Replace one of `user`'s own personas. Curated personas are never editable."""
        user = self._require_user(user)
        store = self._require_user_store()
        store.reload()
        persona_id = (persona_id or "").strip()
        if persona_id in self._registry:
            raise BadRequest(f"{persona_id!r} is a built-in persona and can't be edited")
        if not store.has(user, persona_id):
            raise BadRequest(f"unknown persona {persona_id!r}")
        persona = self._persona_from_draft(draft, persona_id=persona_id)
        store.record(user, persona)
        return self._stored_persona_payload(persona)

    def delete_persona(self, user: str | None, persona_id: str) -> dict[str, Any]:
        """Delete one of `user`'s own personas. Curated personas are never deletable."""
        user = self._require_user(user)
        store = self._require_user_store()
        store.reload()
        persona_id = (persona_id or "").strip()
        if persona_id in self._registry:
            raise BadRequest(f"{persona_id!r} is a built-in persona and can't be deleted")
        if not store.remove(user, persona_id):
            raise BadRequest(f"unknown persona {persona_id!r}")
        return {"deleted": persona_id}

    def draft_persona(self, user: str | None, description: str | None) -> dict[str, Any]:
        """Draft a custom persona for `user` from a plain-English description — *without* persisting.

        Returns the same shape as create/update (`_stored_persona_payload`: a picker summary plus
        the full persona body) so the client drops it straight into the New/Edit form; the user
        tweaks and confirms, and the existing `POST /personas` does the write. The legal `voice.ref`
        ids and served LoRA names are passed to the drafter so a draft can't invent an unservable
        voice/LoRA, and the id is de-duped against the curated + the user's own personas.

        Requires the optional `draft_llm_factory` (the cascade LLM, built lazily). Drafting is
        one-shot and not latency-sensitive, so it runs in this request's worker thread.
        """
        user = self._require_user(user)
        store = self._require_user_store()
        if self._draft_llm_factory is None:
            raise ServerMisconfigured("persona drafting is not enabled")
        description = (description or "").strip()
        if not description:
            raise BadRequest("a persona description is required")
        store.reload()
        existing = set(self._registry.ids()) | set(store.ids_for(user))
        voices = [f"voices/{vid}" for vid in self._voices.ids()] if self._voices is not None else []
        loras = [
            o.id
            for o in served_loras(
                supports_lora=self._supports_lora, lora_modules=self._lora_modules
            )
        ]
        curated = self._curated()
        llm = self._draft_llm_factory()
        try:
            persona = asyncio.run(
                author.draft_persona(
                    llm,
                    description,
                    voices=voices,
                    loras=loras,
                    existing_ids=existing,
                    default_voice=curated.voice.ref if curated else None,
                    default_base_model=curated.llm.base_model if curated else None,
                )
            )
        except PersonaDraftError as exc:
            raise BadRequest(f"could not draft a persona: {exc}") from exc
        return self._stored_persona_payload(persona)

    # -- consent (per-user recording opt-in) -------------------------------------------

    def _require_memory(self) -> MemoryStore:
        if self._memory is None:
            raise ServerMisconfigured("conversation memory is not enabled")
        return self._memory

    def _consent_payload(self, user: str) -> dict[str, Any]:
        c = self._require_memory().get_consent(user)
        return {
            "user": user,
            "granted": c.granted,
            "allow_training": c.allow_training,
            "updated_at": c.updated_at,
        }

    def get_consent(self, user: str | None) -> dict[str, Any]:
        """Return a user's current recording-consent state (to render the client toggle).

        A user who never set consent reads back an ungranted default — the same thing the
        cascade sees when it refuses to record.
        """
        return self._consent_payload(self._require_user(user))

    def set_consent(
        self, user: str | None, *, granted: bool, allow_training: bool = False
    ) -> dict[str, Any]:
        """Grant or withdraw a user's recording consent from the client.

        This writes the same `consent.json` as `personavoice-memory --grant/--revoke`: it's the
        gate the cascade checks before storing any turn. Withdrawing consent keeps already-stored
        data (use the memory CLI's `--delete` to erase it); `allow_training` is the stricter,
        separate opt-in for letting transcripts feed a training distill, and is forced off when
        consent itself is withdrawn (you can't train on what you can't store).

        Trust note: like every `?user=` route here, the user id is client-asserted and only as
        strong as the shared API token — a caller can set consent for any id. That's fine for the
        single-tenant / self-hosted shape; a multi-tenant deployment should derive `user` from the
        authenticated token identity instead of trusting the query param.
        """
        user = self._require_user(user)
        self._require_memory().set_consent(
            user,
            granted=granted,
            allow_training=granted and allow_training,
            note="set via client",
        )
        return self._consent_payload(user)

    # -- LoRA catalog ------------------------------------------------------------------

    def loras(self) -> dict[str, Any]:
        """The selectable served LoRA adapters for a custom persona's `llm.lora`.

        Empty on a non-LoRA backend (Mac / LM Studio), where a LoRA is merged into the base
        model at train time rather than hot-swapped — the `reason` says so.
        """
        options = served_loras(supports_lora=self._supports_lora, lora_modules=self._lora_modules)
        reason = (
            None
            if self._supports_lora
            else "the active LLM backend doesn't hot-swap LoRA; on Mac a LoRA is merged "
            "into the base model at train time"
        )
        return {
            "loras": [o.model_dump() for o in options],
            "backend": self._backend,
            "llm": self._llm_name,
            "supports_lora": self._supports_lora,
            "reason": reason,
        }


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


def _active_llm(settings: Settings) -> tuple[str, bool]:
    """Resolve the active LLM adapter name + whether it can hot-swap LoRA, without loading it.

    Mirrors `_active_tts`: reads the backend YAML and the adapter table's `supports_lora` class
    attribute (True only for vLLM). Tolerates a missing/invalid backend config so /loras answers.
    """
    from ..adapters.factory import LLM_ADAPTERS
    from .config import ConfigError, load_backend_config

    try:
        config = load_backend_config(settings)
    except ConfigError:
        return "", False
    llm_name = config.llm.adapter
    cls = LLM_ADAPTERS.get(llm_name)
    return llm_name, bool(getattr(cls, "supports_lora", False))


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
    # Multi-user custom personas + LoRA picker capability (N3).
    user_personas = load_user_persona_store(settings)
    # Per-user memory store — backs the /consent routes so a client can set its own opt-in
    # (encrypted at rest if PERSONAVOICE_MEMORY_KEY is set; tolerates a missing dir).
    memory = load_memory_store(settings)
    llm_name, supports_lora = _active_llm(settings)
    # What vLLM was launched serving (`--lora-modules`). PERSONAVOICE_LORA_MODULES overrides the
    # docker-compose VLLM_LORA_MODULES var for non-compose deployments.
    lora_modules = (
        os.getenv("PERSONAVOICE_LORA_MODULES") or os.getenv("VLLM_LORA_MODULES") or ""
    ).strip() or None

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

    # Persona authoring helper (Feature K): draft a persona from a description via the cascade LLM.
    # Only wired when custom personas are enabled (the route is user-scoped and writes via the same
    # store). Lazily builds the backend so a draft request, not startup, pays for it.
    draft_llm_factory: DraftLLMFactory | None = None
    if user_personas is not None:
        from ..adapters.factory import build_backend
        from .config import load_backend_config

        def draft_llm_factory() -> Any:  # type: ignore[misc]
            return build_backend(load_backend_config(settings)).llm

    max_clones = int(os.getenv("PERSONAVOICE_MAX_CLONES", str(_DEFAULT_MAX_CLONES)).strip() or 0)
    max_user_personas = int(
        os.getenv("PERSONAVOICE_MAX_USER_PERSONAS", str(_DEFAULT_MAX_USER_PERSONAS)).strip() or 0
    )
    # Optional admission-control early gate (Feature I, step 4): off by default so behavior is
    # unchanged; when on, /token returns 503 once the worker's live gauge reports full.
    admission_gate = _as_bool(os.getenv("PERSONAVOICE_ADMISSION_503"))
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
        # Protect non-preset "inbox" seed clones from deletion. A bundled wav that ships as a
        # voices.yaml preset is excluded — it's already a non-removable preset, not a clone.
        protected_voices=seed_voice_names() - voices.preset_sample_stems(),
        user_personas=user_personas,
        max_user_personas=max_user_personas,
        draft_llm_factory=draft_llm_factory,
        llm_name=llm_name,
        supports_lora=supports_lora,
        lora_modules=lora_modules,
        memory=memory,
        admission_gate=admission_gate,
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

        def _send_json(
            self, status: int, body: dict[str, Any], headers: dict[str, str] | None = None
        ) -> None:
            self._send_raw(
                status, json.dumps(body).encode("utf-8"), "application/json", headers=headers
            )

        def _send_raw(
            self,
            status: int,
            payload: bytes,
            content_type: str,
            headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            # Permissive CORS: the token endpoint is meant to be called from clients.
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
            # Per-response extras, e.g. `Retry-After` on a 503 from the admission gate.
            for name, value in (headers or {}).items():
                self.send_header(name, value)
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
                    # Admission-control load (Feature I). `sessions_max` is static config, so read
                    # it directly (authoritative, always present) rather than round-tripping it
                    # through a live gauge. `sessions_active` is the genuinely live value, read
                    # from the gauge the worker publishes over the shared metrics channel — omitted
                    # when the exporter is off or no worker has reported yet.
                    body: dict[str, Any] = {
                        "status": "ok",
                        "backend": service._backend,
                        "sessions_max": max_sessions(),
                    }
                    sessions = read_sessions()
                    if sessions is not None:
                        body["sessions_active"] = sessions[0]
                    self._send_json(200, body)
                    return
                # Prometheus scrape endpoint. Like /healthz it's unauthenticated and not
                # rate-limited (scrapers don't send a bearer token and poll on a fixed interval);
                # it exposes only aggregate operational counters, no secrets. Empty body when
                # prometheus_client isn't installed.
                if path == "/metrics" and method == "GET":
                    payload, content_type = render_metrics()
                    self._send_raw(200, payload, content_type)
                    return
                # Rate-limit everything else (before auth, to throttle unauthenticated floods)
                # keyed by client IP.
                if not rl.allow(self.client_address[0]):
                    raise TooManyRequests("rate limit exceeded")
                # Everything below is protected by the optional bearer token.
                service.check_auth(self.headers.get("Authorization"))
                if path == "/personas" and method == "GET":
                    self._send_json(200, service.personas(user=query.get("user")))
                elif path == "/personas" and method == "POST":
                    body = self._read_json_body()
                    self._send_json(201, service.create_persona(query.get("user"), body))
                elif path == "/personas/draft" and method == "POST":
                    body = self._read_json_body()
                    self._send_json(
                        200, service.draft_persona(query.get("user"), body.get("description"))
                    )
                elif path.startswith("/personas/") and method == "GET":
                    pid = path[len("/personas/") :]
                    self._send_json(200, service.get_persona(query.get("user"), pid))
                elif path.startswith("/personas/") and method == "PUT":
                    pid = path[len("/personas/") :]
                    body = self._read_json_body()
                    self._send_json(200, service.update_persona(query.get("user"), pid, body))
                elif path.startswith("/personas/") and method == "DELETE":
                    pid = path[len("/personas/") :]
                    self._send_json(200, service.delete_persona(query.get("user"), pid))
                elif path == "/loras" and method == "GET":
                    self._send_json(200, service.loras())
                elif path == "/voices" and method == "GET":
                    self._send_json(200, service.voices_catalog())
                elif path == "/consent" and method == "GET":
                    self._send_json(200, service.get_consent(query.get("user")))
                elif path == "/consent" and method == "POST":
                    body = self._read_json_body()
                    granted = body.get("granted")
                    if not isinstance(granted, bool):
                        raise BadRequest("'granted' must be a boolean (true to opt in, false out)")
                    self._send_json(
                        200,
                        service.set_consent(
                            query.get("user") or body.get("user"),
                            granted=granted,
                            allow_training=bool(body.get("allow_training", False)),
                        ),
                    )
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
                        name=body.get("name") or query.get("name"),
                        persona=body.get("persona") or query.get("persona"),
                        voice=body.get("voice") or query.get("voice"),
                        cefr=body.get("cefr") or query.get("cefr"),
                        demeanor=body.get("demeanor") or query.get("demeanor"),
                        user=body.get("user") or query.get("user"),
                    )
                    self._send_json(200, result)
                else:
                    self._send_json(404, {"error": f"no route for {method} {path}"})
            except TokenServiceError as exc:
                self._send_json(exc.status, {"error": str(exc), **exc.extra}, headers=exc.headers)
            except Exception as exc:  # pragma: no cover - defensive 500
                logger.exception("token server error")
                self._send_json(500, {"error": f"internal error: {exc}"})

        def _handle_clone(self, query: dict[str, str]) -> None:
            """POST /voices/clone: raw wav body + `name`/`authorized` query params.

            The sample rides as the raw request body (Content-Type audio/wav) rather than
            multipart — zero-dependency and stdlib-only (Python 3.13 dropped `cgi`). Cloning
            loads models, so it runs in this request's worker thread via `asyncio.run`.
            """
            audio = self._read_raw_body(max_clone_bytes)
            result = asyncio.run(
                service.enroll_voice(
                    audio=audio,
                    name=query.get("name", ""),
                    authorized=_as_bool(query.get("authorized")),
                )
            )
            self._send_json(201, result)

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def do_PUT(self) -> None:
            self._dispatch("PUT")

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
    gate = "on (503 when full)" if service._admission_gate else "off"
    print(
        f"Token server listening on {scheme}://{host}:{port}  "
        f"(auth: {auth}, rate-limit: {rl}, admission gate: {gate})"
    )
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

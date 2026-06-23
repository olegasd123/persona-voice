"""The HTTP token server: `TokenService` logic + a real-socket integration smoke test."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from personavoice.models import VoiceDef, VoiceRef
from personavoice.persona.registry import PersonaRegistry
from personavoice.server.config import Settings
from personavoice.server.ratelimit import RateLimiter
from personavoice.server.token_server import (
    BadRequest,
    ServerMisconfigured,
    TokenService,
    TokenServiceConfig,
    Unauthorized,
    build_service,
    make_server,
)
from personavoice.server.tokens import decode_token
from personavoice.voice.clone import ClonedVoice, ClonesStore
from personavoice.voice.registry import VoiceRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]
PERSONAS_DIR = REPO_ROOT / "config" / "personas"

SECRET = "test-secret"


@pytest.fixture
def registry() -> PersonaRegistry:
    return PersonaRegistry(PERSONAS_DIR)


def make_service(
    registry: PersonaRegistry,
    *,
    api_token: str | None = None,
    configured: bool = True,
    default: str = "companion",
    voices: VoiceRegistry | None = None,
) -> TokenService:
    config = TokenServiceConfig(
        livekit_url="wss://livekit.example:7880" if configured else "",
        api_key="APIkey" if configured else "",
        api_secret=SECRET if configured else "",
        api_token=api_token,
    )
    return TokenService(config, registry, default_persona=default, backend="mac", voices=voices)


class _FakeCloner:
    """Records a clone into the real store (so catalog/delete work) without loading models."""

    def __init__(self, store: ClonesStore) -> None:
        self._store = store

    async def clone(self, audio: bytes, name: str, *, ref_text: str | None = None) -> VoiceRef:
        self._store.record(
            ClonedVoice(name=name, sample_path=f"{name}.wav", ref_text=ref_text or "hi")
        )
        return VoiceRef(id=name, name=name, sample_path=f"{name}.wav")


def make_cloning_service(
    registry: PersonaRegistry,
    tmp_path: Path,
    *,
    max_clones: int = 50,
    api_token: str | None = None,
) -> tuple[TokenService, ClonesStore]:
    """A TokenService on a cloning backend with a fake cloner and a real clones store."""
    store = ClonesStore(tmp_path / "clones")
    voices = VoiceRegistry(
        {"companion_soft": VoiceDef(presets={"kokoro": "af_heart"})}, clones=store
    )
    config = TokenServiceConfig(
        livekit_url="wss://livekit.example:7880",
        api_key="APIkey",
        api_secret=SECRET,
        api_token=api_token,
    )
    svc = TokenService(
        config,
        registry,
        default_persona="companion",
        backend="mac",
        voices=voices,
        tts_name="f5_mlx",
        supports_cloning=True,
        cloner_factory=lambda: _FakeCloner(store),
        max_clones=max_clones,
    )
    return svc, store


# --- TokenService.issue ---------------------------------------------------------------


def test_issue_generates_room_and_identity(registry: PersonaRegistry) -> None:
    svc = make_service(registry)
    result = svc.issue()
    assert result["url"] == "wss://livekit.example:7880"
    assert result["room"].startswith("pv-")
    assert result["identity"].startswith("user-")
    assert result["persona"] == "companion"


def test_issued_token_is_valid_and_scoped(registry: PersonaRegistry) -> None:
    svc = make_service(registry)
    result = svc.issue(room="my-room", identity="alice", persona="hr_interviewer")
    claims = decode_token(result["token"], SECRET)
    assert claims["sub"] == "alice"
    assert claims["video"]["room"] == "my-room"
    assert claims["video"]["roomJoin"] is True
    # Persona is embedded in metadata for observability and echoed back to the client.
    assert json.loads(claims["metadata"]) == {"persona": "hr_interviewer"}
    assert result["persona"] == "hr_interviewer"


def test_issue_rejects_unknown_persona(registry: PersonaRegistry) -> None:
    svc = make_service(registry)
    with pytest.raises(BadRequest, match="unknown persona"):
        svc.issue(persona="nope")


def test_issue_requires_livekit_credentials(registry: PersonaRegistry) -> None:
    svc = make_service(registry, configured=False)
    with pytest.raises(ServerMisconfigured, match="LiveKit is not configured"):
        svc.issue()


def test_blank_fields_fall_back_to_defaults(registry: PersonaRegistry) -> None:
    svc = make_service(registry)
    result = svc.issue(room="  ", identity="", persona=None)
    assert result["room"].startswith("pv-")
    assert result["identity"].startswith("user-")
    assert result["persona"] == "companion"


# --- issue: session options (voice / cefr / demeanor) ---------------------------------


def test_issue_embeds_and_echoes_session_options(registry: PersonaRegistry) -> None:
    voices = VoiceRegistry({"companion_soft": VoiceDef(presets={"kokoro": "af_heart"})})
    svc = make_service(registry, voices=voices)
    result = svc.issue(persona="companion", voice="companion_soft", cefr="b1", demeanor="kind")
    meta = json.loads(decode_token(result["token"], SECRET)["metadata"])
    assert meta == {
        "persona": "companion",
        "voice": "companion_soft",
        "cefr": "B1",  # canonicalized
        "demeanor": "kind",
    }
    assert (result["voice"], result["cefr"], result["demeanor"]) == ("companion_soft", "B1", "kind")


def test_issue_without_options_keeps_clean_metadata(registry: PersonaRegistry) -> None:
    svc = make_service(registry)
    result = svc.issue(persona="companion")
    assert json.loads(decode_token(result["token"], SECRET)["metadata"]) == {"persona": "companion"}
    assert result["voice"] is None and result["cefr"] is None and result["demeanor"] is None


def test_issue_rejects_unknown_voice(registry: PersonaRegistry) -> None:
    voices = VoiceRegistry({"companion_soft": VoiceDef(presets={"kokoro": "af_heart"})})
    svc = make_service(registry, voices=voices)
    with pytest.raises(BadRequest, match="unknown voice"):
        svc.issue(voice="ghost")


def test_issue_skips_voice_check_without_registry(registry: PersonaRegistry) -> None:
    # No registry to validate against -> any voice string is accepted and echoed.
    svc = make_service(registry, voices=None)
    assert svc.issue(voice="whatever")["voice"] == "whatever"


def test_issue_rejects_unknown_cefr(registry: PersonaRegistry) -> None:
    svc = make_service(registry)
    with pytest.raises(BadRequest, match="cefr"):
        svc.issue(cefr="Z9")


def test_issue_rejects_unknown_demeanor(registry: PersonaRegistry) -> None:
    svc = make_service(registry)
    with pytest.raises(BadRequest, match="demeanor"):
        svc.issue(demeanor="grumpy")


# --- voice catalog + enrollment -------------------------------------------------------


def test_voices_catalog_reports_backend_capability(
    registry: PersonaRegistry, tmp_path: Path
) -> None:
    svc, _ = make_cloning_service(registry, tmp_path)
    catalog = svc.voices_catalog()
    assert catalog["supports_cloning"] is True
    assert catalog["tts"] == "f5_mlx"
    assert isinstance(catalog["voices"], list)


def test_voices_catalog_empty_without_registry(registry: PersonaRegistry) -> None:
    assert make_service(registry).voices_catalog()["voices"] == []


async def test_enroll_voice_happy_path(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, store = make_cloning_service(registry, tmp_path)
    result = await svc.enroll_voice(audio=b"RIFFsample", name="my_voice", authorized=True)
    assert result["id"] == "my_voice" and result["kind"] == "clone"
    assert "my_voice" in store  # persisted
    assert any(o["id"] == "my_voice" for o in svc.voices_catalog()["voices"])  # now selectable


async def test_enroll_requires_authorization(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_cloning_service(registry, tmp_path)
    with pytest.raises(BadRequest, match="authorized"):
        await svc.enroll_voice(audio=b"x", name="v", authorized=False)


async def test_enroll_rejected_on_non_cloning_backend(registry: PersonaRegistry) -> None:
    svc = make_service(registry)  # supports_cloning=False, no cloner factory
    with pytest.raises(BadRequest, match="can't clone"):
        await svc.enroll_voice(audio=b"x", name="v", authorized=True)


async def test_enroll_rejects_invalid_name(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_cloning_service(registry, tmp_path)
    with pytest.raises(BadRequest, match="invalid voice name"):
        await svc.enroll_voice(audio=b"x", name="bad name!", authorized=True)


async def test_enroll_enforces_quota(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_cloning_service(registry, tmp_path, max_clones=1)
    await svc.enroll_voice(audio=b"x", name="one", authorized=True)
    with pytest.raises(BadRequest, match="quota"):
        await svc.enroll_voice(audio=b"x", name="two", authorized=True)


async def test_delete_voice_round_trip(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, store = make_cloning_service(registry, tmp_path)
    await svc.enroll_voice(audio=b"x", name="temp", authorized=True)
    assert svc.delete_voice("temp") == {"deleted": "temp"}
    assert "temp" not in store


def test_delete_unknown_voice(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_cloning_service(registry, tmp_path)
    with pytest.raises(BadRequest, match="unknown clone"):
        svc.delete_voice("ghost")


# --- auth -----------------------------------------------------------------------------


def test_auth_is_open_when_no_token_configured(registry: PersonaRegistry) -> None:
    svc = make_service(registry, api_token=None)
    svc.check_auth(None)  # no raise


def test_auth_enforced_when_token_configured(registry: PersonaRegistry) -> None:
    svc = make_service(registry, api_token="sekret")
    svc.check_auth("Bearer sekret")  # ok
    with pytest.raises(Unauthorized, match="missing or malformed"):
        svc.check_auth(None)
    with pytest.raises(Unauthorized, match="missing or malformed"):
        svc.check_auth("sekret")  # no Bearer prefix
    with pytest.raises(Unauthorized, match="invalid API token"):
        svc.check_auth("Bearer wrong")


# --- personas -------------------------------------------------------------------------


def test_personas_lists_ids_and_default(registry: PersonaRegistry) -> None:
    svc = make_service(registry, default="companion")
    data = svc.personas()
    ids = {p["id"] for p in data["personas"]}
    assert {"companion", "hr_interviewer"} <= ids
    assert all({"name", "description", "voice"} <= p.keys() for p in data["personas"])
    assert data["default"] == "companion"


def test_personas_includes_description_and_voice_blurb(registry: PersonaRegistry) -> None:
    voices = VoiceRegistry.load(
        Settings(
            backend="mac", config_dir=REPO_ROOT / "config", models_dir=REPO_ROOT / "models"
        ).voices_path
    )
    svc = make_service(registry, default="companion", voices=voices)
    companion = next(p for p in svc.personas()["personas"] if p["id"] == "companion")
    # description comes from the persona file; voice blurb resolves via config/voices.yaml.
    assert companion["description"]
    assert companion["voice"] == "warm, soft, feminine"


def test_personas_voice_blank_without_registry(registry: PersonaRegistry) -> None:
    # No voice registry injected (the default) -> voice blurb is "", not an error.
    svc = make_service(registry, default="companion")
    assert all(p["voice"] == "" for p in svc.personas()["personas"])


# --- config + build_service -----------------------------------------------------------


def test_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVEKIT_URL", "wss://lk:7880")
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")
    monkeypatch.setenv("PERSONAVOICE_API_TOKEN", "bearer-tok")
    monkeypatch.setenv("PERSONAVOICE_TOKEN_TTL", "120")
    cfg = TokenServiceConfig.from_env()
    assert cfg.livekit_url == "wss://lk:7880"
    assert cfg.api_key == "k"
    assert cfg.api_secret == "s"
    assert cfg.api_token == "bearer-tok"
    assert cfg.token_ttl == 120


def test_config_from_env_blank_api_token_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERSONAVOICE_API_TOKEN", "   ")
    monkeypatch.delenv("PERSONAVOICE_TOKEN_TTL", raising=False)
    cfg = TokenServiceConfig.from_env()
    assert cfg.api_token is None
    assert cfg.token_ttl == 3600


def test_config_from_env_bad_ttl_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERSONAVOICE_TOKEN_TTL", "soon")
    with pytest.raises(ServerMisconfigured, match="TOKEN_TTL"):
        TokenServiceConfig.from_env()


def test_build_service_defaults_to_first_persona_when_env_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAVOICE_PERSONA", "does-not-exist")
    settings = Settings(
        backend="mac", config_dir=REPO_ROOT / "config", models_dir=REPO_ROOT / "models"
    )
    svc = build_service(settings)
    # Falls back to a real, registered persona rather than the bogus env value.
    assert svc.personas()["default"] in {p["id"] for p in svc.personas()["personas"]}


# --- HTTP integration (real socket on an ephemeral port) ------------------------------


@pytest.fixture
def live_server(registry: PersonaRegistry) -> Iterator[tuple[str, TokenService]]:
    svc = make_service(registry, api_token="sekret")
    httpd = make_server(svc, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    try:
        yield f"http://{host}:{port}", svc
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def _get(url: str, headers: dict[str, str] | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _post(url: str, body: dict, headers: dict[str, str] | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _post_raw(url: str, data: bytes, headers: dict[str, str] | None = None) -> tuple[int, dict]:
    hdrs = {"Content-Type": "application/octet-stream", **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _delete(url: str, headers: dict[str, str] | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(url, headers=headers or {}, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_http_healthz_is_open(live_server: tuple[str, TokenService]) -> None:
    base, _ = live_server
    status, body = _get(f"{base}/healthz")
    assert status == 200
    assert body["status"] == "ok"


def test_http_token_requires_auth(live_server: tuple[str, TokenService]) -> None:
    base, _ = live_server
    status, body = _post(f"{base}/token", {})
    assert status == 401
    assert "error" in body


def test_http_token_happy_path(live_server: tuple[str, TokenService]) -> None:
    base, _ = live_server
    auth = {"Authorization": "Bearer sekret"}
    status, body = _post(f"{base}/token", {"persona": "hr_interviewer"}, auth)
    assert status == 200
    claims = decode_token(body["token"], SECRET)
    assert claims["video"]["room"] == body["room"]
    assert body["persona"] == "hr_interviewer"


def test_http_token_get_with_query(live_server: tuple[str, TokenService]) -> None:
    base, _ = live_server
    auth = {"Authorization": "Bearer sekret"}
    status, body = _get(f"{base}/token?room=q-room&identity=bob", auth)
    assert status == 200
    assert body["room"] == "q-room"
    assert body["identity"] == "bob"


def test_http_personas(live_server: tuple[str, TokenService]) -> None:
    base, _ = live_server
    status, body = _get(f"{base}/personas", {"Authorization": "Bearer sekret"})
    assert status == 200
    assert any(p["id"] == "companion" for p in body["personas"])


def test_http_unknown_route_404(live_server: tuple[str, TokenService]) -> None:
    base, _ = live_server
    status, _ = _get(f"{base}/nope", {"Authorization": "Bearer sekret"})
    assert status == 404


def test_http_unknown_persona_400(live_server: tuple[str, TokenService]) -> None:
    base, _ = live_server
    auth = {"Authorization": "Bearer sekret"}
    status, body = _post(f"{base}/token", {"persona": "ghost"}, auth)
    assert status == 400
    assert "unknown persona" in body["error"]


# --- HTTP: voice library routes -------------------------------------------------------


@pytest.fixture
def cloning_server(
    registry: PersonaRegistry, tmp_path: Path
) -> Iterator[tuple[str, TokenService, ClonesStore]]:
    svc, store = make_cloning_service(registry, tmp_path, api_token="sekret")
    httpd = make_server(svc, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    try:
        yield f"http://{host}:{port}", svc, store
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_http_voices_catalog(cloning_server: tuple[str, TokenService, ClonesStore]) -> None:
    base, _, _ = cloning_server
    status, body = _get(f"{base}/voices", {"Authorization": "Bearer sekret"})
    assert status == 200
    assert body["supports_cloning"] is True


def test_http_voices_requires_auth(cloning_server: tuple[str, TokenService, ClonesStore]) -> None:
    base, _, _ = cloning_server
    status, _ = _get(f"{base}/voices")
    assert status == 401


def test_http_clone_and_delete_round_trip(
    cloning_server: tuple[str, TokenService, ClonesStore],
) -> None:
    base, _, store = cloning_server
    auth = {"Authorization": "Bearer sekret"}
    status, body = _post_raw(
        f"{base}/voices/clone?name=httpvoice&authorized=1", b"RIFFsample", auth
    )
    assert status == 201
    assert body["id"] == "httpvoice"
    assert "httpvoice" in store

    status, body = _delete(f"{base}/voices/clone/httpvoice", auth)
    assert status == 200
    assert body["deleted"] == "httpvoice"
    assert "httpvoice" not in store


def test_http_clone_unauthorized_ack_400(
    cloning_server: tuple[str, TokenService, ClonesStore],
) -> None:
    base, _, _ = cloning_server
    auth = {"Authorization": "Bearer sekret"}
    status, body = _post_raw(f"{base}/voices/clone?name=v", b"RIFFsample", auth)
    assert status == 400
    assert "authorized" in body["error"]


def test_http_clone_oversize_413(
    registry: PersonaRegistry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PERSONAVOICE_MAX_CLONE_BYTES", "8")  # tiny cap so any sample is too big
    svc, _ = make_cloning_service(registry, tmp_path, api_token="sekret")
    httpd = make_server(svc, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    base = f"http://{host}:{port}"
    auth = {"Authorization": "Bearer sekret"}
    try:
        status, body = _post_raw(
            f"{base}/voices/clone?name=big&authorized=1", b"way-too-large-sample", auth
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)
    assert status == 413
    assert "too large" in body["error"]


# --- hardening config + rate limiting --------------------------------------------


def test_config_from_env_security_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERSONAVOICE_REQUIRE_AUTH", "1")
    monkeypatch.setenv("PERSONAVOICE_TLS_CERT", "/etc/certs/server.crt")
    monkeypatch.setenv("PERSONAVOICE_TLS_KEY", "/etc/certs/server.key")
    cfg = TokenServiceConfig.from_env()
    assert cfg.require_auth is True
    assert cfg.tls_enabled is True


def test_config_tls_disabled_without_both_cert_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERSONAVOICE_TLS_CERT", "/etc/certs/server.crt")
    monkeypatch.delenv("PERSONAVOICE_TLS_KEY", raising=False)
    assert TokenServiceConfig.from_env().tls_enabled is False


def test_http_rate_limit_returns_429(registry: PersonaRegistry) -> None:
    svc = make_service(registry, api_token="sekret")
    # burst=1, ~no refill within the test window → the second request is throttled.
    limiter = RateLimiter(rate=0.001, burst=1.0)
    httpd = make_server(svc, "127.0.0.1", 0, limiter=limiter)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    base = f"http://{host}:{port}"
    auth = {"Authorization": "Bearer sekret"}
    try:
        first, _ = _post(f"{base}/token", {}, auth)
        second, body = _post(f"{base}/token", {}, auth)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)
    assert first == 200
    assert second == 429
    assert "rate limit" in body["error"]


def test_http_healthz_not_rate_limited(registry: PersonaRegistry) -> None:
    svc = make_service(registry, api_token="sekret")
    limiter = RateLimiter(rate=0.001, burst=1.0)
    httpd = make_server(svc, "127.0.0.1", 0, limiter=limiter)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    base = f"http://{host}:{port}"
    try:
        # Health checks must never be throttled (used by orchestration probes).
        codes = [_get(f"{base}/healthz")[0] for _ in range(3)]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)
    assert codes == [200, 200, 200]

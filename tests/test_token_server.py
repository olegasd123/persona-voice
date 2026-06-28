"""The HTTP token server: `TokenService` logic + a real-socket integration smoke test."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from personavoice.memory import MemoryStore
from personavoice.models import Persona, VoiceDef, VoiceRef
from personavoice.persona.registry import PersonaRegistry
from personavoice.persona.store import UserPersonaStore
from personavoice.server.config import Settings
from personavoice.server.ratelimit import RateLimiter
from personavoice.server.token_server import (
    BadRequest,
    ServerMisconfigured,
    TokenService,
    TokenServiceConfig,
    Unauthorized,
    Unavailable,
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

    async def clone(self, audio: bytes, name: str) -> VoiceRef:
        self._store.record(ClonedVoice(name=name, sample_path=f"{name}.wav"))
        return VoiceRef(id=name, name=name, sample_path=f"{name}.wav")


def make_cloning_service(
    registry: PersonaRegistry,
    tmp_path: Path,
    *,
    max_clones: int = 50,
    api_token: str | None = None,
    protected_voices: tuple[str, ...] = (),
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
        tts_name="chatterbox",
        supports_cloning=True,
        cloner_factory=lambda: _FakeCloner(store),
        max_clones=max_clones,
        protected_voices=protected_voices,
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
    # No display name given → the `name` claim falls back to the participant id.
    assert claims["name"] == "alice"
    assert result["name"] == "alice"
    assert claims["video"]["room"] == "my-room"
    assert claims["video"]["roomJoin"] is True
    # Persona is embedded in metadata for observability and echoed back to the client.
    assert json.loads(claims["metadata"]) == {"persona": "hr_interviewer"}
    assert result["persona"] == "hr_interviewer"


def test_issue_separates_display_name_from_identity(registry: PersonaRegistry) -> None:
    svc = make_service(registry)
    # A stable, opaque participant id with a separate, free-text display name.
    result = svc.issue(identity="pv-abc123", name="Oleg Smith", persona="companion")
    claims = decode_token(result["token"], SECRET)
    assert claims["sub"] == "pv-abc123"  # unique participant identity
    assert claims["name"] == "Oleg Smith"  # cosmetic, may contain spaces / collide
    assert result["identity"] == "pv-abc123"
    assert result["name"] == "Oleg Smith"


def test_issue_rejects_unknown_persona(registry: PersonaRegistry) -> None:
    svc = make_service(registry)
    with pytest.raises(BadRequest, match="unknown persona"):
        svc.issue(persona="nope")


def test_issue_requires_livekit_credentials(registry: PersonaRegistry) -> None:
    svc = make_service(registry, configured=False)
    with pytest.raises(ServerMisconfigured, match="LiveKit is not configured"):
        svc.issue()


# --- Admission control: token-server 503 early gate (Feature I, step 4) -----------------------


def _gated_service(registry: PersonaRegistry, reader: object, *, gate: bool = True) -> TokenService:
    """A `TokenService` with the optional early gate wired to a fake live-count reader."""
    config = TokenServiceConfig(
        livekit_url="wss://livekit.example:7880", api_key="APIkey", api_secret=SECRET
    )
    return TokenService(
        config,
        registry,
        default_persona="companion",
        backend="mac",
        admission_gate=gate,
        sessions_reader=reader,  # type: ignore[arg-type]
    )


def test_issue_503_when_worker_full(
    registry: PersonaRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Gate on + the worker reports active >= config capacity → refuse to mint with 503 + retry hint.
    monkeypatch.setenv("PERSONAVOICE_MAX_SESSIONS", "1")
    monkeypatch.setenv("PERSONAVOICE_BUSY_RETRY_AFTER", "7")
    svc = _gated_service(registry, lambda: (1, 1))
    with pytest.raises(Unavailable) as exc:
        svc.issue()
    assert exc.value.status == 503
    assert exc.value.headers == {"Retry-After": "7"}
    assert exc.value.extra == {"retry_after": 7}


def test_issue_mints_under_capacity_with_gate(
    registry: PersonaRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PERSONAVOICE_MAX_SESSIONS", "2")
    svc = _gated_service(registry, lambda: (1, 2))  # one slot free
    assert svc.issue()["persona"] == "companion"


def test_issue_fails_open_when_count_unknown(
    registry: PersonaRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The reader returns None (exporter off / no worker reporting) → mint anyway; the worker's load
    # gate is the real rail, this is only an optimization on top.
    monkeypatch.setenv("PERSONAVOICE_MAX_SESSIONS", "1")
    svc = _gated_service(registry, lambda: None)
    assert svc.issue()["persona"] == "companion"


def test_issue_ignores_count_when_gate_off(
    registry: PersonaRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PERSONAVOICE_MAX_SESSIONS", "1")
    svc = _gated_service(registry, lambda: (5, 1), gate=False)  # over cap, but gate disabled
    assert svc.issue()["persona"] == "companion"


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
    assert catalog["tts"] == "chatterbox"
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


async def test_delete_rejects_protected_seed_voice(
    registry: PersonaRegistry, tmp_path: Path
) -> None:
    svc, store = make_cloning_service(registry, tmp_path, protected_voices=("Feminine",))
    await svc.enroll_voice(audio=b"x", name="Feminine", authorized=True)
    with pytest.raises(BadRequest, match="bundled voice"):
        svc.delete_voice("Feminine")
    assert "Feminine" in store  # still there


async def test_catalog_marks_protected_voice_non_removable(
    registry: PersonaRegistry, tmp_path: Path
) -> None:
    svc, _ = make_cloning_service(registry, tmp_path, protected_voices=("Feminine",))
    await svc.enroll_voice(audio=b"x", name="Feminine", authorized=True)  # bundled seed
    await svc.enroll_voice(audio=b"x", name="mine", authorized=True)  # user clone
    by_id = {o["id"]: o for o in svc.voices_catalog()["voices"]}
    assert by_id["Feminine"]["removable"] is False
    assert by_id["mine"]["removable"] is True


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
    assert all(
        {"name", "description", "voice", "cefr", "demeanor"} <= p.keys() for p in data["personas"]
    )
    assert data["default"] == "companion"


def test_personas_session_defaults_none_for_curated(registry: PersonaRegistry) -> None:
    # Curated personas set no session defaults, so the picker reports them as None (not absent).
    svc = make_service(registry, default="companion")
    assert all(p["cefr"] is None and p["demeanor"] is None for p in svc.personas()["personas"])


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


# --- custom personas (multi-user) -----------------------------------------------------


def make_persona_service(
    registry: PersonaRegistry,
    tmp_path: Path,
    *,
    supports_lora: bool = False,
    lora_modules: str | None = None,
    max_user_personas: int = 50,
    api_token: str | None = None,
    draft_llm: object | None = None,
) -> tuple[TokenService, UserPersonaStore]:
    """A TokenService with a real user-persona store + a voice registry for blurbs."""
    store = UserPersonaStore(tmp_path / "user_personas.json")
    voices = VoiceRegistry(
        {"companion_soft": VoiceDef(presets={"kokoro": "af_heart"}, description="warm, soft")}
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
        user_personas=store,
        max_user_personas=max_user_personas,
        draft_llm_factory=(lambda: draft_llm) if draft_llm is not None else None,
        llm_name="vllm" if supports_lora else "lmstudio",
        supports_lora=supports_lora,
        lora_modules=lora_modules,
    )
    return svc, store


class _DraftLLM:
    """An LLM stub for the draft route: `.chat` returns a fixed JSON persona body."""

    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def chat(self, messages: object, persona: object) -> str:
        return self.reply


_DRAFT = {"name": "French Tutor", "system_prompt": "Teach French, gently."}


def test_create_persona_happy_path(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, store = make_persona_service(registry, tmp_path)
    result = svc.create_persona("alice", dict(_DRAFT))
    assert result["custom"] is True
    assert result["id"] == "french-tutor"  # slug of the name
    assert result["persona"]["system_prompt"] == "Teach French, gently."
    # Persisted under the user and now visible in that user's persona list.
    assert store.has("alice", "french-tutor")
    ids = {p["id"] for p in svc.personas(user="alice")["personas"]}
    assert {"companion", "french-tutor"} <= ids


def test_create_persona_fills_defaults_from_curated(
    registry: PersonaRegistry, tmp_path: Path
) -> None:
    svc, _ = make_persona_service(registry, tmp_path)
    # A minimal draft (name + prompt) is enough; voice ref + base model default to the curated.
    result = svc.create_persona("alice", {"name": "Minimal", "system_prompt": "hi"})
    persona = result["persona"]
    assert persona["voice"]["ref"]  # filled
    assert persona["llm"]["base_model"]  # filled


def test_create_persona_with_session_defaults(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_persona_service(registry, tmp_path)
    draft = {**_DRAFT, "session_defaults": {"cefr": "a1", "demeanor": "kind"}}
    result = svc.create_persona("alice", draft)
    assert result["persona"]["session_defaults"]["cefr"] == "A1"  # canonicalized
    # …and the picker summary surfaces those baked-in defaults so the card can show them.
    summary = next(p for p in svc.personas(user="alice")["personas"] if p["id"] == result["id"])
    assert (summary["cefr"], summary["demeanor"]) == ("A1", "kind")


def test_create_persona_requires_user(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_persona_service(registry, tmp_path)
    with pytest.raises(BadRequest, match="user id is required"):
        svc.create_persona(None, dict(_DRAFT))


def test_create_persona_rejects_empty_name(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_persona_service(registry, tmp_path)
    with pytest.raises(BadRequest, match="non-empty name"):
        svc.create_persona("alice", {"system_prompt": "no name"})


def test_create_persona_rejects_extra_field(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_persona_service(registry, tmp_path)
    with pytest.raises(BadRequest, match="invalid persona draft"):
        svc.create_persona("alice", {**_DRAFT, "bogus": 1})


def test_create_persona_id_uniqueness(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_persona_service(registry, tmp_path)
    first = svc.create_persona("alice", dict(_DRAFT))
    second = svc.create_persona("alice", dict(_DRAFT))  # same name → distinct id
    assert first["id"] == "french-tutor" and second["id"] == "french-tutor-2"


def test_create_persona_never_shadows_curated_id(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_persona_service(registry, tmp_path)
    # A name that slugs to a curated id is suffixed so it can't shadow the built-in.
    result = svc.create_persona("alice", {"name": "Companion", "system_prompt": "hi"})
    assert result["id"] != "companion"


def test_create_persona_enforces_quota(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_persona_service(registry, tmp_path, max_user_personas=1)
    svc.create_persona("alice", {"name": "One", "system_prompt": "hi"})
    with pytest.raises(BadRequest, match="quota"):
        svc.create_persona("alice", {"name": "Two", "system_prompt": "hi"})


def test_personas_are_user_scoped(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_persona_service(registry, tmp_path)
    svc.create_persona("alice", dict(_DRAFT))
    # Bob doesn't see Alice's custom persona, only the curated set.
    bob_ids = {p["id"] for p in svc.personas(user="bob")["personas"]}
    assert "french-tutor" not in bob_ids
    assert "companion" in bob_ids
    # And with no user, only curated are listed.
    assert all(not p["custom"] for p in svc.personas()["personas"])


def test_personas_curated_win_on_clash(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, store = make_persona_service(registry, tmp_path)
    # Force a stored persona whose id collides with a curated one (bypassing slug uniqueness).
    store.record(
        "alice",
        Persona.model_validate(
            {
                "id": "companion",
                "name": "Evil Companion",
                "system_prompt": "x",
                "llm": {"base_model": "m"},
                "voice": {"ref": "voices/companion_soft"},
            }
        ),
    )
    companions = [p for p in svc.personas(user="alice")["personas"] if p["id"] == "companion"]
    # Only the curated one survives; the shadowing custom persona is dropped from the list.
    assert len(companions) == 1 and companions[0]["custom"] is False


def test_update_persona_own(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_persona_service(registry, tmp_path)
    created = svc.create_persona("alice", dict(_DRAFT))
    updated = svc.update_persona("alice", created["id"], {**_DRAFT, "name": "Spanish Tutor"})
    assert updated["id"] == created["id"]  # id preserved
    assert updated["name"] == "Spanish Tutor"


def test_update_curated_rejected(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_persona_service(registry, tmp_path)
    with pytest.raises(BadRequest, match="built-in persona"):
        svc.update_persona("alice", "companion", dict(_DRAFT))


def test_update_unknown_rejected(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_persona_service(registry, tmp_path)
    with pytest.raises(BadRequest, match="unknown persona"):
        svc.update_persona("alice", "ghost", dict(_DRAFT))


def test_delete_persona_own(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, store = make_persona_service(registry, tmp_path)
    created = svc.create_persona("alice", dict(_DRAFT))
    assert svc.delete_persona("alice", created["id"]) == {"deleted": created["id"]}
    assert not store.has("alice", created["id"])


def test_delete_curated_rejected(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_persona_service(registry, tmp_path)
    with pytest.raises(BadRequest, match="built-in persona"):
        svc.delete_persona("alice", "companion")


def test_delete_unknown_rejected(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_persona_service(registry, tmp_path)
    with pytest.raises(BadRequest, match="unknown persona"):
        svc.delete_persona("alice", "ghost")


def test_persona_routes_disabled_without_store(registry: PersonaRegistry) -> None:
    svc = make_service(registry)  # no user_personas store wired
    with pytest.raises(ServerMisconfigured, match="custom personas are not enabled"):
        svc.create_persona("alice", dict(_DRAFT))


# --- get one persona (full body, for an edit form) ------------------------------------


def test_get_persona_returns_custom_full_body(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_persona_service(registry, tmp_path)
    created = svc.create_persona("alice", {**_DRAFT, "session_defaults": {"cefr": "a1"}})
    got = svc.get_persona("alice", created["id"])
    assert got["custom"] is True
    # The full body the list summary omits is present for prefilling the form.
    assert got["persona"]["system_prompt"] == _DRAFT["system_prompt"]
    assert got["persona"]["session_defaults"]["cefr"] == "A1"


def test_get_persona_returns_curated_full_body(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_persona_service(registry, tmp_path)
    got = svc.get_persona("alice", "companion")
    assert got["custom"] is False
    assert got["persona"]["id"] == "companion" and got["persona"]["system_prompt"]


def test_get_custom_persona_requires_owner(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_persona_service(registry, tmp_path)
    created = svc.create_persona("alice", dict(_DRAFT))
    # Bob can't read Alice's persona — it simply doesn't exist for him.
    with pytest.raises(BadRequest, match="unknown persona"):
        svc.get_persona("bob", created["id"])


def test_get_persona_unknown_rejected(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_persona_service(registry, tmp_path)
    with pytest.raises(BadRequest, match="unknown persona"):
        svc.get_persona("alice", "ghost")


# --- issue() with a scoping user (custom-persona resolution at call time) --------------


def test_issue_embeds_and_echoes_user(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_persona_service(registry, tmp_path)
    result = svc.issue(persona="companion", user="alice")
    assert result["user"] == "alice"
    meta = json.loads(decode_token(result["token"], SECRET)["metadata"])
    # The agent reads `user` from metadata to resolve a custom persona + key memory.
    assert meta["user"] == "alice"


def test_issue_omits_user_when_absent(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_persona_service(registry, tmp_path)
    result = svc.issue(persona="companion")
    assert result["user"] is None
    assert "user" not in json.loads(decode_token(result["token"], SECRET)["metadata"])


def test_issue_accepts_owned_custom_persona(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_persona_service(registry, tmp_path)
    created = svc.create_persona("alice", dict(_DRAFT))
    result = svc.issue(persona=created["id"], user="alice")
    assert result["persona"] == created["id"]
    meta = json.loads(decode_token(result["token"], SECRET)["metadata"])
    assert meta["persona"] == created["id"] and meta["user"] == "alice"


def test_issue_rejects_custom_persona_without_user(
    registry: PersonaRegistry, tmp_path: Path
) -> None:
    svc, _ = make_persona_service(registry, tmp_path)
    created = svc.create_persona("alice", dict(_DRAFT))
    # Without the scoping user a custom id is just unknown (curated registry only).
    with pytest.raises(BadRequest, match="unknown persona"):
        svc.issue(persona=created["id"])


def test_issue_rejects_other_users_custom_persona(
    registry: PersonaRegistry, tmp_path: Path
) -> None:
    svc, _ = make_persona_service(registry, tmp_path)
    created = svc.create_persona("alice", dict(_DRAFT))
    with pytest.raises(BadRequest, match="unknown persona"):
        svc.issue(persona=created["id"], user="bob")


def test_issue_rejects_invalid_user(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_persona_service(registry, tmp_path)
    with pytest.raises(BadRequest, match="invalid user id"):
        svc.issue(persona="companion", user="bad id!")


# --- LoRA catalog ---------------------------------------------------------------------


def test_loras_empty_on_non_lora_backend(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_persona_service(registry, tmp_path, supports_lora=False)
    result = svc.loras()
    assert result["supports_lora"] is False
    assert result["loras"] == []
    assert "merged" in result["reason"]


def test_loras_lists_served_modules(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_persona_service(
        registry, tmp_path, supports_lora=True, lora_modules="hr_interviewer=/m/hr pm=/m/pm"
    )
    result = svc.loras()
    assert result["supports_lora"] is True and result["reason"] is None
    assert {o["id"] for o in result["loras"]} == {"hr_interviewer", "pm"}


def test_create_persona_validates_lora_on_lora_backend(
    registry: PersonaRegistry, tmp_path: Path
) -> None:
    svc, _ = make_persona_service(
        registry, tmp_path, supports_lora=True, lora_modules="hr_interviewer=/m/hr"
    )
    # An unknown served adapter is rejected; a served one is accepted (routed by basename).
    with pytest.raises(BadRequest, match="unknown lora"):
        svc.create_persona("alice", {**_DRAFT, "llm": {"base_model": "m", "lora": "ghost"}})
    ok = svc.create_persona(
        "alice", {**_DRAFT, "llm": {"base_model": "m", "lora": "adapters/hr_interviewer"}}
    )
    assert ok["persona"]["llm"]["lora"] == "adapters/hr_interviewer"


# --- persona drafting (authoring helper) ----------------------------------------------


_DRAFT_REPLY = json.dumps(
    {
        "name": "French Tutor",
        "system_prompt": "You are a patient French tutor.",
        "voice": {"ref": "voices/companion_soft", "emotion": "warm"},
        "session_defaults": {"cefr": "b1"},
    }
)


def test_draft_persona_returns_draft_without_persisting(
    registry: PersonaRegistry, tmp_path: Path
) -> None:
    svc, store = make_persona_service(registry, tmp_path, draft_llm=_DraftLLM(_DRAFT_REPLY))
    result = svc.draft_persona("alice", "a patient French tutor who only speaks in B1")
    assert result["custom"] is True
    assert result["id"] == "french-tutor"
    assert result["persona"]["system_prompt"] == "You are a patient French tutor."
    assert result["persona"]["session_defaults"]["cefr"] == "B1"
    # The draft is NOT written — the client confirms via POST /personas separately.
    assert store.ids_for("alice") == []


def test_draft_persona_id_avoids_curated_and_own(registry: PersonaRegistry, tmp_path: Path) -> None:
    reply = json.dumps({"name": "Companion", "system_prompt": "You are a clone."})
    svc, _ = make_persona_service(registry, tmp_path, draft_llm=_DraftLLM(reply))
    result = svc.draft_persona("alice", "a companion")
    assert result["id"] == "companion-2"  # never shadows the curated 'companion'


def test_draft_persona_requires_user_and_description(
    registry: PersonaRegistry, tmp_path: Path
) -> None:
    svc, _ = make_persona_service(registry, tmp_path, draft_llm=_DraftLLM(_DRAFT_REPLY))
    with pytest.raises(BadRequest, match="user id is required"):
        svc.draft_persona(None, "a tutor")
    with pytest.raises(BadRequest, match="description is required"):
        svc.draft_persona("alice", "   ")


def test_draft_persona_disabled_without_factory(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_persona_service(registry, tmp_path)  # no draft_llm wired
    with pytest.raises(ServerMisconfigured, match="not enabled"):
        svc.draft_persona("alice", "a tutor")


def test_draft_persona_bad_reply_is_bad_request(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_persona_service(registry, tmp_path, draft_llm=_DraftLLM("not json at all"))
    with pytest.raises(BadRequest, match="could not draft"):
        svc.draft_persona("alice", "a tutor")


# --- consent (per-user recording opt-in) ----------------------------------------------


def make_consent_service(
    registry: PersonaRegistry,
    tmp_path: Path,
    *,
    api_token: str | None = None,
) -> tuple[TokenService, MemoryStore]:
    """A TokenService backed by a real (plaintext) memory store for the /consent routes."""
    store = MemoryStore(tmp_path / "memory")
    config = TokenServiceConfig(
        livekit_url="wss://livekit.example:7880",
        api_key="APIkey",
        api_secret=SECRET,
        api_token=api_token,
    )
    svc = TokenService(config, registry, default_persona="companion", backend="mac", memory=store)
    return svc, store


def test_consent_grant_reflected_in_store_and_get(
    registry: PersonaRegistry, tmp_path: Path
) -> None:
    svc, store = make_consent_service(registry, tmp_path)
    out = svc.set_consent("alice", granted=True)
    assert (out["user"], out["granted"], out["allow_training"]) == ("alice", True, False)
    assert store.get_consent("alice").granted is True  # persisted to consent.json
    assert svc.get_consent("alice")["granted"] is True  # and read back


def test_consent_revoke(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, store = make_consent_service(registry, tmp_path)
    svc.set_consent("alice", granted=True)
    out = svc.set_consent("alice", granted=False)
    assert out["granted"] is False
    assert store.get_consent("alice").granted is False  # data is kept; just gated off


def test_consent_training_optin(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, store = make_consent_service(registry, tmp_path)
    out = svc.set_consent("alice", granted=True, allow_training=True)
    assert out["allow_training"] is True
    assert store.get_consent("alice").allow_training is True


def test_consent_revoke_forces_training_off(registry: PersonaRegistry, tmp_path: Path) -> None:
    # Can't train on what you can't store: withdrawing consent drops the training opt-in too.
    svc, _ = make_consent_service(registry, tmp_path)
    out = svc.set_consent("alice", granted=False, allow_training=True)
    assert out["granted"] is False and out["allow_training"] is False


def test_consent_get_defaults_to_ungranted(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_consent_service(registry, tmp_path)
    out = svc.get_consent("newcomer")  # never set consent
    assert out["granted"] is False and out["allow_training"] is False


def test_consent_requires_user(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_consent_service(registry, tmp_path)
    with pytest.raises(BadRequest, match="user id is required"):
        svc.set_consent(None, granted=True)


def test_consent_rejects_invalid_user(registry: PersonaRegistry, tmp_path: Path) -> None:
    svc, _ = make_consent_service(registry, tmp_path)
    with pytest.raises(BadRequest, match="invalid user id"):
        svc.set_consent("bad id!", granted=True)


def test_consent_disabled_without_store(registry: PersonaRegistry) -> None:
    svc = make_service(registry)  # no memory store wired
    with pytest.raises(ServerMisconfigured, match="memory is not enabled"):
        svc.set_consent("alice", granted=True)


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


def _get_text(url: str, headers: dict[str, str] | None = None) -> tuple[int, str, str]:
    """GET returning (status, body-as-text, content-type) for non-JSON routes like /metrics."""
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read().decode("utf-8"), resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8"), exc.headers.get("Content-Type", "")


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


def _put(url: str, body: dict, headers: dict[str, str] | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=hdrs, method="PUT")
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


def test_http_healthz_reports_session_load(
    live_server: tuple[str, TokenService], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Admission-control visibility (Feature I). `sessions_max` comes from config so it's always
    # present and authoritative; `sessions_active` comes from the worker's live gauge (only when
    # the exporter is present).
    from personavoice.obs import metrics_enabled, set_sessions

    monkeypatch.setenv("PERSONAVOICE_MAX_SESSIONS", "2")
    base, _ = live_server
    if metrics_enabled():
        set_sessions(1, 2)
    status, body = _get(f"{base}/healthz")
    assert status == 200
    assert body["sessions_max"] == 2  # from config, not the gauge
    if metrics_enabled():
        assert body["sessions_active"] == 1


def test_http_token_503_when_full(
    registry: PersonaRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    # End-to-end: a gated token server returns 503 with a Retry-After header and a retry_after body
    # field once the (fake) worker reports full, so the client backs off instead of connecting.
    monkeypatch.setenv("PERSONAVOICE_MAX_SESSIONS", "1")
    monkeypatch.setenv("PERSONAVOICE_BUSY_RETRY_AFTER", "12")
    svc = _gated_service(registry, lambda: (1, 1))
    httpd = make_server(svc, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    base = f"http://{host}:{port}"
    try:
        req = urllib.request.Request(
            f"{base}/token", data=b"{}", headers={"Content-Type": "application/json"}, method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        assert exc.value.code == 503
        assert exc.value.headers.get("Retry-After") == "12"
        assert json.loads(exc.value.read())["retry_after"] == 12
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_http_metrics_is_open(live_server: tuple[str, TokenService]) -> None:
    # Like /healthz, /metrics is unauthenticated (a scraper doesn't send the bearer token) and
    # returns a Prometheus text exposition.
    base, _ = live_server
    status, body, content_type = _get_text(f"{base}/metrics")
    assert status == 200
    assert content_type.startswith("text/plain")
    # The exporter is a dev dependency, so the per-turn metric families are registered + exposed.
    from personavoice.obs import metrics_enabled

    if metrics_enabled():
        assert "personavoice_turns_total" in body


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
    status, body = _get(f"{base}/token?room=q-room&identity=bob&name=Bob", auth)
    assert status == 200
    assert body["room"] == "q-room"
    assert body["identity"] == "bob"
    assert body["name"] == "Bob"


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
        # Nor metrics scrapes (Prometheus polls on a fixed interval).
        metrics_codes = [_get_text(f"{base}/metrics")[0] for _ in range(3)]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)
    assert codes == [200, 200, 200]
    assert metrics_codes == [200, 200, 200]


# --- HTTP: custom-persona + LoRA routes -----------------------------------------------


@pytest.fixture
def persona_server(
    registry: PersonaRegistry, tmp_path: Path
) -> Iterator[tuple[str, TokenService, UserPersonaStore]]:
    svc, store = make_persona_service(
        registry, tmp_path, supports_lora=True, lora_modules="hr=/m/hr", api_token="sekret"
    )
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


def test_http_persona_crud_round_trip(
    persona_server: tuple[str, TokenService, UserPersonaStore],
) -> None:
    base, _, store = persona_server
    auth = {"Authorization": "Bearer sekret"}

    # Create.
    status, body = _post(
        f"{base}/personas?user=alice", {"name": "French Tutor", "system_prompt": "teach"}, auth
    )
    assert status == 201 and body["id"] == "french-tutor" and body["custom"] is True
    assert store.has("alice", "french-tutor")

    # List (curated + custom) for that user.
    status, body = _get(f"{base}/personas?user=alice", auth)
    assert status == 200
    assert "french-tutor" in {p["id"] for p in body["personas"]}

    # Update.
    status, body = _put(
        f"{base}/personas/french-tutor?user=alice",
        {"name": "Spanish Tutor", "system_prompt": "teach"},
        auth,
    )
    assert status == 200 and body["name"] == "Spanish Tutor"

    # Delete.
    status, body = _delete(f"{base}/personas/french-tutor?user=alice", auth)
    assert status == 200 and body["deleted"] == "french-tutor"
    assert not store.has("alice", "french-tutor")


def test_http_persona_create_requires_auth(
    persona_server: tuple[str, TokenService, UserPersonaStore],
) -> None:
    base, _, _ = persona_server
    status, _ = _post(f"{base}/personas?user=alice", {"name": "x", "system_prompt": "y"})
    assert status == 401


def test_http_persona_create_bad_draft_400(
    persona_server: tuple[str, TokenService, UserPersonaStore],
) -> None:
    base, _, _ = persona_server
    auth = {"Authorization": "Bearer sekret"}
    status, body = _post(f"{base}/personas?user=alice", {"system_prompt": "no name"}, auth)
    assert status == 400 and "name" in body["error"]


def test_http_loras(persona_server: tuple[str, TokenService, UserPersonaStore]) -> None:
    base, _, _ = persona_server
    status, body = _get(f"{base}/loras", {"Authorization": "Bearer sekret"})
    assert status == 200
    assert body["supports_lora"] is True
    assert {o["id"] for o in body["loras"]} == {"hr"}


# --- HTTP: consent routes -------------------------------------------------------------


@pytest.fixture
def consent_server(
    registry: PersonaRegistry, tmp_path: Path
) -> Iterator[tuple[str, TokenService, MemoryStore]]:
    svc, store = make_consent_service(registry, tmp_path, api_token="sekret")
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


def test_http_consent_round_trip(consent_server: tuple[str, TokenService, MemoryStore]) -> None:
    base, _, store = consent_server
    auth = {"Authorization": "Bearer sekret"}

    # Default: a user who never opted in reads back ungranted.
    status, body = _get(f"{base}/consent?user=alice", auth)
    assert status == 200 and body["granted"] is False

    # Grant (the state reports need).
    status, body = _post(f"{base}/consent?user=alice", {"granted": True}, auth)
    assert status == 200 and body["granted"] is True
    assert store.get_consent("alice").granted is True

    # Revoke.
    status, body = _post(f"{base}/consent?user=alice", {"granted": False}, auth)
    assert status == 200 and body["granted"] is False
    assert store.get_consent("alice").granted is False


def test_http_consent_requires_auth(consent_server: tuple[str, TokenService, MemoryStore]) -> None:
    base, _, _ = consent_server
    status, _ = _get(f"{base}/consent?user=alice")
    assert status == 401


def test_http_consent_missing_granted_400(
    consent_server: tuple[str, TokenService, MemoryStore],
) -> None:
    base, _, _ = consent_server
    auth = {"Authorization": "Bearer sekret"}
    status, body = _post(f"{base}/consent?user=alice", {"allow_training": True}, auth)
    assert status == 400
    assert "granted" in body["error"]

"""ConversationMemory facade (M8): consent gating, cross-session recall, scope, cadence."""

from __future__ import annotations

from pathlib import Path

from personavoice.memory import ConversationMemory, MemoryStore
from personavoice.memory.profile import UserProfile
from personavoice.models import MemorySettings, Persona

from .fakes import make_persona


def _persona(*, enabled: bool = True, scope: str = "per_user", summarize_every: int = 6) -> Persona:
    base = make_persona("companion")
    return base.model_copy(
        update={"memory": MemorySettings(enabled=enabled, scope=scope, summarize_every=summarize_every)}
    )


class _ReplyLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    async def chat(self, messages: object, persona: object) -> str:
        self.calls += 1
        return self.reply


def _load_profile(store: MemoryStore, user: str) -> UserProfile:
    raw = store.load_profile_raw(user)
    return UserProfile(user_id=user) if raw is None else UserProfile.model_validate(raw)


# --------------------------------------------------------------------------------------
# consent / enable gating
# --------------------------------------------------------------------------------------


async def test_recall_empty_without_consent(tmp_path: Path) -> None:
    mem = ConversationMemory(MemoryStore(tmp_path))
    assert await mem.recall("alice", "hello", persona=_persona()) == ""


async def test_recall_empty_when_persona_disabled(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.set_consent("alice", granted=True)
    mem = ConversationMemory(store)
    assert await mem.recall("alice", "hello", persona=_persona(enabled=False)) == ""


async def test_record_noop_without_consent(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    mem = ConversationMemory(store)
    await mem.record_user("alice", "s1", _persona(), "hi there")
    assert store.read_turns("alice") == []  # silently dropped (no consent)


async def test_record_persists_with_consent(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.set_consent("alice", granted=True)
    mem = ConversationMemory(store, summarize_every=0)  # no auto-consolidation
    await mem.record_user("alice", "s1", _persona(), "hi")
    await mem.record_assistant("alice", "s1", _persona(), "hello!")
    assert [t.content for t in store.read_turns("alice")] == ["hi", "hello!"]


# --------------------------------------------------------------------------------------
# cross-session recall (the acceptance criterion)
# --------------------------------------------------------------------------------------


async def test_recalls_prior_session_facts(tmp_path: Path) -> None:
    """Record a session, distill a profile, then a fresh facade recalls those facts."""
    store = MemoryStore(tmp_path)
    store.set_consent("alice", granted=True)
    persona = _persona(summarize_every=0)

    # --- session 1: the user shares facts, then we consolidate the profile.
    llm = _ReplyLLM('{"facts": ["name is Sam", "has a dog named Rex"], "summary": "Sam, a dog owner."}')
    mem1 = ConversationMemory(store, llm=llm, summarize_every=0)
    s1 = mem1.start_session("alice", persona.id)
    await mem1.record_user("alice", s1, persona, "Hi, I'm Sam and I have a dog named Rex")
    await mem1.record_assistant("alice", s1, persona, "Nice to meet you, Sam!")
    await mem1.consolidate("alice", persona=persona)

    # --- session 2: a brand-new facade (simulating a later process) recalls the facts.
    mem2 = ConversationMemory(MemoryStore(tmp_path))
    s2 = mem2.start_session("alice", persona.id)
    block = await mem2.recall("alice", "how is my dog doing?", persona=persona, session_id=s2)
    assert "name is Sam" in block
    assert "has a dog named Rex" in block


async def test_recall_excludes_current_session(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.set_consent("alice", granted=True)
    persona = _persona(summarize_every=0)
    mem = ConversationMemory(store, summarize_every=0)
    await mem.record_user("alice", "live", persona, "my favorite color is teal")
    # Querying within the same live session should not echo it back as a "relevant moment".
    block = await mem.recall("alice", "what is my favorite color", persona=persona, session_id="live")
    assert "teal" not in block


# --------------------------------------------------------------------------------------
# per-persona scope
# --------------------------------------------------------------------------------------


async def test_per_user_persona_scope_filters_turns(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.set_consent("alice", granted=True)
    companion = _persona(scope="per_user_persona", summarize_every=0)
    hr = companion.model_copy(update={"id": "hr_interviewer"})

    mem = ConversationMemory(store, summarize_every=0)
    await mem.record_user("alice", "s1", companion, "I love painting landscapes")
    await mem.record_user("alice", "s2", hr, "I managed a team of five engineers")

    # Recall scoped to the companion persona must not surface the HR-session turn.
    block = await mem.recall("alice", "tell me about engineers and painting", persona=companion, session_id="x")
    assert "painting" in block
    assert "engineers" not in block


# --------------------------------------------------------------------------------------
# auto-consolidation cadence
# --------------------------------------------------------------------------------------


async def test_auto_consolidation_runs_in_background(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.set_consent("alice", granted=True)
    persona = _persona(summarize_every=2)
    llm = _ReplyLLM('{"facts": ["enjoys cooking"], "summary": "A home cook."}')
    mem = ConversationMemory(store, llm=llm, summarize_every=2)

    s = mem.start_session("alice", persona.id)
    await mem.record_user("alice", s, persona, "I spent all weekend cooking")  # pending -> 1
    await mem.record_assistant("alice", s, persona, "That sounds delicious!")  # pending -> 2 -> schedule
    await mem.aclose()  # flush the background distillation

    assert llm.calls == 1
    assert "enjoys cooking" in _load_profile(store, "alice").fact_texts()


async def test_no_consolidation_without_llm(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.set_consent("alice", granted=True)
    persona = _persona(summarize_every=2)
    mem = ConversationMemory(store, summarize_every=2)  # no llm
    s = mem.start_session("alice", persona.id)
    await mem.record_user("alice", s, persona, "hi")
    await mem.record_assistant("alice", s, persona, "hello")
    await mem.aclose()
    assert store.load_profile_raw("alice") is None  # nothing distilled

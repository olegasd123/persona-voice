"""Memory wiring: memory injected into the streaming pipeline + surfaced by `server --check`."""

from __future__ import annotations

from pathlib import Path

from personavoice.memory import ConversationMemory, MemoryStore
from personavoice.models import MemorySettings, Persona, Role
from personavoice.orchestrator.streaming import StreamingPipeline
from personavoice.server.check import run_check
from personavoice.server.config import Settings

from .conftest import CONFIG_DIR
from .fakes import make_backend, make_persona


def _mem_persona() -> Persona:
    return make_persona("companion").model_copy(
        update={"memory": MemorySettings(enabled=True, summarize_every=0)}
    )


async def test_streaming_pipeline_injects_and_records_memory(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.set_consent("alice", granted=True)
    store.save_profile_raw(
        "alice",
        {"user_id": "alice", "summary": "Alice has a dog named Rex.", "facts": [{"text": "has a dog named Rex"}]},
    )
    memory = ConversationMemory(store, summarize_every=0)
    backend = make_backend(llm_reply="Rex sounds lovely.")
    persona = _mem_persona()

    pipe = StreamingPipeline(backend, persona, memory=memory, user_id="alice")
    chunks = [c async for c in pipe.stream_response("how is my dog?")]
    assert chunks  # produced audio

    # The recalled profile rode in as a second system message.
    system_msgs = [m for m in (backend.llm.last_messages or []) if m.role is Role.system]
    assert any("has a dog named Rex" in m.content for m in system_msgs)

    # The exchange was persisted under this run's session, tagged with the persona.
    turns = store.read_turns("alice")
    assert [t.role for t in turns] == [Role.user, Role.assistant]
    assert turns[0].content == "how is my dog?"
    assert turns[0].session_id == pipe.session_id
    assert turns[0].persona_id == "companion"


async def test_streaming_pipeline_stateless_without_consent(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)  # no consent granted
    memory = ConversationMemory(store, summarize_every=0)
    backend = make_backend()
    pipe = StreamingPipeline(backend, _mem_persona(), memory=memory, user_id="alice")
    _ = [c async for c in pipe.stream_response("hello")]

    # Nothing recorded, and no memory system message injected.
    assert store.read_turns("alice") == []
    system_msgs = [m for m in (backend.llm.last_messages or []) if m.role is Role.system]
    assert len(system_msgs) == 1  # only the persona prompt


def test_check_reports_memory(tmp_path: Path) -> None:
    settings = Settings(
        backend="mac",
        config_dir=CONFIG_DIR,
        models_dir=tmp_path,
        memory_dir=tmp_path / "memory",
    )
    report = run_check(settings)
    assert report.ok
    assert report.memory_dir == str(tmp_path / "memory")
    assert report.memory_encrypted is False
    # Personas ship with memory enabled but no key set -> a plaintext warning.
    assert any("unencrypted at rest" in w for w in report.warnings)

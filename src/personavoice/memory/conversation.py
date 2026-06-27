"""`ConversationMemory`: the runtime facade the cascade talks to.

It ties the three pieces together behind a small, consent-gated API the orchestrator uses:

    session_id = mem.start_session(user_id, persona.id)
    block = await mem.recall(user_id, user_text, persona=persona, session_id=session_id)
    ...                                  # inject `block`, run the turn
    await mem.record_user(user_id, session_id, persona.id, user_text)
    await mem.record_assistant(user_id, session_id, persona.id, reply)

Two design rules carry over from the rest of the repo:
  - **Consent gates everything.** With no granted consent, `recall` returns "" and the
    `record_*`/`consolidate` calls are no-ops — the assistant simply behaves statelessly, so
    memory can be wired in unconditionally and stays dormant until the user opts in.
  - **The profile is refreshed lazily.** Every `summarize_every` recorded turns (and on
    `consolidate`), the LLM distills recent turns into the rolling profile. With no LLM
    injected, distillation is skipped and recall still works off retrieval alone.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid

from ..eval.report import ReportBuilder, SessionReport
from ..models import Msg, Persona, Role
from .profile import ProfileBuilder, UserProfile, _ChatLLM
from .rag import MemoryRetriever, recall_context
from .store import MemoryStore, MemoryStoreError, MemoryTurn

logger = logging.getLogger("personavoice.memory")

# How many prior-session turns to consider for retrieval. Bounds work + keeps recall recent.
_DEFAULT_RECALL_WINDOW = 400


class ConversationMemory:
    """Consent-gated, cross-session memory for one running cascade."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        llm: _ChatLLM | None = None,
        retriever: MemoryRetriever | None = None,
        k: int = 4,
        summarize_every: int = 6,
        recall_window: int = _DEFAULT_RECALL_WINDOW,
    ) -> None:
        self._store = store
        self._llm = llm
        self._retriever = retriever
        self._k = k
        self._summarize_every = summarize_every
        self._recall_window = recall_window
        # Per (user, session) count of turns recorded since the last consolidation.
        self._pending: dict[tuple[str, str], int] = {}
        # Strong refs to in-flight background consolidations (awaited by `aclose`).
        self._tasks: set[asyncio.Task[UserProfile | None]] = set()

    @property
    def store(self) -> MemoryStore:
        return self._store

    def start_session(self, user_id: str, persona_id: str) -> str:
        """Mint a session id. (user_id/persona_id accepted for symmetry / future scoping.)"""
        del user_id, persona_id
        return uuid.uuid4().hex

    def _enabled(self, persona: Persona | None) -> bool:
        """Memory is active only when the persona opts in (or no persona is given)."""
        return persona is None or persona.memory.enabled

    def _persona_filter(self, persona: Persona | None) -> str | None:
        """When a persona scopes memory `per_user_persona`, restrict to that persona's turns."""
        if persona is not None and persona.memory.scope == "per_user_persona":
            return persona.id
        return None

    async def recall(
        self,
        user_id: str,
        query: str,
        *,
        persona: Persona | None = None,
        session_id: str | None = None,
    ) -> str:
        """The memory block to inject for this turn, or "" (disabled / no consent / nothing)."""
        if not self._enabled(persona) or not query.strip():
            return ""
        try:
            if not self._store.has_consent(user_id):
                return ""
            profile = self._load_profile(user_id)
            persona_id = self._persona_filter(persona)
            turns = self._store.read_turns(
                user_id, persona_id=persona_id, limit=self._recall_window
            )
        except MemoryStoreError:
            logger.warning("memory recall failed for user %r", user_id, exc_info=True)
            return ""
        # Don't surface the in-progress session — the live history already carries it.
        if session_id is not None:
            turns = [t for t in turns if t.session_id != session_id]
        return recall_context(profile, turns, query, retriever=self._retriever, k=self._k)

    async def record_user(
        self, user_id: str, session_id: str, persona: Persona, content: str
    ) -> None:
        await self._record(user_id, session_id, persona, Role.user, content)

    async def record_assistant(
        self, user_id: str, session_id: str, persona: Persona, content: str
    ) -> None:
        await self._record(user_id, session_id, persona, Role.assistant, content)
        # Consolidate on a cadence so the profile keeps up without an LLM call every turn.
        # Run it in the background so distillation never delays the next turn or barge-in.
        if self._summarize_every > 0 and self._llm is not None:
            key = (user_id, session_id)
            if self._pending.get(key, 0) >= self._summarize_every:
                self._pending[key] = 0
                self._schedule_consolidation(user_id, persona, session_id)

    def _schedule_consolidation(self, user_id: str, persona: Persona, session_id: str) -> None:
        task = asyncio.create_task(
            self.consolidate(user_id, persona=persona, session_id=session_id)
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def aclose(self) -> None:
        """Await any background consolidations (call on session/worker shutdown)."""
        for task in list(self._tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def _record(
        self, user_id: str, session_id: str, persona: Persona, role: Role, content: str
    ) -> None:
        if not self._enabled(persona) or not content.strip():
            return
        try:
            if not self._store.has_consent(user_id):
                return
            self._store.record_turn(
                user_id,
                MemoryTurn(
                    session_id=session_id, persona_id=persona.id, role=role, content=content.strip()
                ),
            )
            self._pending[(user_id, session_id)] = self._pending.get((user_id, session_id), 0) + 1
        except MemoryStoreError:
            logger.warning("memory record failed for user %r", user_id, exc_info=True)

    async def consolidate(
        self,
        user_id: str,
        *,
        persona: Persona | None = None,
        session_id: str | None = None,
        max_turns: int = 40,
    ) -> UserProfile | None:
        """Distill recent turns into the rolling profile (no-op without consent or an LLM)."""
        if self._llm is None:
            return None
        try:
            if not self._store.has_consent(user_id):
                return None
            profile = self._load_profile(user_id)
            persona_id = self._persona_filter(persona)
            turns = self._store.read_turns(
                user_id, persona_id=persona_id, session_id=session_id, limit=max_turns
            )
        except MemoryStoreError:
            logger.warning("memory consolidate read failed for user %r", user_id, exc_info=True)
            return None
        if not turns:
            return profile

        builder = ProfileBuilder(self._llm, persona)
        try:
            updated = await builder.update(profile, turns)
        except Exception:  # an LLM/distill hiccup must never break the conversation
            logger.warning("profile distillation failed for user %r", user_id, exc_info=True)
            return profile
        if updated is not profile:
            self._store.save_profile_raw(user_id, updated.model_dump())
            logger.info(
                "memory profile updated for user %r (%d facts)", user_id, len(updated.facts)
            )
        return updated

    async def finalize_report(
        self,
        user_id: str,
        persona: Persona,
        session_id: str,
        transcript: list[Msg],
        *,
        min_user_turns: int = 2,
    ) -> SessionReport | None:
        """Score this session and persist a feedback report (no-op without consent or an LLM).

        Unlike recall/recording, this does **not** require `persona.memory.enabled` — the report
        is about *this* session and works off the live `transcript` (the agent's pipeline
        history), so an interview/tutor persona that doesn't store turns still gets a report. It
        is still consent-gated like everything else in memory; without consent (or an LLM, or
        enough turns) it returns None and stores nothing. Errors are swallowed — a report is a
        nice-to-have at teardown and must never break it.
        """
        if self._llm is None:
            return None
        try:
            if not self._store.has_consent(user_id):
                return None
        except MemoryStoreError:
            logger.warning("memory consent check failed for user %r", user_id, exc_info=True)
            return None
        try:
            report = await ReportBuilder(self._llm).build(
                persona, transcript, session_id=session_id, min_user_turns=min_user_turns
            )
        except Exception:  # an LLM hiccup must never break session teardown
            logger.warning("session report generation failed for user %r", user_id, exc_info=True)
            return None
        if report is None:
            return None
        try:
            self._store.save_report(user_id, session_id, report.model_dump(mode="json"))
        except MemoryStoreError:
            logger.warning("saving session report failed for user %r", user_id, exc_info=True)
            return report
        logger.info("session report saved for user %r session %r", user_id, session_id)
        return report

    def _load_profile(self, user_id: str) -> UserProfile:
        raw = self._store.load_profile_raw(user_id)
        if raw is None:
            return UserProfile(user_id=user_id)
        return UserProfile.model_validate(raw)

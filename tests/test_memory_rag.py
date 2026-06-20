"""Retrieval + context assembly: KeywordRetriever ranking and recall_context."""

from __future__ import annotations

from personavoice.memory.profile import ProfileFact, UserProfile
from personavoice.memory.rag import KeywordRetriever, recall_context
from personavoice.memory.store import MemoryTurn
from personavoice.models import Role


def _turn(content: str, *, role: Role = Role.user, session: str = "s1") -> MemoryTurn:
    return MemoryTurn(session_id=session, persona_id="companion", role=role, content=content)


# --------------------------------------------------------------------------------------
# KeywordRetriever
# --------------------------------------------------------------------------------------


def test_keyword_retriever_ranks_by_overlap() -> None:
    turns = [
        _turn("we talked about my sister's wedding in June"),
        _turn("I really enjoy playing chess on weekends"),
        _turn("the weather has been rainy lately"),
    ]
    ranked = KeywordRetriever().rank("tell me about the wedding", turns, k=2)
    assert ranked, "expected at least one match"
    assert "wedding" in ranked[0][1].content


def test_keyword_retriever_skips_zero_overlap() -> None:
    turns = [_turn("chess and hiking")]
    assert KeywordRetriever().rank("astrophysics quasar", turns, k=5) == []


def test_keyword_retriever_respects_k_and_empty_query() -> None:
    turns = [_turn("dogs"), _turn("dogs and cats"), _turn("dogs cats birds")]
    assert len(KeywordRetriever().rank("dogs cats", turns, k=2)) == 2
    assert KeywordRetriever().rank("", turns, k=5) == []
    assert KeywordRetriever().rank("dogs", turns, k=0) == []


def test_keyword_retriever_ignores_stopwords() -> None:
    # A query of only stopwords carries no signal.
    turns = [_turn("the and of to")]
    assert KeywordRetriever().rank("the and of", turns, k=5) == []


# --------------------------------------------------------------------------------------
# recall_context
# --------------------------------------------------------------------------------------


def test_recall_context_includes_profile_and_turns() -> None:
    profile = UserProfile(
        user_id="alice",
        summary="Alice loves jazz.",
        facts=[ProfileFact(text="has a dog named Rex")],
    )
    turns = [_turn("I'm worried about my dog Rex's vet visit")]
    block = recall_context(profile, turns, "how is Rex doing?", k=3)
    assert "Alice loves jazz." in block
    assert "has a dog named Rex" in block
    assert "Rex" in block
    assert "do not recite it verbatim" in block  # the guard header


def test_recall_context_empty_when_nothing() -> None:
    assert recall_context(None, [], "anything", k=3) == ""
    assert recall_context(UserProfile(user_id="a"), [], "anything", k=3) == ""


def test_recall_context_profile_only_when_no_turns_match() -> None:
    profile = UserProfile(user_id="alice", summary="Alice loves jazz.")
    block = recall_context(profile, [_turn("totally unrelated content")], "quantum mechanics", k=3)
    assert "Alice loves jazz." in block
    assert "Relevant moments" not in block


def test_recall_context_turns_only_without_profile() -> None:
    turns = [_turn("my favorite color is teal")]
    block = recall_context(None, turns, "what is my favorite color", k=2)
    assert "Relevant moments from earlier conversations" in block
    assert "teal" in block

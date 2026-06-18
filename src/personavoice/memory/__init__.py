"""Memory / learn-from-conversations.

Continuity across sessions on top of the cascade: a consent-gated per-user store, a
rolling LLM-distilled profile, and retrieval of relevant past turns — assembled into a
memory block injected into the prompt each turn.

    store.py        — MemoryStore: per-user transcripts/profile/consent + delete/export (privacy)
    profile.py      — UserProfile + ProfileBuilder (LLM distillation into durable facts/summary)
    rag.py          — retrievers (keyword default, optional embeddings) + recall_context
    conversation.py — ConversationMemory: the consent-gated facade the orchestrator uses
    distill.py      — transcripts → persona-LoRA training data (the memory→LoRA bridge)
"""

from .conversation import ConversationMemory
from .distill import distill_user, transcripts_to_examples
from .profile import ProfileBuilder, ProfileFact, UserProfile
from .rag import EmbeddingRetriever, KeywordRetriever, MemoryRetriever, recall_context
from .store import (
    Cipher,
    Consent,
    FernetCipher,
    MemoryStore,
    MemoryStoreError,
    MemoryTurn,
    NullCipher,
    cipher_from_key,
)

__all__ = [
    "Cipher",
    "Consent",
    "ConversationMemory",
    "EmbeddingRetriever",
    "FernetCipher",
    "KeywordRetriever",
    "MemoryRetriever",
    "MemoryStore",
    "MemoryStoreError",
    "MemoryTurn",
    "NullCipher",
    "ProfileBuilder",
    "ProfileFact",
    "UserProfile",
    "cipher_from_key",
    "distill_user",
    "recall_context",
    "transcripts_to_examples",
]

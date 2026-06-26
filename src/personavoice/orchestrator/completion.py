"""Semantic endpointing (Feature G) — "did the user actually finish talking?".

Endpointing is otherwise pure Silero-VAD silence (`endpointing.py`): the agent treats *any*
trailing pause as the end of a turn. That can't tell a mid-thought pause ("I think… *pause*
…it's fine") from a finished sentence, so it both barges in on the user mid-thought and, with a
longer silence window, adds dead air. This module is the cheap "is this utterance complete?" gate
the plan calls for: a punctuation/heuristic classifier over the *transcript* of a VAD segment.

`agent.py` consults it after VAD fires END_OF_SPEECH and STT returns text: a fragment that looks
unfinished (trails off into an ellipsis, or ends on a trailing conjunction / preposition /
determiner or a filler sound) is *held* and merged with the next utterance instead of triggering
its own reply; a grace timer
flushes the held fragment if the user doesn't continue, so a misjudged hold only costs that delay,
never a dropped turn.

Everything here is pure (no I/O, no ML deps, no LiveKit) so the grammar is fully unit-testable
offline — the same discipline as `endpointing.py` / `emotion.py`. The whole feature is opt-in via
``PERSONAVOICE_SEMANTIC_ENDPOINTING`` (default off): when it's off the agent never holds, so
behavior is unchanged.

The heuristic is deliberately *conservative*: a false "complete" lets the user get barged in on
(bad), while a false "incomplete" only delays the reply by the grace window (mildly bad, always
recovered). So the continuation-word sets cover only words that almost never end a natural spoken
sentence; ambiguous enders (modals like "will", object-pronoun "her", "for"/"in"/"on") are left
out on purpose and judged complete.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

# Coordinating / subordinating conjunctions that dangle: a turn ending on one is almost always
# waiting for its second clause ("it's broken **because**", "I went **and**"). Conjunctions that
# double as common sentence enders ("that", "as", "for", "since", "while") are excluded to avoid
# holding a finished line like "I like that".
CONJUNCTIONS: frozenset[str] = frozenset(
    {"and", "but", "or", "nor", "so", "because", "although", "though", "whereas", "unless"}
)

# Prepositions that expect an object to follow ("I'm thinking **of**", "I want **to**", "put it
# **into**"). Limited to ones that rarely end a complete utterance; stranded-preposition enders
# ("what are you waiting **for**", "come **on**") are excluded.
PREPOSITIONS: frozenset[str] = frozenset(
    {"of", "to", "with", "from", "into", "onto", "upon", "toward", "towards"}
)

# Articles / possessive determiners that must be followed by a noun ("pass me **the**", "I saw
# **a**", "it's **your**"). Object-form pronouns ("his", "her", "its") are excluded — they double
# as sentence enders ("I gave it to **her**").
DETERMINERS: frozenset[str] = frozenset({"a", "an", "the", "my", "your", "our", "their"})

# Vocal hesitation sounds: a turn that trails off into one is mid-thought ("it was, **um**"). Kept
# to unambiguous fillers — "like"/"well"/"you know" carry meaning too often to treat as a hold.
FILLERS: frozenset[str] = frozenset({"um", "uh", "uhh", "uhm", "er", "erm", "hmm", "mmm", "ah"})

# Trailing function words that signal a continuation (everything but the fillers, which get their
# own verdict reason). Ending a fragment on one of these means "hold and wait for more".
CONTINUATION_WORDS: frozenset[str] = CONJUNCTIONS | PREPOSITIONS | DETERMINERS

# A trailing ellipsis ("…", or two-or-more ASCII dots) is a *trailing off* — the canonical
# mid-thought pause ("I think…") — so it reads as INCOMPLETE even though it ends in a dot. STT
# routinely renders a hesitation this way, so it's checked before plain terminal punctuation.
_ELLIPSIS: tuple[str, ...] = ("…", "..")

# Single sentence-final punctuation: if STT emitted one, the utterance is a finished sentence.
# Note "." but not ".." — a lone dot ends a sentence; a run of them is the ellipsis above.
_TERMINALS: tuple[str, ...] = (".", "!", "?")

# Word = a run of letters (with internal apostrophes for contractions). Used to read the last word.
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")

_DEFAULT_GRACE_MS = 1500
_TRUTHY = ("1", "true", "yes", "on")


@dataclass(frozen=True)
class CompletionVerdict:
    """Whether a transcript fragment reads as a finished turn, plus why (for logs/tests)."""

    complete: bool
    reason: str


def _last_word(text: str) -> str:
    """The final alphabetic word of `text`, lowercased (or '' if there is none)."""
    words = _WORD_RE.findall(text.lower())
    return words[-1] if words else ""


def assess_completion(text: str) -> CompletionVerdict:
    """Classify a VAD utterance transcript as a finished turn or a mid-thought pause.

    Complete when empty, ended with a single sentence-final mark, or ended on an ordinary content
    word. Incomplete when it trails off into an ellipsis ("I think…"), ends on a hesitation
    filler, or ends on a dangling function word (conjunction / preposition / determiner) that
    expects more to follow.
    """
    stripped = text.strip()
    if not stripped:
        return CompletionVerdict(complete=True, reason="empty")
    if stripped.endswith(_ELLIPSIS):
        return CompletionVerdict(complete=False, reason="trailing-ellipsis")
    if stripped.endswith(_TERMINALS):
        return CompletionVerdict(complete=True, reason="terminal-punctuation")
    last = _last_word(stripped)
    if not last:
        return CompletionVerdict(complete=True, reason="no-word")
    if last in FILLERS:
        return CompletionVerdict(complete=False, reason="trailing-filler")
    if last in CONTINUATION_WORDS:
        return CompletionVerdict(complete=False, reason="trailing-function-word")
    return CompletionVerdict(complete=True, reason="default")


def is_complete(text: str) -> bool:
    """Convenience boolean wrapper over `assess_completion`."""
    return assess_completion(text).complete


def join_fragments(held: str | None, new: str) -> str:
    """Merge a previously held fragment with the next utterance into one user turn.

    A held fragment is only ever an unfinished line (no terminal punctuation), so a single space
    join reads naturally ("I think" + "it's fine" → "I think it's fine"). Tolerates a missing or
    blank side.
    """
    new = new.strip()
    if not held or not held.strip():
        return new
    held = held.strip()
    if not new:
        return held
    return f"{held} {new}"


def semantic_endpointing_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Whether semantic endpointing is on, from ``PERSONAVOICE_SEMANTIC_ENDPOINTING`` (default off).

    Off by default so existing deployments keep pure-VAD endpointing; set it to a truthy value
    (``1``/``true``/``yes``/``on``) to have the agent hold and merge unfinished utterances.
    """
    env = os.environ if env is None else env
    return (env.get("PERSONAVOICE_SEMANTIC_ENDPOINTING") or "").strip().lower() in _TRUTHY


def endpointing_grace_s(env: Mapping[str, str] | None = None) -> float:
    """Grace window (seconds) to wait for a continuation after an unfinished utterance.

    From ``PERSONAVOICE_ENDPOINTING_GRACE_MS`` (default 1500 ms); a blank, non-integer, or
    negative value falls back to the default. This bounds the worst case of a misjudged hold:
    if the user really had finished, the reply is delayed by at most this long before flushing.
    """
    env = os.environ if env is None else env
    raw = (env.get("PERSONAVOICE_ENDPOINTING_GRACE_MS") or "").strip()
    if not raw:
        return _DEFAULT_GRACE_MS / 1000.0
    try:
        ms = int(raw)
    except ValueError:
        return _DEFAULT_GRACE_MS / 1000.0
    if ms < 0:
        return _DEFAULT_GRACE_MS / 1000.0
    return ms / 1000.0

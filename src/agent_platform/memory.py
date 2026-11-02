"""Per-tenant conversation memory built on ``agent_memory``.

Each tenant is one ``agent_memory`` namespace in a separate SQLite file next to the
platform database, so memories never mix across tenants and the platform schema is
untouched. A request that carries a ``session_id``:

* stores the question and the answer as episodes in that session, which runs the library's
  fact extraction and versioning on the write path;
* lets the knowledge-base search tool also return the tenant's most relevant memories,
  labeled ``[memory:<id>]``, so an answer drawn from memory is cited and checked for
  grounding exactly like one drawn from a document.

The memory store's SQLite connection is shared across server threads, so every call is
serialized with one lock.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_memory import AgentMemory

MEMORY_PREFIX = "memory:"
MAX_SESSION_ID = 128


@dataclass(frozen=True)
class MemoryHit:
    """One recalled memory, in the shape the agent's search tool renders."""

    memory_id: int
    text: str
    score: float
    explanation: str

    @property
    def doc_id(self) -> str:
        return f"{MEMORY_PREFIX}{self.memory_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.memory_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "score": round(self.score, 4),
            "explanation": self.explanation,
        }


def validate_session_id(session_id: object) -> str:
    """Return ``session_id`` if it is a short identifier, else raise ``ValueError``."""
    if (
        not isinstance(session_id, str)
        or not 0 < len(session_id) <= MAX_SESSION_ID
        or not all(c.isalnum() or c in "_.:@-" for c in session_id)
    ):
        raise ValueError(
            f"'session_id' must be 1-{MAX_SESSION_ID} characters of letters, digits, and _ . : @ -"
        )
    return session_id


class TenantMemory:
    """Long-term memory for every tenant, backed by one ``agent_memory`` store."""

    def __init__(self, path: Path | str) -> None:
        self._root = AgentMemory(str(path), namespace="platform")
        self._lock = threading.Lock()

    def _for(self, tenant: str) -> AgentMemory:
        return self._root.for_namespace(tenant)

    def remember(self, tenant: str, text: str, *, session_id: str = "notes") -> dict[str, Any]:
        """Store ``text`` for ``tenant``; return the episode id and any extracted facts."""
        if not text.strip():
            raise ValueError("'text' must be a non-empty string")
        with self._lock:
            result = self._for(tenant).add_turn(text, "user", session_id=session_id)
        return {
            "episode_id": result.episode.id,
            "facts": [f"{f.subject} {f.predicate} {f.object}" for f in result.facts],
            "actions": list(result.actions),
        }

    def record_turn(self, tenant: str, session_id: str, question: str, answer: str) -> None:
        """Store one question and its answer as episodes in ``session_id``."""
        with self._lock:
            memory = self._for(tenant)
            memory.add_turn(question, "user", session_id=session_id)
            memory.add_turn(answer, "assistant", session_id=session_id)

    def recall(self, tenant: str, query: str, k: int = 3) -> list[MemoryHit]:
        """Return up to ``k`` memories for ``query`` with their score explanations."""
        if not query.strip() or k <= 0:
            return []
        with self._lock:
            results = self._for(tenant).recall(query, k)
        return [
            MemoryHit(
                memory_id=item.id,
                text=item.text,
                score=float(item.score.total),
                explanation=item.explanation,
            )
            for item in results
        ]

    def session(self, tenant: str, session_id: str) -> list[dict[str, Any]]:
        """Return the episodes of one session, oldest first."""
        with self._lock:
            episodes = sorted(self._for(tenant).episodes(session_id), key=lambda e: e.id)
        return [{"role": e.role, "text": e.content, "created_at": e.created_at} for e in episodes]

    def close(self) -> None:
        self._root.close()

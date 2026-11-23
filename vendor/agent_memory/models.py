"""Typed records for the three memory kinds and for retrieval results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class MemoryKind(StrEnum):
    """The three memory kinds the store holds."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class Status(StrEnum):
    """Lifecycle state of a stored memory.

    ``active`` memories are retrievable by default. ``superseded`` facts and preferences were
    replaced by newer evidence and are kept as history. ``contested`` facts arrived as weaker
    evidence that did not override the current value; they are also kept as history.
    """

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    CONTESTED = "contested"


@dataclass(frozen=True)
class Episode:
    """An episodic memory: one timestamped conversation turn or event."""

    id: int
    namespace: str
    content: str
    created_at: float
    session_id: str
    role: str
    importance: float
    access_count: int = 0
    last_accessed: float | None = None
    consolidated: bool = False
    status: Status = Status.ACTIVE
    kind: MemoryKind = MemoryKind.EPISODIC

    @property
    def text(self) -> str:
        return self.content


@dataclass(frozen=True)
class Fact:
    """A semantic memory: a distilled (subject, predicate, object) triple with confidence."""

    id: int
    namespace: str
    subject: str
    predicate: str
    object: str
    confidence: float
    created_at: float
    importance: float
    updated_at: float
    evidence_count: int = 1
    source_episode_id: int | None = None
    superseded_by: int | None = None
    access_count: int = 0
    last_accessed: float | None = None
    status: Status = Status.ACTIVE
    kind: MemoryKind = MemoryKind.SEMANTIC

    @property
    def text(self) -> str:
        return render_fact(self.subject, self.predicate, self.object)


@dataclass(frozen=True)
class Preference:
    """A procedural memory: a key-value preference or instruction with provenance."""

    id: int
    namespace: str
    key: str
    value: str
    provenance: str
    created_at: float
    importance: float
    source_episode_id: int | None = None
    superseded_by: int | None = None
    access_count: int = 0
    last_accessed: float | None = None
    status: Status = Status.ACTIVE
    kind: MemoryKind = MemoryKind.PROCEDURAL

    @property
    def text(self) -> str:
        return f"preference {self.key.replace('_', ' ')}: {self.value}"


Memory = Episode | Fact | Preference


def render_fact(subject: str, predicate: str, obj: str) -> str:
    """Render a triple as searchable text, for example ``user preference for editor is vim``."""
    if predicate == "summary":
        return f"{subject} summary: {obj}"
    if predicate.startswith("prefers:"):
        topic = predicate.split(":", 1)[1]
        return f"{subject} preference for {topic} is {obj}"
    return f"{subject} {predicate.replace('_', ' ')} {obj}"


@dataclass(frozen=True)
class ScoreBreakdown:
    """Per-term contributions to a retrieval score; ``total`` is their weighted sum."""

    lexical: float
    semantic: float
    recency: float
    importance: float
    total: float
    matched_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievedMemory:
    """A retrieval result with its score breakdown and a human-readable explanation."""

    memory: Memory
    score: ScoreBreakdown
    explanation: str

    @property
    def id(self) -> int:
        return self.memory.id

    @property
    def text(self) -> str:
        return self.memory.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.memory.id,
            "kind": self.memory.kind.value,
            "status": self.memory.status.value,
            "text": self.memory.text,
            "created_at": self.memory.created_at,
            "score": round(self.score.total, 4),
            "components": {
                "lexical": round(self.score.lexical, 4),
                "semantic": round(self.score.semantic, 4),
                "recency": round(self.score.recency, 4),
                "importance": round(self.score.importance, 4),
            },
            "matched_terms": list(self.score.matched_terms),
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class IngestResult:
    """What one ``add_turn`` call wrote, including any fact merges it triggered."""

    episode: Episode
    facts: tuple[Fact, ...] = ()
    preferences: tuple[Preference, ...] = ()
    actions: tuple[str, ...] = ()


@dataclass
class ConsolidationReport:
    """Counts from one consolidation pass."""

    sessions_summarized: int = 0
    episodes_consolidated: int = 0
    facts_written: int = 0
    episodes_forgotten: int = 0
    facts_forgotten: int = 0
    details: list[str] = field(default_factory=list)

"""The public facade: a namespace-scoped long-term memory for one user or agent."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from agent_memory.consolidation import run_consolidation
from agent_memory.embedding import Embedder, HashingEmbedder
from agent_memory.extraction import CandidateFact, CandidatePreference, Extractor, RuleExtractor
from agent_memory.merge import FactMerger, MergeOutcome, PreferenceOutcome
from agent_memory.models import (
    ConsolidationReport,
    Episode,
    Fact,
    IngestResult,
    Memory,
    MemoryKind,
    Preference,
    RetrievedMemory,
    Status,
    render_fact,
)
from agent_memory.scoring import RetrievalRequest, score_candidates
from agent_memory.settings import MemorySettings, ScoringWeights
from agent_memory.store import MemoryStore, validate_namespace

EXPORT_FORMAT = "agent-memory/v1"


def episode_importance(content: str, role: str, extracted: bool) -> float:
    """Heuristic importance for a new episode: turns that state durable facts matter most."""
    importance = 0.8 if extracted else 0.3
    if role != "user":
        importance -= 0.1
    if content.rstrip().endswith("?"):
        importance -= 0.1
    return round(min(max(importance, 0.05), 1.0), 3)


class AgentMemory:
    """Long-term memory for one namespace (a user, an agent, or a user-agent pair).

    Example::

        memory = AgentMemory("memory.db", namespace="alice")
        memory.add_turn("My favorite editor is Neovim.")
        for hit in memory.recall("which editor does alice prefer?"):
            print(hit.text, hit.explanation)
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        namespace: str = "default",
        *,
        settings: MemorySettings | None = None,
        extractor: Extractor | None = None,
        embedder: Embedder | None = None,
        clock: Callable[[], float] | None = None,
        store: MemoryStore | None = None,
    ) -> None:
        self.namespace = validate_namespace(namespace)
        self.settings = settings or MemorySettings()
        self.extractor: Extractor = extractor or RuleExtractor()
        self.embedder: Embedder = embedder or HashingEmbedder()
        self.clock = clock or time.time
        self.store = store or MemoryStore(path)
        self._owns_store = store is None
        self.merger = FactMerger(
            self.store,
            self.embedder,
            self.settings.multi_valued_predicates,
            self.settings.conflict_margin,
        )

    def for_namespace(self, namespace: str) -> AgentMemory:
        """A view of another namespace that shares this store, settings, and components."""
        return AgentMemory(
            namespace=namespace,
            settings=self.settings,
            extractor=self.extractor,
            embedder=self.embedder,
            clock=self.clock,
            store=self.store,
        )

    def close(self) -> None:
        if self._owns_store:
            self.store.close()

    def __enter__(self) -> AgentMemory:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -------------------------------------------------------------- write path

    def add_turn(
        self,
        content: str,
        role: str = "user",
        *,
        session_id: str = "default",
        timestamp: float | None = None,
        importance: float | None = None,
    ) -> IngestResult:
        """Store one conversation turn as an episode and merge any facts it states."""
        if not content.strip():
            raise ValueError("content must not be empty")
        created_at = self.clock() if timestamp is None else timestamp
        extraction = None
        if self.settings.extract_on_write:
            extraction = self.extractor.extract(content, role)
        extracted = extraction is not None and not extraction.empty
        episode = self.store.insert_episode(
            self.namespace,
            content,
            session_id=session_id,
            role=role,
            created_at=created_at,
            importance=(
                episode_importance(content, role, extracted) if importance is None else importance
            ),
            embedding=self.embedder.embed(content),
        )
        if extraction is None:
            return IngestResult(episode=episode)
        facts, preferences, actions = self._apply(extraction.facts, extraction.preferences, episode)
        return IngestResult(episode, tuple(facts), tuple(preferences), tuple(actions))

    def _apply(
        self,
        facts: Iterable[CandidateFact],
        preferences: Iterable[CandidatePreference],
        episode: Episode,
    ) -> tuple[list[Fact], list[Preference], list[str]]:
        merged: list[MergeOutcome] = [
            self.merger.merge(self.namespace, fact, episode.created_at, episode.id)
            for fact in facts
        ]
        provenance = f"extracted from episode {episode.id} ({type(self.extractor).__name__})"
        set_preferences: list[PreferenceOutcome] = [
            self.merger.merge_preference(
                self.namespace, item, episode.created_at, provenance, episode.id
            )
            for item in preferences
        ]
        return (
            [outcome.fact for outcome in merged],
            [outcome.preference for outcome in set_preferences],
            [outcome.action for outcome in merged] + [item.action for item in set_preferences],
        )

    def extract_and_merge(self, episode: Episode) -> list[str]:
        """Run extraction on a stored episode (used by consolidation when write-time is off)."""
        extraction = self.extractor.extract(episode.content, episode.role)
        _, _, actions = self._apply(extraction.facts, extraction.preferences, episode)
        return actions

    def remember(self, text: str, *, session_id: str = "notes") -> IngestResult:
        """Store a free-form note from the agent or user, extracting any facts it states."""
        return self.add_turn(text, "user", session_id=session_id)

    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        confidence: float = 0.9,
        timestamp: float | None = None,
    ) -> MergeOutcome:
        """Write a fact directly, applying the same merge rules as extracted facts."""
        observed = self.clock() if timestamp is None else timestamp
        candidate = CandidateFact(subject, predicate, obj, min(max(confidence, 0.0), 1.0))
        return self.merger.merge(self.namespace, candidate, observed)

    def set_preference(
        self, key: str, value: str, *, provenance: str = "explicit", timestamp: float | None = None
    ) -> Preference:
        observed = self.clock() if timestamp is None else timestamp
        outcome = self.merger.merge_preference(
            self.namespace, CandidatePreference(key, value, 1.0), observed, provenance
        )
        return outcome.preference

    def forget(self, memory_id: int) -> bool:
        """Delete one memory from this namespace. Returns False if it does not exist here."""
        return self.store.delete(self.namespace, [memory_id]) > 0

    # --------------------------------------------------------------- read path

    def recall(
        self,
        query: str,
        k: int = 5,
        *,
        include_history: bool = False,
        kinds: Sequence[MemoryKind] | None = None,
        weights: ScoringWeights | None = None,
        now: float | None = None,
        touch: bool = True,
    ) -> list[RetrievedMemory]:
        """Return the ``k`` best memories for ``query`` with score breakdowns and explanations."""
        if k <= 0:
            return []
        statuses = (
            [Status.ACTIVE, Status.SUPERSEDED, Status.CONTESTED]
            if include_history
            else [Status.ACTIVE]
        )
        moment = self.clock() if now is None else now
        request = RetrievalRequest(
            namespace=self.namespace,
            query=query,
            k=k,
            now=moment,
            weights=weights or self.settings.weights,
            half_life_days=self.settings.recency_half_life_days,
            candidate_pool=max(self.settings.candidate_pool, k),
            min_semantic_similarity=self.settings.min_semantic_similarity,
            statuses=statuses,
            kinds=frozenset(kinds) if kinds else None,
        )
        results = score_candidates(self.store, self.embedder, request)
        if touch and results:
            self.store.touch(self.namespace, [item.id for item in results], moment)
        return results

    def get(self, memory_id: int) -> Memory | None:
        return self.store.get(self.namespace, memory_id)

    def facts(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        include_history: bool = False,
    ) -> list[Fact]:
        statuses = None if include_history else [Status.ACTIVE]
        found = self.store.list_memories(
            self.namespace, kind=MemoryKind.SEMANTIC, statuses=statuses
        )
        return [
            fact
            for fact in found
            if isinstance(fact, Fact)
            and (subject is None or fact.subject == subject)
            and (predicate is None or fact.predicate == predicate)
        ]

    def current_fact(self, subject: str, predicate: str) -> Fact | None:
        active = [
            fact
            for fact in self.store.facts_for(self.namespace, subject, predicate)
            if fact.status is Status.ACTIVE
        ]
        return max(active, key=lambda fact: (fact.updated_at, fact.id)) if active else None

    def fact_history(self, subject: str, predicate: str) -> list[Fact]:
        """Every value a fact has held, oldest first, including superseded and contested ones."""
        return self.store.facts_for(self.namespace, subject, predicate)

    def preferences(self, include_history: bool = False) -> list[Preference]:
        statuses = None if include_history else [Status.ACTIVE]
        found = self.store.list_memories(
            self.namespace, kind=MemoryKind.PROCEDURAL, statuses=statuses
        )
        return [item for item in found if isinstance(item, Preference)]

    def episodes(self, session_id: str | None = None) -> list[Episode]:
        found = self.store.list_memories(self.namespace, kind=MemoryKind.EPISODIC)
        return [
            item
            for item in found
            if isinstance(item, Episode) and (session_id is None or item.session_id == session_id)
        ]

    # ------------------------------------------------------------ maintenance

    def consolidate(self, now: float | None = None) -> ConsolidationReport:
        return run_consolidation(
            self.store,
            self.namespace,
            self.settings,
            self.embedder,
            self.extract_and_merge,
            self.clock() if now is None else now,
        )

    def stats(self) -> dict[str, Any]:
        counts = self.store.counts(self.namespace)
        return {
            "namespace": self.namespace,
            "episodes": counts.get("episodic.active", 0),
            "episodes_consolidated": counts.get("episodic.consolidated", 0),
            "facts_active": counts.get("semantic.active", 0),
            "facts_history": counts.get("semantic.superseded", 0)
            + counts.get("semantic.contested", 0),
            "preferences_active": counts.get("procedural.active", 0),
            "preferences_history": counts.get("procedural.superseded", 0),
            "database_bytes": self.store.size_bytes(),
        }

    # ---------------------------------------------------------- export/import

    def export_json(self) -> dict[str, Any]:
        """Serialize this namespace (all statuses) to a JSON-compatible dictionary."""
        records: list[dict[str, Any]] = []
        for item in self.store.list_memories(self.namespace):
            record: dict[str, Any] = {
                "id": item.id,
                "kind": item.kind.value,
                "status": item.status.value,
                "created_at": item.created_at,
                "importance": item.importance,
                "access_count": item.access_count,
                "last_accessed": item.last_accessed,
            }
            if isinstance(item, Episode):
                record |= {
                    "content": item.content,
                    "session_id": item.session_id,
                    "role": item.role,
                    "consolidated": item.consolidated,
                }
            elif isinstance(item, Fact):
                record |= {
                    "subject": item.subject,
                    "predicate": item.predicate,
                    "object": item.object,
                    "confidence": item.confidence,
                    "updated_at": item.updated_at,
                    "evidence_count": item.evidence_count,
                    "source_episode_id": item.source_episode_id,
                    "superseded_by": item.superseded_by,
                }
            else:
                record |= {
                    "key": item.key,
                    "value": item.value,
                    "provenance": item.provenance,
                    "source_episode_id": item.source_episode_id,
                    "superseded_by": item.superseded_by,
                }
            records.append(record)
        return {"format": EXPORT_FORMAT, "namespace": self.namespace, "memories": records}

    def export_to(self, path: str | Path) -> int:
        payload = self.export_json()
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return len(payload["memories"])

    def import_json(self, payload: dict[str, Any]) -> int:
        """Load an export into this namespace. Identifiers are remapped; links are preserved."""
        if payload.get("format") != EXPORT_FORMAT:
            raise ValueError(f"unsupported export format: {payload.get('format')!r}")
        id_map: dict[int, int] = {}
        pending_links: list[tuple[int, Status, int]] = []
        store, namespace = self.store, self.namespace
        for record in payload.get("memories", []):
            kind = MemoryKind(record["kind"])
            status = Status(record.get("status", "active"))
            if kind is MemoryKind.EPISODIC:
                new: Memory = store.insert_episode(
                    namespace,
                    record["content"],
                    session_id=record["session_id"],
                    role=record["role"],
                    created_at=record["created_at"],
                    importance=record["importance"],
                    embedding=self.embedder.embed(record["content"]),
                )
                if record.get("consolidated"):
                    store.mark_consolidated(namespace, [new.id])
            elif kind is MemoryKind.SEMANTIC:
                text = render_fact(record["subject"], record["predicate"], record["object"])
                new = store.insert_fact(
                    namespace,
                    subject=record["subject"],
                    predicate=record["predicate"],
                    obj=record["object"],
                    text=text,
                    confidence=record["confidence"],
                    created_at=record["created_at"],
                    importance=record["importance"],
                    embedding=self.embedder.embed(text),
                    status=status,
                )
            else:
                text = f"preference {record['key'].replace('_', ' ')}: {record['value']}"
                new = store.insert_preference(
                    namespace,
                    key=record["key"],
                    value=record["value"],
                    text=text,
                    provenance=record["provenance"],
                    created_at=record["created_at"],
                    importance=record["importance"],
                    embedding=self.embedder.embed(text),
                )
            id_map[int(record["id"])] = new.id
            if status is not Status.ACTIVE or record.get("superseded_by") is not None:
                pending_links.append((new.id, status, record.get("superseded_by") or 0))
        for new_id, status, old_target in pending_links:
            store.set_status(namespace, new_id, status, superseded_by=id_map.get(old_target))
        return len(id_map)

    def import_from(self, path: str | Path) -> int:
        return self.import_json(json.loads(Path(path).read_text(encoding="utf-8")))

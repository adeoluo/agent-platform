"""Write-path deduplication and conflict resolution for facts and preferences.

Facts are keyed by ``(subject, predicate)``. For a candidate fact:

1. If an active fact already has the same normalized object, it is *reinforced*: confidences
   combine as independent evidence, ``1 - (1 - c_old) * (1 - c_new)``, and the evidence count
   and last-observed time are updated. No new row is written.
2. If the predicate is multi-valued (for example ``allergic_to``), the candidate is *added*
   alongside existing values.
3. Otherwise the predicate is single-valued and the values conflict. Newer evidence wins unless it
   is much weaker: the candidate *supersedes* the current fact when it was observed at or after
   the current fact and ``c_new + margin >= c_old``. A newer but much weaker candidate is stored
   as *contested* history. An older candidate (out-of-order ingestion) is stored as history
   unless it is more confident by at least ``margin``.

Superseded and contested rows are never deleted by the merge, so the full history of a fact
remains queryable (``AgentMemory.fact_history``).
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_memory.embedding import Embedder
from agent_memory.extraction import CandidateFact, CandidatePreference
from agent_memory.models import Fact, Preference, Status, render_fact
from agent_memory.store import MemoryStore
from agent_memory.text import normalize_value

FACT_IMPORTANCE = 0.8
PREFERENCE_IMPORTANCE = 0.9


@dataclass(frozen=True)
class MergeOutcome:
    fact: Fact
    action: str


@dataclass(frozen=True)
class PreferenceOutcome:
    preference: Preference
    action: str


def combine_confidence(old: float, new: float) -> float:
    return 1.0 - (1.0 - old) * (1.0 - new)


class FactMerger:
    """Applies the reinforcement, addition, and supersession rules above."""

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        multi_valued: frozenset[str],
        margin: float,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.multi_valued = multi_valued
        self.margin = margin

    def _insert(
        self,
        namespace: str,
        candidate: CandidateFact,
        observed_at: float,
        source_episode_id: int | None,
        status: Status = Status.ACTIVE,
        superseded_by: int | None = None,
    ) -> Fact:
        text = render_fact(candidate.subject, candidate.predicate, candidate.object)
        return self.store.insert_fact(
            namespace,
            subject=candidate.subject,
            predicate=candidate.predicate,
            obj=candidate.object,
            text=text,
            confidence=candidate.confidence,
            created_at=observed_at,
            importance=FACT_IMPORTANCE,
            embedding=self.embedder.embed(text),
            source_episode_id=source_episode_id,
            status=status,
            superseded_by=superseded_by,
        )

    def merge(
        self,
        namespace: str,
        candidate: CandidateFact,
        observed_at: float,
        source_episode_id: int | None = None,
    ) -> MergeOutcome:
        existing = self.store.facts_for(namespace, candidate.subject, candidate.predicate)
        active = [fact for fact in existing if fact.status is Status.ACTIVE]
        wanted = normalize_value(candidate.object)
        for fact in active:
            if normalize_value(fact.object) == wanted:
                confidence = combine_confidence(fact.confidence, candidate.confidence)
                updated = self.store.reinforce_fact(namespace, fact.id, confidence, observed_at)
                return MergeOutcome(updated, f"reinforced fact {fact.id}")

        if candidate.predicate in self.multi_valued or not active:
            fact = self._insert(namespace, candidate, observed_at, source_episode_id)
            return MergeOutcome(fact, f"added fact {fact.id}")

        current = max(active, key=lambda fact: (fact.updated_at, fact.id))
        newer = observed_at >= current.updated_at
        if (newer and candidate.confidence + self.margin >= current.confidence) or (
            not newer and candidate.confidence >= current.confidence + self.margin
        ):
            fact = self._insert(namespace, candidate, observed_at, source_episode_id)
            for old in active:
                self.store.set_status(namespace, old.id, Status.SUPERSEDED, superseded_by=fact.id)
            return MergeOutcome(fact, f"fact {fact.id} superseded fact {current.id}")

        status = Status.CONTESTED if newer else Status.SUPERSEDED
        fact = self._insert(
            namespace, candidate, observed_at, source_episode_id, status, superseded_by=current.id
        )
        return MergeOutcome(fact, f"fact {fact.id} kept as {status.value} history of {current.id}")

    def merge_preference(
        self,
        namespace: str,
        candidate: CandidatePreference,
        observed_at: float,
        provenance: str,
        source_episode_id: int | None = None,
    ) -> PreferenceOutcome:
        existing = [
            item
            for item in self.store.preferences_for(namespace, candidate.key)
            if item.status is Status.ACTIVE
        ]
        for item in existing:
            if normalize_value(item.value) == normalize_value(candidate.value):
                return PreferenceOutcome(item, f"preference {item.id} unchanged")
        text = f"preference {candidate.key.replace('_', ' ')}: {candidate.value}"
        preference = self.store.insert_preference(
            namespace,
            key=candidate.key,
            value=candidate.value,
            text=text,
            provenance=provenance,
            created_at=observed_at,
            importance=PREFERENCE_IMPORTANCE,
            embedding=self.embedder.embed(text),
            source_episode_id=source_episode_id,
        )
        for item in existing:
            self.store.set_status(
                namespace, item.id, Status.SUPERSEDED, superseded_by=preference.id
            )
        action = f"preference {preference.id} set"
        if existing:
            action += f", superseding {', '.join(str(item.id) for item in existing)}"
        return PreferenceOutcome(preference, action)

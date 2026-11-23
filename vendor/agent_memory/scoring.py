"""The hybrid retrieval score and its per-result explanations.

For a query ``q`` and a candidate memory ``m`` the score is a weighted sum of four terms, each in
``[0, 1]``::

    score(m, q) = w_lex * L(m, q) + w_sem * S(m, q) + w_rec * R(m) + w_imp * I(m)

* ``L`` (lexical): the FTS5 BM25 score of ``m`` for ``q`` divided by the best BM25 score among
  this query's candidates, so the strongest lexical match scores 1 and non-matches score 0.
* ``S`` (semantic): cosine similarity between the query and memory embeddings, clipped at 0.
* ``R`` (recency): ``2 ** (-age / half_life)`` where ``age`` is the time since the memory's
  evidence was last observed (creation for episodes and preferences, last reinforcement for facts).
* ``I`` (importance): the stored importance; for facts it is multiplied by the fact's confidence.

Candidates come only from the channels with positive weight (FTS5 for ``L``, vector search for
``S``), so recency and importance re-rank relevant memories but never surface unrelated ones.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from agent_memory.embedding import Embedder
from agent_memory.models import Fact, Memory, MemoryKind, RetrievedMemory, ScoreBreakdown, Status
from agent_memory.settings import DAY_SECONDS, ScoringWeights
from agent_memory.store import MemoryStore
from agent_memory.text import content_tokens


@dataclass(frozen=True)
class RetrievalRequest:
    namespace: str
    query: str
    k: int
    now: float
    weights: ScoringWeights
    half_life_days: float
    candidate_pool: int
    min_semantic_similarity: float
    statuses: Sequence[Status]
    kinds: frozenset[MemoryKind] | None = None


def recency(age_seconds: float, half_life_days: float) -> float:
    if half_life_days <= 0:
        return 0.0
    return float(2.0 ** (-max(age_seconds, 0.0) / (half_life_days * DAY_SECONDS)))


def evidence_time(memory: Memory) -> float:
    return memory.updated_at if isinstance(memory, Fact) else memory.created_at


def effective_importance(memory: Memory) -> float:
    if isinstance(memory, Fact):
        return memory.importance * memory.confidence
    return memory.importance


def _stem(token: str) -> str:
    return token[:5] if len(token) > 5 else token


def matched_terms(query: str, text: str) -> tuple[str, ...]:
    text_stems = {_stem(token) for token in content_tokens(text)}
    seen: list[str] = []
    for token in content_tokens(query):
        if _stem(token) in text_stems and token not in seen:
            seen.append(token)
    return tuple(seen)


def _describe(memory: Memory) -> str:
    if isinstance(memory, Fact):
        return f"semantic fact ({memory.status.value}, confidence {memory.confidence:.2f})"
    if memory.kind is MemoryKind.PROCEDURAL:
        return f"procedural preference ({memory.status.value})"
    return "episodic memory"


def explain(memory: Memory, breakdown: ScoreBreakdown, weights: ScoringWeights, age: float) -> str:
    parts = [_describe(memory)]
    if breakdown.matched_terms:
        terms = ", ".join(repr(term) for term in breakdown.matched_terms)
        parts.append(f"lexical match on {terms} (BM25 {breakdown.lexical:.2f} of best)")
    parts.append(f"embedding similarity {breakdown.semantic:.2f}")
    parts.append(f"observed {age / DAY_SECONDS:.1f} days ago (recency {breakdown.recency:.2f})")
    parts.append(f"importance {breakdown.importance:.2f}")
    contributions = {
        "lexical": weights.lexical * breakdown.lexical,
        "semantic": weights.semantic * breakdown.semantic,
        "recency": weights.recency * breakdown.recency,
        "importance": weights.importance * breakdown.importance,
    }
    leader = max(contributions, key=lambda name: contributions[name])
    parts.append(f"largest contribution: {leader} ({contributions[leader]:.2f})")
    return "; ".join(parts)


def score_candidates(
    store: MemoryStore, embedder: Embedder, request: RetrievalRequest
) -> list[RetrievedMemory]:
    """Generate candidates, score them with the hybrid function, and return the top ``k``."""
    weights = request.weights
    use_lexical = weights.lexical > 0
    use_semantic = weights.semantic > 0
    if not use_lexical and not use_semantic:
        use_lexical = use_semantic = True

    lexical_scores: dict[int, float] = {}
    if use_lexical:
        lexical_scores = dict(
            store.lexical_search(
                request.namespace, request.query, request.candidate_pool, request.statuses
            )
        )
    query_vector = embedder.embed(request.query)
    semantic_scores: dict[int, float] = {}
    if use_semantic:
        allowed = store.ids_with_status(request.namespace, request.statuses)
        for memory_id, similarity in store.vector_search(
            request.namespace, query_vector, request.candidate_pool, allowed
        ):
            if similarity >= request.min_semantic_similarity:
                semantic_scores[memory_id] = similarity

    candidate_ids = sorted(set(lexical_scores) | set(semantic_scores))
    memories = store.get_many(request.namespace, candidate_ids)
    best_lexical = max(lexical_scores.values(), default=0.0)

    results: list[RetrievedMemory] = []
    for memory_id in candidate_ids:
        memory = memories.get(memory_id)
        if memory is None or (request.kinds is not None and memory.kind not in request.kinds):
            continue
        lexical = lexical_scores.get(memory_id, 0.0) / best_lexical if best_lexical > 0 else 0.0
        if memory_id in semantic_scores:
            semantic = max(semantic_scores[memory_id], 0.0)
        else:
            stored = store.embedding_of(request.namespace, memory_id)
            semantic = max(float(stored @ query_vector), 0.0) if stored is not None else 0.0
        age = request.now - evidence_time(memory)
        fresh = recency(age, request.half_life_days)
        importance = min(max(effective_importance(memory), 0.0), 1.0)
        total = (
            weights.lexical * lexical
            + weights.semantic * semantic
            + weights.recency * fresh
            + weights.importance * importance
        )
        breakdown = ScoreBreakdown(
            lexical=lexical,
            semantic=semantic,
            recency=fresh,
            importance=importance,
            total=total,
            matched_terms=matched_terms(request.query, memory.text),
        )
        results.append(RetrievedMemory(memory, breakdown, explain(memory, breakdown, weights, age)))

    results.sort(key=lambda item: (-item.score.total, -evidence_time(item.memory), -item.id))
    return results[: request.k]

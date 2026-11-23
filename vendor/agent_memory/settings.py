"""Tunable settings for scoring, fact merging, consolidation, and forgetting."""

from __future__ import annotations

from dataclasses import dataclass, field

DAY_SECONDS = 86_400.0

DEFAULT_MULTI_VALUED: frozenset[str] = frozenset(
    {"allergic_to", "has_pet", "likes", "dislikes", "speaks", "knows"}
)


@dataclass(frozen=True)
class ScoringWeights:
    """Weights of the four terms in the retrieval score (see ``scoring.score_candidates``)."""

    lexical: float = 0.40
    semantic: float = 0.30
    recency: float = 0.15
    importance: float = 0.15

    def without(self, term: str) -> ScoringWeights:
        """Return a copy with one term zeroed, for ablation studies."""
        values = {
            "lexical": self.lexical,
            "semantic": self.semantic,
            "recency": self.recency,
            "importance": self.importance,
        }
        if term not in values:
            raise ValueError(f"unknown scoring term: {term}")
        values[term] = 0.0
        return ScoringWeights(**values)


@dataclass(frozen=True)
class ForgettingSettings:
    """Exponential-decay forgetting applied during consolidation.

    ``retention = importance * 2 ** (-idle / half_life) * (1 + access_boost * ln(1 + accesses))``
    where ``idle`` is the time since the memory was last accessed (or created). Memories whose
    retention falls below ``threshold`` are deleted. Only episodes that were already summarized
    are eligible, so forgetting never discards information that has not been distilled.
    """

    enabled: bool = True
    half_life_days: float = 30.0
    access_boost: float = 0.5
    threshold: float = 0.05
    forget_superseded_facts: bool = False


@dataclass(frozen=True)
class MemorySettings:
    """Top-level settings for an :class:`agent_memory.AgentMemory` instance."""

    weights: ScoringWeights = field(default_factory=ScoringWeights)
    recency_half_life_days: float = 30.0
    candidate_pool: int = 50
    min_semantic_similarity: float = 0.15
    extract_on_write: bool = True
    multi_valued_predicates: frozenset[str] = DEFAULT_MULTI_VALUED
    conflict_margin: float = 0.2
    consolidate_after_days: float = 7.0
    summary_sentences: int = 3
    forgetting: ForgettingSettings = field(default_factory=ForgettingSettings)

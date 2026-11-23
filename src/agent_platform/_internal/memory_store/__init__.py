"""Long-term memory for AI agents.

The public entry point is :class:`AgentMemory`, a namespace-scoped facade over a SQLite store
that supports episodic, semantic, and procedural memory with hybrid retrieval.
"""

from agent_platform._internal.memory_store.embedding import Embedder, HashingEmbedder
from agent_platform._internal.memory_store.extraction import Extractor, LLMExtractor, RuleExtractor
from agent_platform._internal.memory_store.fake_provider import FakeProvider
from agent_platform._internal.memory_store.memory import AgentMemory
from agent_platform._internal.memory_store.models import (
    ConsolidationReport,
    Episode,
    Fact,
    IngestResult,
    MemoryKind,
    Preference,
    RetrievedMemory,
    ScoreBreakdown,
)
from agent_platform._internal.memory_store.provider import AnthropicProvider, Provider
from agent_platform._internal.memory_store.settings import (
    ForgettingSettings,
    MemorySettings,
    ScoringWeights,
)

__version__ = "0.1.1"

__all__ = [
    "AgentMemory",
    "AnthropicProvider",
    "ConsolidationReport",
    "Embedder",
    "Episode",
    "Extractor",
    "Fact",
    "FakeProvider",
    "ForgettingSettings",
    "HashingEmbedder",
    "IngestResult",
    "LLMExtractor",
    "MemoryKind",
    "MemorySettings",
    "Preference",
    "Provider",
    "RetrievedMemory",
    "RuleExtractor",
    "ScoreBreakdown",
    "ScoringWeights",
    "__version__",
]

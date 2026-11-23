"""Long-term memory for AI agents.

The public entry point is :class:`AgentMemory`, a namespace-scoped facade over a SQLite store
that supports episodic, semantic, and procedural memory with hybrid retrieval.
"""

from agent_memory.embedding import Embedder, HashingEmbedder
from agent_memory.extraction import Extractor, LLMExtractor, RuleExtractor
from agent_memory.fake_provider import FakeProvider
from agent_memory.memory import AgentMemory
from agent_memory.models import (
    ConsolidationReport,
    Episode,
    Fact,
    IngestResult,
    MemoryKind,
    Preference,
    RetrievedMemory,
    ScoreBreakdown,
)
from agent_memory.provider import AnthropicProvider, Provider
from agent_memory.settings import ForgettingSettings, MemorySettings, ScoringWeights

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

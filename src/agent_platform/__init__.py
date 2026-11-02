"""Agent platform: an authenticated, multi-tenant AI service composed from small libraries.

The platform wires together a safety gateway, a ReAct agent, hybrid retrieval with an HNSW
vector index, constrained structured output, a durable multi-agent job runner, tracing,
and an evaluation suite behind one HTTP API backed by SQLite.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("agent-platform")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"

__all__ = ["__version__"]

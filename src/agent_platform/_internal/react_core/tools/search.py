"""A deterministic, offline search tool over a small bundled knowledge base.

Real agents call a web or vector search; for a hermetic, reproducible project this tool
searches a fixed in-memory knowledge base. The retrieval is deliberately simple (keyword
overlap) but behaves like a real search tool from the agent's perspective: a query string
in, a short passage out, or a "no results" message.
"""

from __future__ import annotations

import re

_KNOWLEDGE_BASE: dict[str, str] = {
    "france": "France is a country in Western Europe; its capital is Paris.",
    "paris": "Paris is the capital and most populous city of France.",
    "japan": "Japan is an island country in East Asia; its capital is Tokyo.",
    "tokyo": "Tokyo is the capital of Japan and its largest city.",
    "python": "Python is a high-level programming language created by Guido van Rossum.",
    "transformer": "The transformer is a neural network architecture introduced in 2017 "
    "that relies on self-attention.",
    "everest": "Mount Everest is Earth's highest mountain above sea level, at 8,849 metres.",
}

_WORD = re.compile(r"[a-z]+")


class SearchTool:
    """Look up a short factual passage from a bundled knowledge base."""

    name = "search"
    description = (
        "Look up a short fact by keyword, for example 'capital of France'. "
        "Returns a passage from a small built-in knowledge base."
    )

    def __call__(self, tool_input: str) -> str:
        query_words = set(_WORD.findall(tool_input.lower()))
        if not query_words:
            return "No results."

        best_key: str | None = None
        best_overlap = 0
        for key, passage in _KNOWLEDGE_BASE.items():
            passage_words = set(_WORD.findall(passage.lower())) | {key}
            overlap = len(query_words & passage_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_key = key

        if best_key is None:
            return "No results."
        return _KNOWLEDGE_BASE[best_key]

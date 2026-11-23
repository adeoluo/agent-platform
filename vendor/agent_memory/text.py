"""Tokenization and normalization shared by extraction, embedding, and lexical search."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:[+#'][a-z0-9+#]*)?")

STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "about",
        "after",
        "again",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "for",
        "from",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, keeping symbols that matter in names such as ``c++`` and ``c#``."""
    return _TOKEN.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens with stopwords removed."""
    return [token for token in tokenize(text) if token not in STOPWORDS]


def normalize_value(value: str) -> str:
    """Canonical form of an extracted value, used to decide whether two facts agree."""
    cleaned = re.sub(r"\s+", " ", value.strip().strip(".,!?;:\"'")).lower()
    return re.sub(r"^(the|a|an)\s+", "", cleaned)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on terminal punctuation."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]

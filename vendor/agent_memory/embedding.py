"""Embedding interface and a deterministic feature-hashing embedder built on numpy.

The built-in :class:`HashingEmbedder` maps word unigrams, word bigrams, and character trigrams
into a fixed number of signed buckets (the hashing trick) and L2-normalizes the result. It is a
sparse lexical-morphological representation, not a neural sentence encoder: it captures shared
word stems and spelling variants ("prefer", "preferred", "preference") but cannot relate true
synonyms with no surface overlap. Any model that implements :class:`Embedder` can replace it.
"""

from __future__ import annotations

import hashlib
import itertools
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from agent_memory.text import content_tokens

Vector = NDArray[np.float32]


class Embedder(Protocol):
    """Maps text to a fixed-width float32 vector; cosine similarity is used for ranking."""

    @property
    def dimension(self) -> int:
        """Width of the vectors this embedder produces."""

    def embed(self, text: str) -> Vector:
        """Embed ``text`` as a float32 vector, unit norm unless the text has no features."""


def _bucket(feature: str, dimension: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    sign = 1.0 if value & 1 else -1.0
    return (value >> 1) % dimension, sign


class HashingEmbedder:
    """Signed feature hashing over words, word bigrams, and character trigrams."""

    def __init__(
        self,
        dimension: int = 256,
        word_weight: float = 1.0,
        bigram_weight: float = 0.5,
        trigram_weight: float = 0.35,
    ) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self._dimension = dimension
        self._word_weight = word_weight
        self._bigram_weight = bigram_weight
        self._trigram_weight = trigram_weight

    @property
    def dimension(self) -> int:
        return self._dimension

    def features(self, text: str) -> list[tuple[str, float]]:
        tokens = content_tokens(text)
        features: list[tuple[str, float]] = [(f"w:{token}", self._word_weight) for token in tokens]
        features.extend(
            (f"b:{left}_{right}", self._bigram_weight) for left, right in itertools.pairwise(tokens)
        )
        for token in tokens:
            padded = f"#{token}#"
            features.extend(
                (f"c:{padded[i : i + 3]}", self._trigram_weight) for i in range(len(padded) - 2)
            )
        return features

    def embed(self, text: str) -> Vector:
        vector = np.zeros(self._dimension, dtype=np.float32)
        for feature, weight in self.features(text):
            index, sign = _bucket(feature, self._dimension)
            vector[index] += sign * weight
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector


def to_blob(vector: Vector) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> Vector:
    return np.frombuffer(blob, dtype=np.float32).copy()

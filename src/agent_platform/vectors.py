"""A vector retriever: hashed embeddings indexed in an HNSW graph.

:class:`HashingEmbedder` maps text to a fixed-size vector with the feature-hashing trick
over word tokens and character trigrams, so it needs no training and is deterministic
across processes. Trigrams give partial credit to morphological variants ("refund" and
"refunds") that exact-term retrievers miss. :class:`HnswRetriever` indexes those vectors in
the ``ann_vector_search_library`` HNSW graph and satisfies the ``hybrid_rag`` retriever
protocol, so it can be fused with BM25 and TF-IDF by reciprocal rank fusion.

This is a lexical-statistical embedding, not a learned semantic one; it is a stand-in that
keeps the platform offline. A learned embedding model would replace ``HashingEmbedder``
without changing the retriever.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from ann_vector_search_library import HNSW
from hybrid_rag import Chunk, ScoredChunk, tokenize


class HashingEmbedder:
    """Deterministic feature-hashing embeddings over words and character trigrams."""

    def __init__(self, dim: int = 256) -> None:
        if dim < 8:
            raise ValueError("dim must be at least 8")
        self.dim = dim

    def _features(self, text: str) -> list[tuple[str, float]]:
        features: list[tuple[str, float]] = []
        for token in tokenize(text):
            features.append((f"w:{token}", 1.0))
            padded = f"#{token}#"
            for start in range(len(padded) - 2):
                features.append((f"c:{padded[start : start + 3]}", 0.5))
        return features

    def embed(self, text: str) -> list[float]:
        """Return the L2-normalized hashed embedding of ``text``."""
        vector = [0.0] * self.dim
        for feature, weight in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dim
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[index] += sign * weight
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0.0:
            return vector
        return [x / norm for x in vector]


class HnswRetriever:
    """Approximate nearest-neighbor retrieval over hashed chunk embeddings."""

    def __init__(self, embedder: HashingEmbedder | None = None, *, m: int = 8, seed: int = 0):
        self._embedder = embedder or HashingEmbedder()
        self._m = m
        self._seed = seed
        self._chunks: list[Chunk] = []
        self._index = HNSW(dim=self._embedder.dim, m=m, metric="cosine", seed=seed)

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Embed and index ``chunks``, replacing any previous index."""
        self._chunks = list(chunks)
        self._index = HNSW(dim=self._embedder.dim, m=self._m, metric="cosine", seed=self._seed)
        for node_id, chunk in enumerate(self._chunks):
            self._index.add(node_id, self._embedder.embed(chunk.text))

    def retrieve(self, query: str, *, k: int = 10) -> list[ScoredChunk]:
        """Return up to ``k`` chunks ranked by cosine similarity to ``query``."""
        if not self._chunks:
            return []
        vector = self._embedder.embed(query)
        if not any(vector):
            return []
        results: list[ScoredChunk] = []
        for node_id, distance in self._index.search(vector, k):
            similarity = 1.0 - distance
            if similarity > 0.0:
                results.append(ScoredChunk(chunk=self._chunks[node_id], score=similarity))
        return results

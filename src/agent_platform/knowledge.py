"""Per-tenant knowledge bases: durable documents and cached hybrid retrievers.

Documents live in SQLite keyed by ``(tenant, doc_id)``. Each tenant has a monotonically
increasing ``kb_version`` that is bumped in the same transaction as an ingest, so any
process (an API thread or a job worker) can tell when its in-memory index is stale and
rebuild it.

The retriever for a tenant is built only from rows selected with ``WHERE tenant = ?`` and
cached under that tenant's name. There is no shared index across tenants, so one tenant's
query cannot rank another tenant's chunks. As defense in depth, every chunk carries its
tenant in metadata and results are filtered on it before they leave this module.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any

from hybrid_rag import Chunk, Document, ScoredChunk, SentenceChunker, chunk_documents
from hybrid_rag.retrievers import BM25Retriever, HybridRetriever, TfidfRetriever

from agent_platform.db import Database
from agent_platform.vectors import HnswRetriever

MAX_DOCUMENTS_PER_REQUEST = 500
MAX_DOCUMENT_CHARS = 100_000


@dataclass(frozen=True)
class DocumentInput:
    """A document submitted for ingestion."""

    doc_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class IngestResult:
    """The outcome of an ingest call."""

    documents: int
    kb_version: int


@dataclass(frozen=True)
class Passage:
    """A retrieved chunk, with the document it came from."""

    doc_id: str
    chunk_id: str
    text: str
    score: float


def parse_documents(raw: object) -> list[DocumentInput]:
    """Validate a JSON ``documents`` array into :class:`DocumentInput` records."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("'documents' must be a non-empty array")
    if len(raw) > MAX_DOCUMENTS_PER_REQUEST:
        raise ValueError(f"at most {MAX_DOCUMENTS_PER_REQUEST} documents per request")
    documents: list[DocumentInput] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each document must be an object")
        doc_id = item.get("id")
        text = item.get("text")
        metadata = item.get("metadata", {})
        if not isinstance(doc_id, str) or not doc_id.strip() or len(doc_id) > 200:
            raise ValueError("document 'id' must be a non-empty string of at most 200 chars")
        if any(ch in doc_id for ch in "[]#\n"):
            raise ValueError("document 'id' may not contain '[', ']', '#', or newlines")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"document {doc_id!r} needs non-empty 'text'")
        if len(text) > MAX_DOCUMENT_CHARS:
            raise ValueError(f"document {doc_id!r} exceeds {MAX_DOCUMENT_CHARS} characters")
        if not isinstance(metadata, dict):
            raise ValueError(f"document {doc_id!r} 'metadata' must be an object")
        documents.append(DocumentInput(doc_id=doc_id.strip(), text=text, metadata=metadata))
    return documents


class _TenantIndex:
    def __init__(self, tenant: str, version: int, chunks: list[Chunk]) -> None:
        self.tenant = tenant
        self.version = version
        self.chunks = chunks
        self.retriever = HybridRetriever([BM25Retriever(), TfidfRetriever(), HnswRetriever()])
        self.retriever.index(chunks)
        self.lock = threading.Lock()


class KnowledgeBase:
    """Stores tenant documents and serves tenant-scoped hybrid retrieval."""

    def __init__(self, db: Database, *, max_words_per_chunk: int = 60) -> None:
        self._db = db
        self._chunker = SentenceChunker(max_words=max_words_per_chunk)
        self._indexes: dict[str, _TenantIndex] = {}
        self._lock = threading.Lock()

    def ingest(self, tenant: str, documents: list[DocumentInput]) -> IngestResult:
        """Upsert ``documents`` for ``tenant`` and bump its knowledge-base version."""
        now = time.time()
        with self._db.transaction() as conn:
            for doc in documents:
                conn.execute(
                    "INSERT INTO documents (tenant, doc_id, text, metadata, created_at) "
                    "VALUES (?, ?, ?, ?, ?) ON CONFLICT (tenant, doc_id) DO UPDATE SET "
                    "text = excluded.text, metadata = excluded.metadata",
                    (tenant, doc.doc_id, doc.text, json.dumps(doc.metadata), now),
                )
            conn.execute(
                "INSERT INTO tenant_state (tenant, kb_version) VALUES (?, 1) "
                "ON CONFLICT (tenant) DO UPDATE SET kb_version = kb_version + 1",
                (tenant,),
            )
            row = conn.execute(
                "SELECT kb_version FROM tenant_state WHERE tenant = ?", (tenant,)
            ).fetchone()
        return IngestResult(documents=len(documents), kb_version=int(row["kb_version"]))

    def version(self, tenant: str) -> int:
        """Return the tenant's current knowledge-base version (0 if empty)."""
        row = (
            self._db.conn()
            .execute("SELECT kb_version FROM tenant_state WHERE tenant = ?", (tenant,))
            .fetchone()
        )
        return int(row["kb_version"]) if row is not None else 0

    def document_count(self, tenant: str) -> int:
        """Return how many documents ``tenant`` has stored."""
        row = (
            self._db.conn()
            .execute("SELECT COUNT(*) AS n FROM documents WHERE tenant = ?", (tenant,))
            .fetchone()
        )
        return int(row["n"])

    def _load_chunks(self, tenant: str) -> list[Chunk]:
        rows = (
            self._db.conn()
            .execute(
                "SELECT doc_id, text, metadata FROM documents WHERE tenant = ? ORDER BY doc_id",
                (tenant,),
            )
            .fetchall()
        )
        docs = [
            Document(
                id=str(row["doc_id"]),
                text=str(row["text"]),
                metadata={**json.loads(row["metadata"]), "tenant": tenant},
            )
            for row in rows
        ]
        return chunk_documents(docs, self._chunker)

    def _index_for(self, tenant: str) -> _TenantIndex | None:
        version = self.version(tenant)
        if version == 0:
            return None
        with self._lock:
            current = self._indexes.get(tenant)
            if current is not None and current.version == version:
                return current
        rebuilt = _TenantIndex(tenant, version, self._load_chunks(tenant))
        with self._lock:
            current = self._indexes.get(tenant)
            if current is None or current.version < rebuilt.version:
                self._indexes[tenant] = rebuilt
                current = rebuilt
            return current

    def search(self, tenant: str, query: str, *, k: int = 3) -> list[Passage]:
        """Return the top ``k`` passages for ``query`` from ``tenant``'s documents only."""
        index = self._index_for(tenant)
        if index is None or not index.chunks:
            return []
        with index.lock:
            scored: list[ScoredChunk] = index.retriever.retrieve(query, k=k)
        return [
            Passage(
                doc_id=item.chunk.document_id,
                chunk_id=item.chunk.id,
                text=item.chunk.text,
                score=round(item.score, 6),
            )
            for item in scored
            if item.chunk.metadata.get("tenant") == tenant
        ]

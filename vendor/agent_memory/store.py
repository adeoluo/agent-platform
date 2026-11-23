"""SQLite persistence: memories, FTS5 lexical index, vector table, and namespace isolation.

Every public method takes a ``namespace`` and every query filters on it, so one user's or
agent's memories can never be read, updated, or deleted through another namespace.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from agent_memory.embedding import Vector, from_blob, to_blob
from agent_memory.models import Episode, Fact, Memory, MemoryKind, Preference, Status
from agent_memory.text import content_tokens, normalize_value

SCHEMA_VERSION = 1
_T = TypeVar("_T", Episode, Fact, Preference)
_NAMESPACE = re.compile(r"^[A-Za-z0-9_.:@\-]{1,128}$")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at REAL NOT NULL,
    importance REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed REAL,
    status TEXT NOT NULL DEFAULT 'active'
);
CREATE INDEX IF NOT EXISTS ix_memories_namespace ON memories(namespace, kind, status);

CREATE TABLE IF NOT EXISTS episodes (
    memory_id INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    consolidated INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_episodes_session ON episodes(session_id);

CREATE TABLE IF NOT EXISTS facts (
    memory_id INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    object_norm TEXT NOT NULL,
    confidence REAL NOT NULL,
    updated_at REAL NOT NULL,
    evidence_count INTEGER NOT NULL DEFAULT 1,
    source_episode_id INTEGER,
    superseded_by INTEGER
);
CREATE INDEX IF NOT EXISTS ix_facts_key ON facts(subject, predicate);

CREATE TABLE IF NOT EXISTS preferences (
    memory_id INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    provenance TEXT NOT NULL,
    source_episode_id INTEGER,
    superseded_by INTEGER
);
CREATE INDEX IF NOT EXISTS ix_preferences_key ON preferences(key);

CREATE TABLE IF NOT EXISTS vectors (
    memory_id INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    text, content='memories', content_rowid='id', tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS memories_after_insert AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_after_delete AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_after_update AFTER UPDATE OF text ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
END;
"""

_SELECT = """
SELECT m.id, m.namespace, m.kind, m.text, m.created_at, m.importance, m.access_count,
       m.last_accessed, m.status,
       e.session_id, e.role, e.consolidated,
       f.subject, f.predicate, f.object, f.confidence, f.updated_at, f.evidence_count,
       f.source_episode_id, f.superseded_by,
       p.key, p.value, p.provenance, p.source_episode_id, p.superseded_by
FROM memories m
LEFT JOIN episodes e ON e.memory_id = m.id
LEFT JOIN facts f ON f.memory_id = m.id
LEFT JOIN preferences p ON p.memory_id = m.id
"""


def validate_namespace(namespace: str) -> str:
    if not _NAMESPACE.match(namespace):
        raise ValueError(
            "namespace must be 1-128 characters of letters, digits, and _ . : @ -"
            f" (got {namespace!r})"
        )
    return namespace


def fts_query(text: str) -> str | None:
    """Build a safe FTS5 query: content tokens, each quoted, joined with OR."""
    tokens = sorted(set(content_tokens(text)))
    if not tokens:
        return None
    return " OR ".join('"' + token.replace('"', "") + '"' for token in tokens)


def _row_to_memory(row: sqlite3.Row | tuple[Any, ...]) -> Memory:
    (mid, namespace, kind, text, created_at, importance, access_count, last_accessed, status,
     session_id, role, consolidated,
     subject, predicate, obj, confidence, updated_at, evidence_count, f_source, f_superseded,
     key, value, provenance, p_source, p_superseded) = tuple(row)  # fmt: skip
    common: dict[str, Any] = {
        "id": int(mid),
        "namespace": namespace,
        "created_at": float(created_at),
        "importance": float(importance),
        "access_count": int(access_count),
        "last_accessed": last_accessed,
        "status": Status(status),
    }
    if kind == MemoryKind.EPISODIC.value:
        return Episode(
            content=text,
            session_id=session_id,
            role=role,
            consolidated=bool(consolidated),
            **common,
        )
    if kind == MemoryKind.SEMANTIC.value:
        return Fact(
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=float(confidence),
            updated_at=float(updated_at),
            evidence_count=int(evidence_count),
            source_episode_id=f_source,
            superseded_by=f_superseded,
            **common,
        )
    return Preference(
        key=key,
        value=value,
        provenance=provenance,
        source_episode_id=p_source,
        superseded_by=p_superseded,
        **common,
    )


class MemoryStore:
    """Namespace-scoped SQLite storage for all memory kinds."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self.path = str(Path(self.path).expanduser())
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.executescript(_SCHEMA)
        self.connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.connection.commit()
        self._vector_cache: dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = {}

    def close(self) -> None:
        self.connection.close()

    # ------------------------------------------------------------------ writes

    def _insert_memory(
        self,
        namespace: str,
        kind: MemoryKind,
        text: str,
        created_at: float,
        importance: float,
        embedding: Vector | None,
        status: Status = Status.ACTIVE,
    ) -> int:
        validate_namespace(namespace)
        cursor = self.connection.execute(
            "INSERT INTO memories (namespace, kind, text, created_at, importance, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (namespace, kind.value, text, created_at, importance, status.value),
        )
        memory_id = int(cursor.lastrowid or 0)
        if embedding is not None:
            self.connection.execute(
                "INSERT INTO vectors (memory_id, embedding) VALUES (?, ?)",
                (memory_id, to_blob(embedding)),
            )
            self._vector_cache.pop(namespace, None)
        return memory_id

    def insert_episode(
        self,
        namespace: str,
        content: str,
        *,
        session_id: str,
        role: str,
        created_at: float,
        importance: float,
        embedding: Vector | None,
    ) -> Episode:
        memory_id = self._insert_memory(
            namespace, MemoryKind.EPISODIC, content, created_at, importance, embedding
        )
        self.connection.execute(
            "INSERT INTO episodes (memory_id, session_id, role) VALUES (?, ?, ?)",
            (memory_id, session_id, role),
        )
        self.connection.commit()
        return self._require(namespace, memory_id, Episode)

    def insert_fact(
        self,
        namespace: str,
        *,
        subject: str,
        predicate: str,
        obj: str,
        text: str,
        confidence: float,
        created_at: float,
        importance: float,
        embedding: Vector | None,
        source_episode_id: int | None = None,
        status: Status = Status.ACTIVE,
        superseded_by: int | None = None,
    ) -> Fact:
        memory_id = self._insert_memory(
            namespace, MemoryKind.SEMANTIC, text, created_at, importance, embedding, status
        )
        self.connection.execute(
            "INSERT INTO facts (memory_id, subject, predicate, object, object_norm, confidence, "
            "updated_at, source_episode_id, superseded_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                memory_id,
                subject,
                predicate,
                obj,
                normalize_value(obj),
                confidence,
                created_at,
                source_episode_id,
                superseded_by,
            ),
        )
        self.connection.commit()
        return self._require(namespace, memory_id, Fact)

    def insert_preference(
        self,
        namespace: str,
        *,
        key: str,
        value: str,
        text: str,
        provenance: str,
        created_at: float,
        importance: float,
        embedding: Vector | None,
        source_episode_id: int | None = None,
    ) -> Preference:
        memory_id = self._insert_memory(
            namespace, MemoryKind.PROCEDURAL, text, created_at, importance, embedding
        )
        self.connection.execute(
            "INSERT INTO preferences (memory_id, key, value, provenance, source_episode_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (memory_id, key, value, provenance, source_episode_id),
        )
        self.connection.commit()
        return self._require(namespace, memory_id, Preference)

    def set_status(
        self, namespace: str, memory_id: int, status: Status, superseded_by: int | None = None
    ) -> None:
        cursor = self.connection.execute(
            "UPDATE memories SET status = ? WHERE id = ? AND namespace = ?",
            (status.value, memory_id, namespace),
        )
        if cursor.rowcount:
            for table in ("facts", "preferences"):
                self.connection.execute(
                    f"UPDATE {table} SET superseded_by = ? WHERE memory_id = ?",
                    (superseded_by, memory_id),
                )
        self.connection.commit()

    def reinforce_fact(
        self, namespace: str, memory_id: int, confidence: float, updated_at: float
    ) -> Fact:
        self._require(namespace, memory_id, Fact)
        self.connection.execute(
            "UPDATE facts SET confidence = ?, updated_at = MAX(updated_at, ?), "
            "evidence_count = evidence_count + 1 WHERE memory_id = ?",
            (confidence, updated_at, memory_id),
        )
        self.connection.commit()
        return self._require(namespace, memory_id, Fact)

    def touch(self, namespace: str, memory_ids: Iterable[int], now: float) -> None:
        self.connection.executemany(
            "UPDATE memories SET access_count = access_count + 1, last_accessed = ? "
            "WHERE id = ? AND namespace = ?",
            [(now, memory_id, namespace) for memory_id in memory_ids],
        )
        self.connection.commit()

    def mark_consolidated(self, namespace: str, memory_ids: Sequence[int]) -> None:
        self.connection.executemany(
            "UPDATE episodes SET consolidated = 1 WHERE memory_id = "
            "(SELECT id FROM memories WHERE id = ? AND namespace = ?)",
            [(memory_id, namespace) for memory_id in memory_ids],
        )
        self.connection.commit()

    def delete(self, namespace: str, memory_ids: Sequence[int]) -> int:
        cursor = self.connection.executemany(
            "DELETE FROM memories WHERE id = ? AND namespace = ?",
            [(memory_id, namespace) for memory_id in memory_ids],
        )
        self.connection.commit()
        self._vector_cache.pop(namespace, None)
        return max(cursor.rowcount, 0)

    # ------------------------------------------------------------------- reads

    def get(self, namespace: str, memory_id: int) -> Memory | None:
        row = self.connection.execute(
            _SELECT + " WHERE m.id = ? AND m.namespace = ?", (memory_id, namespace)
        ).fetchone()
        return _row_to_memory(row) if row else None

    def _require(self, namespace: str, memory_id: int, kind: type[_T]) -> _T:
        memory = self.get(namespace, memory_id)
        if not isinstance(memory, kind):
            raise KeyError(f"no {kind.__name__.lower()} {memory_id} in namespace {namespace!r}")
        return memory

    def get_many(self, namespace: str, memory_ids: Sequence[int]) -> dict[int, Memory]:
        found: dict[int, Memory] = {}
        for start in range(0, len(memory_ids), 500):
            chunk = list(memory_ids[start : start + 500])
            marks = ",".join("?" * len(chunk))
            rows = self.connection.execute(
                _SELECT + f" WHERE m.namespace = ? AND m.id IN ({marks})", (namespace, *chunk)
            ).fetchall()
            for row in rows:
                memory = _row_to_memory(row)
                found[memory.id] = memory
        return found

    def list_memories(
        self,
        namespace: str,
        *,
        kind: MemoryKind | None = None,
        statuses: Sequence[Status] | None = None,
    ) -> list[Memory]:
        clauses = ["m.namespace = ?"]
        params: list[Any] = [namespace]
        if kind is not None:
            clauses.append("m.kind = ?")
            params.append(kind.value)
        if statuses:
            clauses.append(f"m.status IN ({','.join('?' * len(statuses))})")
            params.extend(status.value for status in statuses)
        rows = self.connection.execute(
            _SELECT + " WHERE " + " AND ".join(clauses) + " ORDER BY m.created_at, m.id", params
        ).fetchall()
        return [_row_to_memory(row) for row in rows]

    def facts_for(self, namespace: str, subject: str, predicate: str) -> list[Fact]:
        rows = self.connection.execute(
            _SELECT + " WHERE m.namespace = ? AND f.subject = ? AND f.predicate = ? "
            "ORDER BY m.created_at, m.id",
            (namespace, subject, predicate),
        ).fetchall()
        return [memory for memory in map(_row_to_memory, rows) if isinstance(memory, Fact)]

    def preferences_for(self, namespace: str, key: str) -> list[Preference]:
        rows = self.connection.execute(
            _SELECT + " WHERE m.namespace = ? AND p.key = ? ORDER BY m.created_at, m.id",
            (namespace, key),
        ).fetchall()
        return [memory for memory in map(_row_to_memory, rows) if isinstance(memory, Preference)]

    def unconsolidated_episodes(self, namespace: str, before: float) -> list[Episode]:
        rows = self.connection.execute(
            _SELECT + " WHERE m.namespace = ? AND m.kind = 'episodic' AND e.consolidated = 0 "
            "AND m.created_at < ? ORDER BY m.created_at, m.id",
            (namespace, before),
        ).fetchall()
        return [memory for memory in map(_row_to_memory, rows) if isinstance(memory, Episode)]

    def lexical_search(
        self, namespace: str, query: str, limit: int, statuses: Sequence[Status]
    ) -> list[tuple[int, float]]:
        """Return ``(memory_id, bm25)`` pairs, higher is better (FTS5 reports negated BM25)."""
        expression = fts_query(query)
        if expression is None:
            return []
        marks = ",".join("?" * len(statuses))
        rows = self.connection.execute(
            "SELECT m.id, -bm25(memories_fts) AS score FROM memories_fts "
            "JOIN memories m ON m.id = memories_fts.rowid "
            f"WHERE memories_fts MATCH ? AND m.namespace = ? AND m.status IN ({marks}) "
            "ORDER BY score DESC LIMIT ?",
            (expression, namespace, *(status.value for status in statuses), limit),
        ).fetchall()
        return [(int(row[0]), float(row[1])) for row in rows]

    def _vectors(self, namespace: str) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        cached = self._vector_cache.get(namespace)
        if cached is not None:
            return cached
        rows = self.connection.execute(
            "SELECT v.memory_id, v.embedding FROM vectors v JOIN memories m ON m.id = v.memory_id "
            "WHERE m.namespace = ? ORDER BY v.memory_id",
            (namespace,),
        ).fetchall()
        if rows:
            ids = np.array([int(row[0]) for row in rows], dtype=np.int64)
            matrix = np.vstack([from_blob(row[1]) for row in rows])
        else:
            ids = np.zeros(0, dtype=np.int64)
            matrix = np.zeros((0, 0), dtype=np.float32)
        self._vector_cache[namespace] = (ids, matrix)
        return ids, matrix

    def vector_search(
        self, namespace: str, query: Vector, limit: int, allowed: set[int] | None = None
    ) -> list[tuple[int, float]]:
        """Brute-force cosine similarity over the namespace's vectors (vectors are unit norm)."""
        ids, matrix = self._vectors(namespace)
        if ids.size == 0 or matrix.shape[1] != query.shape[0]:
            return []
        similarities = matrix @ query
        order = np.argsort(-similarities, kind="stable")
        results: list[tuple[int, float]] = []
        for index in order:
            memory_id = int(ids[index])
            if allowed is not None and memory_id not in allowed:
                continue
            results.append((memory_id, float(similarities[index])))
            if len(results) >= limit:
                break
        return results

    def embedding_of(self, namespace: str, memory_id: int) -> Vector | None:
        row = self.connection.execute(
            "SELECT v.embedding FROM vectors v JOIN memories m ON m.id = v.memory_id "
            "WHERE m.id = ? AND m.namespace = ?",
            (memory_id, namespace),
        ).fetchone()
        return from_blob(row[0]) if row else None

    def ids_with_status(self, namespace: str, statuses: Sequence[Status]) -> set[int]:
        marks = ",".join("?" * len(statuses))
        rows = self.connection.execute(
            f"SELECT id FROM memories WHERE namespace = ? AND status IN ({marks})",
            (namespace, *(status.value for status in statuses)),
        ).fetchall()
        return {int(row[0]) for row in rows}

    # ---------------------------------------------------------------- metadata

    def namespaces(self) -> list[str]:
        rows = self.connection.execute(
            "SELECT DISTINCT namespace FROM memories ORDER BY namespace"
        ).fetchall()
        return [str(row[0]) for row in rows]

    def counts(self, namespace: str) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT kind, status, COUNT(*) FROM memories WHERE namespace = ? GROUP BY kind, status",
            (namespace,),
        ).fetchall()
        counts: dict[str, int] = {}
        for kind, status, count in rows:
            counts[f"{kind}.{status}"] = int(count)
        consolidated = self.connection.execute(
            "SELECT COUNT(*) FROM episodes e JOIN memories m ON m.id = e.memory_id "
            "WHERE m.namespace = ? AND e.consolidated = 1",
            (namespace,),
        ).fetchone()
        counts["episodic.consolidated"] = int(consolidated[0])
        return counts

    def size_bytes(self, *, compact: bool = False) -> int:
        """Database size from page accounting; ``compact`` runs VACUUM first to drop free pages."""
        if compact:
            self.connection.execute("INSERT INTO memories_fts(memories_fts) VALUES ('optimize')")
            self.connection.commit()
            self.connection.execute("VACUUM")
        page_count = self.connection.execute("PRAGMA page_count").fetchone()[0]
        page_size = self.connection.execute("PRAGMA page_size").fetchone()[0]
        return int(page_count) * int(page_size)

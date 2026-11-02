"""SQLite storage: schema, per-thread connections, and explicit write transactions.

Every component (API server threads, worker processes, the CLI) opens its own connection
to the same database file. WAL mode lets readers proceed while one writer commits, and a
busy timeout makes concurrent writers queue instead of failing. Writes that must be atomic
use ``BEGIN IMMEDIATE`` so the write lock is taken up front, which is what makes job claims
and checkpoints race-free across processes.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    key_id      TEXT PRIMARY KEY,
    tenant      TEXT NOT NULL,
    key_hash    TEXT NOT NULL,
    label       TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    revoked_at  REAL
);
CREATE INDEX IF NOT EXISTS api_keys_tenant ON api_keys (tenant);

CREATE TABLE IF NOT EXISTS documents (
    tenant      TEXT NOT NULL,
    doc_id      TEXT NOT NULL,
    text        TEXT NOT NULL,
    metadata    TEXT NOT NULL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (tenant, doc_id)
);

CREATE TABLE IF NOT EXISTS tenant_state (
    tenant      TEXT PRIMARY KEY,
    kb_version  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id        TEXT PRIMARY KEY,
    tenant        TEXT NOT NULL,
    kind          TEXT NOT NULL,
    payload       TEXT NOT NULL,
    status        TEXT NOT NULL,
    board         TEXT,
    result        TEXT,
    error         TEXT,
    trace_id      TEXT NOT NULL,
    attempts      INTEGER NOT NULL DEFAULT 0,
    lease_owner   TEXT,
    lease_expires REAL,
    lease_token   INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    completed_at  REAL
);
CREATE INDEX IF NOT EXISTS jobs_claimable ON jobs (status, created_at);

CREATE TABLE IF NOT EXISTS job_steps (
    job_id       TEXT NOT NULL,
    seq          INTEGER NOT NULL,
    subtask_id   TEXT NOT NULL,
    output       TEXT NOT NULL,
    worker_id    TEXT NOT NULL,
    lease_token  INTEGER NOT NULL,
    completed_at REAL NOT NULL,
    PRIMARY KEY (job_id, subtask_id),
    UNIQUE (job_id, seq)
);

CREATE TABLE IF NOT EXISTS step_attempts (
    attempt_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id       TEXT NOT NULL,
    subtask_id   TEXT NOT NULL,
    worker_id    TEXT NOT NULL,
    lease_token  INTEGER NOT NULL,
    started_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS step_attempts_job ON step_attempts (job_id);

CREATE TABLE IF NOT EXISTS job_claims (
    claim_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id        TEXT NOT NULL,
    worker_id     TEXT NOT NULL,
    lease_token   INTEGER NOT NULL,
    claimed_at    REAL NOT NULL,
    lease_expires REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS job_claims_job ON job_claims (job_id);

CREATE TABLE IF NOT EXISTS spans (
    trace_id    TEXT NOT NULL,
    span_id     TEXT NOT NULL,
    tenant      TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (trace_id, span_id)
);
CREATE INDEX IF NOT EXISTS spans_tenant ON spans (tenant, trace_id);
"""


def connect(path: Path) -> sqlite3.Connection:
    """Open a connection configured for concurrent multi-process use."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


class Database:
    """A database file plus one lazily opened connection per thread."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._local = threading.local()
        self._all: list[sqlite3.Connection] = []
        self._lock = threading.Lock()
        # Threads in one process queue on this lock instead of SQLite's busy handler, whose
        # sleep-and-retry backoff produces long tail latencies under contention. Other
        # processes still coordinate through SQLite's own file lock.
        self._write_lock = threading.Lock()
        self.conn().executescript(SCHEMA)

    def conn(self) -> sqlite3.Connection:
        """Return this thread's connection, opening it on first use."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = connect(self.path)
            self._local.conn = conn
            with self._lock:
                self._all.append(conn)
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a block inside ``BEGIN IMMEDIATE`` and commit, or roll back on error."""
        conn = self.conn()
        with self._write_lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    def ping(self) -> bool:
        """Return whether the database answers a trivial query."""
        try:
            row = self.conn().execute("SELECT 1").fetchone()
        except sqlite3.Error:
            return False
        return row is not None and int(row[0]) == 1

    def close(self) -> None:
        """Close every connection opened through this object."""
        with self._lock:
            for conn in self._all:
                conn.close()
            self._all.clear()
        self._local = threading.local()

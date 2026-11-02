"""Durable job storage: submission, lease-based claiming, checkpoints, and a step ledger.

The protocol, all enforced inside SQLite ``BEGIN IMMEDIATE`` transactions:

* **Claim.** A worker atomically selects the oldest job that is ``queued`` or ``running``
  with an expired lease, and sets ``lease_owner``, ``lease_expires``, and a fresh
  ``lease_token`` (a fencing token incremented on every claim). Because the write lock is
  held from the SELECT to the UPDATE, two workers can never claim the same job at once.
* **Heartbeat.** The owner extends ``lease_expires`` while it works. A worker that dies
  stops renewing, so its lease lapses and the job becomes claimable again.
* **Checkpoint.** After each step the worker writes the step's output to ``job_steps`` and
  the orchestrator's blackboard snapshot to ``jobs.board`` in one transaction, guarded by
  ``lease_owner`` and ``lease_token``. A worker whose lease was taken over cannot record
  anything, so each step is recorded exactly once even though, after a crash between
  executing a step and checkpointing it, the step may be executed twice.
* **Resume.** A new owner restores the blackboard from the last checkpoint and continues
  with the first incomplete step.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

from agent_platform.db import Database

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
STATUSES = (QUEUED, RUNNING, SUCCEEDED, FAILED)


@dataclass(frozen=True)
class Job:
    """A persisted job row."""

    job_id: str
    tenant: str
    kind: str
    payload: dict[str, Any]
    status: str
    board: dict[str, Any] | None
    result: dict[str, Any] | None
    error: str | None
    trace_id: str
    attempts: int
    lease_owner: str | None
    lease_expires: float | None
    lease_token: int
    created_at: float
    updated_at: float
    completed_at: float | None

    def to_dict(self, *, steps_completed: int, steps_total: int | None) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "attempts": self.attempts,
            "steps_completed": steps_completed,
            "steps_total": steps_total,
            "result": self.result,
            "error": self.error,
            "trace_id": self.trace_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
        }


@dataclass(frozen=True)
class Lease:
    """Proof of ownership of a claimed job."""

    job_id: str
    owner: str
    token: int
    expires_at: float


@dataclass(frozen=True)
class StepRecord:
    """One row of the step ledger."""

    job_id: str
    seq: int
    subtask_id: str
    output: str
    worker_id: str
    lease_token: int
    completed_at: float


def _row_to_job(row: Any) -> Job:
    return Job(
        job_id=str(row["job_id"]),
        tenant=str(row["tenant"]),
        kind=str(row["kind"]),
        payload=json.loads(row["payload"]),
        status=str(row["status"]),
        board=json.loads(row["board"]) if row["board"] else None,
        result=json.loads(row["result"]) if row["result"] else None,
        error=row["error"],
        trace_id=str(row["trace_id"]),
        attempts=int(row["attempts"]),
        lease_owner=row["lease_owner"],
        lease_expires=row["lease_expires"],
        lease_token=int(row["lease_token"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        completed_at=row["completed_at"],
    )


class JobStore:
    """All job state transitions, each a single SQLite transaction."""

    def __init__(self, db: Database, *, max_attempts: int = 5) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._db = db
        self._max_attempts = max_attempts

    # -- submission and reads ---------------------------------------------------------- #
    def submit(self, tenant: str, kind: str, payload: dict[str, Any]) -> Job:
        """Persist a new queued job and return it."""
        job_id = f"job-{uuid.uuid4().hex[:16]}"
        now = time.time()
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, tenant, kind, payload, status, trace_id, created_at,"
                " updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (job_id, tenant, kind, json.dumps(payload), QUEUED, f"trace-{job_id}", now, now),
            )
        job = self.get(job_id)
        assert job is not None
        return job

    def get(self, job_id: str, *, tenant: str | None = None) -> Job | None:
        """Return the job, or ``None`` if absent or owned by a different tenant."""
        row = self._db.conn().execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        job = _row_to_job(row)
        if tenant is not None and job.tenant != tenant:
            return None
        return job

    def steps(self, job_id: str) -> list[StepRecord]:
        """Return the step ledger for ``job_id`` in completion order."""
        rows = (
            self._db.conn()
            .execute("SELECT * FROM job_steps WHERE job_id = ? ORDER BY seq", (job_id,))
            .fetchall()
        )
        return [
            StepRecord(
                job_id=str(r["job_id"]),
                seq=int(r["seq"]),
                subtask_id=str(r["subtask_id"]),
                output=str(r["output"]),
                worker_id=str(r["worker_id"]),
                lease_token=int(r["lease_token"]),
                completed_at=float(r["completed_at"]),
            )
            for r in rows
        ]

    def attempts(self, job_id: str) -> list[tuple[str, str, int]]:
        """Return ``(subtask_id, worker_id, lease_token)`` for every step execution start."""
        rows = (
            self._db.conn()
            .execute(
                "SELECT subtask_id, worker_id, lease_token FROM step_attempts WHERE job_id = ? "
                "ORDER BY attempt_id",
                (job_id,),
            )
            .fetchall()
        )
        return [(str(r[0]), str(r[1]), int(r[2])) for r in rows]

    def claims(self, job_id: str) -> list[tuple[str, int, float, float]]:
        """Return ``(worker_id, lease_token, claimed_at, lease_expires)`` per claim."""
        rows = (
            self._db.conn()
            .execute(
                "SELECT worker_id, lease_token, claimed_at, lease_expires FROM job_claims "
                "WHERE job_id = ? ORDER BY claim_id",
                (job_id,),
            )
            .fetchall()
        )
        return [(str(r[0]), int(r[1]), float(r[2]), float(r[3])) for r in rows]

    def counts(self) -> dict[str, int]:
        """Return the number of jobs in each status."""
        rows = (
            self._db.conn().execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall()
        )
        counts = dict.fromkeys(STATUSES, 0)
        for status, n in rows:
            counts[str(status)] = int(n)
        return counts

    # -- the lease protocol -------------------------------------------------------------- #
    def claim(self, owner: str, *, lease_seconds: float) -> tuple[Job, Lease] | None:
        """Atomically claim the next runnable job for ``owner``, or return ``None``."""
        now = time.time()
        with self._db.transaction() as conn:
            while True:
                row = conn.execute(
                    "SELECT job_id, attempts FROM jobs WHERE status = ? OR "
                    "(status = ? AND lease_expires < ?) ORDER BY created_at, job_id LIMIT 1",
                    (QUEUED, RUNNING, now),
                ).fetchone()
                if row is None:
                    return None
                if int(row["attempts"]) >= self._max_attempts:
                    # A job that keeps losing its worker is failed rather than retried forever.
                    conn.execute(
                        "UPDATE jobs SET status = ?, error = ?, lease_owner = NULL, "
                        "updated_at = ?, completed_at = ? WHERE job_id = ?",
                        (FAILED, "exceeded max attempts", now, now, row["job_id"]),
                    )
                    continue
                expires = now + lease_seconds
                conn.execute(
                    "UPDATE jobs SET status = ?, lease_owner = ?, lease_expires = ?, "
                    "lease_token = lease_token + 1, attempts = attempts + 1, updated_at = ? "
                    "WHERE job_id = ?",
                    (RUNNING, owner, expires, now, row["job_id"]),
                )
                job_row = conn.execute(
                    "SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)
                ).fetchone()
                job = _row_to_job(job_row)
                conn.execute(
                    "INSERT INTO job_claims (job_id, worker_id, lease_token, claimed_at, "
                    "lease_expires) VALUES (?, ?, ?, ?, ?)",
                    (job.job_id, owner, job.lease_token, now, expires),
                )
                return job, Lease(job.job_id, owner, job.lease_token, expires)

    def _owns(self, conn: Any, lease: Lease) -> bool:
        row = conn.execute(
            "SELECT 1 FROM jobs WHERE job_id = ? AND status = ? AND lease_owner = ? "
            "AND lease_token = ?",
            (lease.job_id, RUNNING, lease.owner, lease.token),
        ).fetchone()
        return row is not None

    def renew(self, lease: Lease, *, lease_seconds: float) -> bool:
        """Extend the lease; return ``False`` if it is no longer held."""
        now = time.time()
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET lease_expires = ?, updated_at = ? WHERE job_id = ? AND "
                "status = ? AND lease_owner = ? AND lease_token = ?",
                (now + lease_seconds, now, lease.job_id, RUNNING, lease.owner, lease.token),
            )
        return cursor.rowcount == 1

    def save_board(self, lease: Lease, board: dict[str, Any]) -> bool:
        """Persist the initial plan for a freshly claimed job."""
        with self._db.transaction() as conn:
            if not self._owns(conn, lease):
                return False
            conn.execute(
                "UPDATE jobs SET board = ?, updated_at = ? WHERE job_id = ?",
                (json.dumps(board), time.time(), lease.job_id),
            )
        return True

    def record_attempt(self, lease: Lease, subtask_id: str) -> bool:
        """Log that ``subtask_id`` is about to execute under ``lease``."""
        with self._db.transaction() as conn:
            if not self._owns(conn, lease):
                return False
            conn.execute(
                "INSERT INTO step_attempts (job_id, subtask_id, worker_id, lease_token, "
                "started_at) VALUES (?, ?, ?, ?, ?)",
                (lease.job_id, subtask_id, lease.owner, lease.token, time.time()),
            )
        return True

    def checkpoint(
        self,
        lease: Lease,
        subtask_id: str,
        output: str,
        board: dict[str, Any],
        *,
        lease_seconds: float,
    ) -> bool:
        """Record a completed step and the new board atomically, if the lease is still held."""
        now = time.time()
        with self._db.transaction() as conn:
            if not self._owns(conn, lease):
                return False
            seq_row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM job_steps WHERE job_id = ?",
                (lease.job_id,),
            ).fetchone()
            conn.execute(
                "INSERT INTO job_steps (job_id, seq, subtask_id, output, worker_id, lease_token,"
                " completed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (lease.job_id, int(seq_row[0]), subtask_id, output, lease.owner, lease.token, now),
            )
            conn.execute(
                "UPDATE jobs SET board = ?, lease_expires = ?, updated_at = ? WHERE job_id = ?",
                (json.dumps(board), now + lease_seconds, now, lease.job_id),
            )
        return True

    def complete(self, lease: Lease, result: dict[str, Any]) -> bool:
        """Mark the job succeeded with ``result``."""
        return self._finish(lease, SUCCEEDED, result=result, error=None)

    def fail(self, lease: Lease, error: str) -> bool:
        """Mark the job failed with ``error``."""
        return self._finish(lease, FAILED, result=None, error=error)

    def _finish(
        self, lease: Lease, status: str, *, result: dict[str, Any] | None, error: str | None
    ) -> bool:
        now = time.time()
        with self._db.transaction() as conn:
            if not self._owns(conn, lease):
                return False
            conn.execute(
                "UPDATE jobs SET status = ?, result = ?, error = ?, lease_owner = NULL, "
                "lease_expires = NULL, updated_at = ?, completed_at = ? WHERE job_id = ?",
                (
                    status,
                    json.dumps(result) if result is not None else None,
                    error,
                    now,
                    now,
                    lease.job_id,
                ),
            )
        return True

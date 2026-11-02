"""The job worker: claim, heartbeat, execute step by step, checkpoint, finish.

A worker is a loop over :class:`~agent_platform.jobs.JobStore`. For each claimed job it
restores the orchestrator blackboard from the last checkpoint (or plans a fresh one),
executes the first incomplete subtask, and checkpoints the result together with the new
blackboard before moving on. A background heartbeat renews the lease while a step runs.
If the heartbeat finds the lease gone (another worker took over after an expiry), the
worker abandons the job without recording anything further.

Workers hold no state between jobs that matters for correctness, so any number of them can
run in separate processes against the same database file.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from llm_observability import Tracer
from multi_agent_orchestrator import Blackboard

from agent_platform.agent import AgentRuntime
from agent_platform.jobs import Job, JobStore, Lease
from agent_platform.orchestration import build_orchestrator
from agent_platform.tracing import TraceStore


def default_worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"


@dataclass(frozen=True)
class JobOutcome:
    """What happened to a claimed job in this worker."""

    job_id: str
    status: str  # "succeeded", "failed", or "lost_lease"
    steps_executed: int


class _Heartbeat:
    def __init__(self, store: JobStore, lease: Lease, lease_seconds: float) -> None:
        self._store = store
        self._lease = lease
        self._lease_seconds = lease_seconds
        self._stop = threading.Event()
        self.lost = threading.Event()
        self._thread = threading.Thread(target=self._run, name="lease-heartbeat", daemon=True)

    def _run(self) -> None:
        interval = max(self._lease_seconds / 3.0, 0.01)
        while not self._stop.wait(interval):
            if not self._store.renew(self._lease, lease_seconds=self._lease_seconds):
                self.lost.set()
                return

    def __enter__(self) -> _Heartbeat:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()


class JobWorker:
    """Processes jobs from the shared store until stopped."""

    def __init__(
        self,
        store: JobStore,
        runtime: AgentRuntime,
        traces: TraceStore,
        *,
        worker_id: str | None = None,
        lease_seconds: float = 10.0,
        poll_interval: float = 0.2,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.store = store
        self.runtime = runtime
        self.traces = traces
        self.worker_id = worker_id or default_worker_id()
        self.lease_seconds = lease_seconds
        self.poll_interval = poll_interval
        self.stop_event = threading.Event()

    def run_once(self) -> JobOutcome | None:
        """Claim and process one job; return ``None`` if nothing was runnable."""
        claimed = self.store.claim(self.worker_id, lease_seconds=self.lease_seconds)
        if claimed is None:
            return None
        job, lease = claimed
        with _Heartbeat(self.store, lease, self.lease_seconds) as heartbeat:
            return self._process(job, lease, heartbeat)

    def run(self, *, max_jobs: int | None = None, exit_when_idle: bool = False) -> int:
        """Loop until stopped; return the number of jobs processed."""
        processed = 0
        while not self.stop_event.is_set():
            outcome = self.run_once()
            if outcome is None:
                if exit_when_idle:
                    break
                self.stop_event.wait(self.poll_interval)
                continue
            processed += 1
            if max_jobs is not None and processed >= max_jobs:
                break
        return processed

    def _process(self, job: Job, lease: Lease, heartbeat: _Heartbeat) -> JobOutcome:
        tracer = Tracer()
        executed = 0
        try:
            with tracer.span(
                "job.claim", job_id=job.job_id, worker=self.worker_id, lease_token=lease.token
            ):
                pass
            orchestrator = build_orchestrator(self.runtime, job.tenant, job.payload, tracer)
            if job.board is None:
                with tracer.span("job.plan", job_id=job.job_id):
                    board = orchestrator.plan_board(f"research job {job.job_id}")
                if not self.store.save_board(lease, board.snapshot()):
                    return JobOutcome(job.job_id, "lost_lease", executed)
            else:
                board = Blackboard.restore(job.board)
            while not board.all_complete():
                if heartbeat.lost.is_set() or self.stop_event.is_set():
                    return JobOutcome(job.job_id, "lost_lease", executed)
                ready = board.ready_subtasks()
                if not ready:
                    self.store.fail(lease, "no runnable subtask (unsatisfiable plan)")
                    return JobOutcome(job.job_id, "failed", executed)
                subtask = ready[0]
                if not self.store.record_attempt(lease, subtask.id):
                    return JobOutcome(job.job_id, "lost_lease", executed)
                with tracer.span(
                    "job.step", job_id=job.job_id, subtask=subtask.id, worker=self.worker_id
                ) as span:
                    approved = orchestrator.attempt_subtask(board, subtask)
                    span.set_attribute("approved", approved)
                executed += 1
                result = board.result(subtask.id)
                if not approved or result is None:
                    self.store.fail(lease, f"step {subtask.id} was rejected by the critic")
                    return JobOutcome(job.job_id, "failed", executed)
                # Persist spans before the checkpoint: a checkpointed step always has its
                # spans, even if the process dies immediately afterwards.
                self.traces.save_tracer(job.tenant, tracer, trace_id=job.trace_id)
                if not self.store.checkpoint(
                    lease,
                    subtask.id,
                    result.output,
                    board.snapshot(),
                    lease_seconds=self.lease_seconds,
                ):
                    return JobOutcome(job.job_id, "lost_lease", executed)
            final = board.result("report")
            report = json.loads(final.output) if final is not None else {}
            answers = {
                s.id: json.loads(r.output)
                for s in board.subtasks
                if s.id != "report" and (r := board.result(s.id)) is not None
            }
            done = self.store.complete(lease, {"report": report, "answers": answers})
            return JobOutcome(job.job_id, "succeeded" if done else "lost_lease", executed)
        except Exception as error:  # a bug in a step must not kill the worker loop
            self.store.fail(lease, f"{type(error).__name__}: {error}")
            return JobOutcome(job.job_id, "failed", executed)
        finally:
            self.traces.save_tracer(job.tenant, tracer, trace_id=job.trace_id)


def wait_for(predicate: Callable[[], object], *, timeout: float, interval: float = 0.02) -> bool:
    """Poll ``predicate`` until it returns a truthy value or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())

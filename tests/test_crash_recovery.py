"""Cross-process crash recovery.

A real worker process is started, given a multi-step job, and killed with SIGKILL after it
has checkpointed at least one step. A replacement worker process must finish the job from
the last checkpoint. The assertions are made against the SQLite records: the step ledger,
the execution log, and the claim history.
"""

from __future__ import annotations

import os
import signal
from pathlib import Path

import pytest

from agent_platform.app import Platform
from agent_platform.corpus import ACME_DOCUMENTS
from agent_platform.jobs import SUCCEEDED
from agent_platform.knowledge import DocumentInput
from agent_platform.processes import spawn_worker, stop_process
from agent_platform.worker import wait_for
from tests.conftest import make_settings

QUESTIONS = [
    "How long does standard shipping take?",
    "How long is the Lumen warranty?",
    "When is the support team available?",
    "How long are raw sensor readings retained?",
]
LEASE_SECONDS = 1.0


def _platform(path: Path) -> Platform:
    path.mkdir(parents=True, exist_ok=True)
    platform = Platform(make_settings(path, lease_seconds=LEASE_SECONDS))
    platform.knowledge.ingest(
        "acme", [DocumentInput(d["id"], d["text"], {}) for d in ACME_DOCUMENTS]
    )
    return platform


def test_killed_worker_is_replaced_and_the_job_resumes(workdir: Path) -> None:
    # Reference: the same job run without interruption on a separate database.
    reference = _platform(workdir / "reference")
    ref_job = reference.jobs.submit(
        "acme", "research", {"questions": QUESTIONS, "step_delay_seconds": 0.0}
    )
    reference.worker(worker_id="reference").run(exit_when_idle=True)
    expected = reference.jobs.get(ref_job.job_id)
    assert expected is not None and expected.status == SUCCEEDED
    reference.close()

    platform = _platform(workdir / "live")
    db_path = platform.settings.db_path
    job = platform.jobs.submit(
        "acme", "research", {"questions": QUESTIONS, "step_delay_seconds": 0.3}
    )
    first = spawn_worker(
        db_path,
        worker_id="worker-a",
        lease_seconds=LEASE_SECONDS,
        log_path=workdir / "worker-a.log",
    )
    second = None
    try:
        assert wait_for(lambda: len(platform.jobs.steps(job.job_id)) >= 1, timeout=30)
        os.kill(first.pid, signal.SIGKILL)
        assert first.wait(timeout=10) == -signal.SIGKILL
        before = {s.subtask_id: s for s in platform.jobs.steps(job.job_id)}
        assert 1 <= len(before) < len(QUESTIONS) + 1
        assert all(s.worker_id == "worker-a" for s in before.values())
        mid = platform.jobs.get(job.job_id)
        assert mid is not None and mid.status == "running" and mid.lease_owner == "worker-a"

        second = spawn_worker(
            db_path,
            worker_id="worker-b",
            lease_seconds=LEASE_SECONDS,
            log_path=workdir / "worker-b.log",
        )
        assert wait_for(
            lambda: (j := platform.jobs.get(job.job_id)) is not None and j.status == SUCCEEDED,
            timeout=60,
        ), (workdir / "worker-b.log").read_text()
    finally:
        stop_process(first)
        if second is not None:
            stop_process(second)

    done = platform.jobs.get(job.job_id)
    assert done is not None

    # 1. The result is identical to the uninterrupted run.
    assert done.result == expected.result

    # 2. The ledger has every step exactly once, in order, with no gaps.
    steps = platform.jobs.steps(job.job_id)
    assert [s.subtask_id for s in steps] == ["q1", "q2", "q3", "q4", "report"]
    assert [s.seq for s in steps] == [1, 2, 3, 4, 5]

    # 3. Steps checkpointed before the kill were not executed again.
    executions: dict[str, list[str]] = {}
    for subtask_id, worker_id, _ in platform.jobs.attempts(job.job_id):
        executions.setdefault(subtask_id, []).append(worker_id)
    for subtask_id, record in before.items():
        assert executions[subtask_id] == ["worker-a"], subtask_id
        assert record == next(s for s in steps if s.subtask_id == subtask_id)

    # 4. Everything after the checkpoint was recorded by the replacement, under a new token.
    claims = platform.jobs.claims(job.job_id)
    assert [c[0] for c in claims] == ["worker-a", "worker-b"]
    new_token = claims[1][1]
    for record in steps:
        if record.subtask_id not in before:
            assert (record.worker_id, record.lease_token) == ("worker-b", new_token)
    # The replacement only claimed after worker-a's last renewed lease expired.
    assert mid.lease_expires is not None
    assert claims[1][2] >= mid.lease_expires

    # 5. The job trace contains spans from both processes.
    trace = platform.traces.get("acme", done.trace_id)
    assert trace is not None
    writers = {span.attributes.get("worker") for span in trace.spans_named("job.step")}
    assert writers == {"worker-a", "worker-b"}
    platform.close()


@pytest.mark.slow
def test_repeated_kills_still_record_every_step_once(workdir: Path) -> None:
    """Kill three successive workers on one long job; a fourth finishes it."""
    platform = _platform(workdir / "soak")
    questions = QUESTIONS * 3
    job = platform.jobs.submit(
        "acme", "research", {"questions": questions, "step_delay_seconds": 0.3}
    )
    db_path = platform.settings.db_path
    for round_index in range(3):
        before = len(platform.jobs.steps(job.job_id))
        process = spawn_worker(db_path, worker_id=f"w{round_index}", lease_seconds=LEASE_SECONDS)
        assert wait_for(lambda b=before: len(platform.jobs.steps(job.job_id)) > b, timeout=30)
        os.kill(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    final = spawn_worker(db_path, worker_id="w-final", lease_seconds=LEASE_SECONDS)
    try:
        assert wait_for(
            lambda: (j := platform.jobs.get(job.job_id)) is not None and j.status == SUCCEEDED,
            timeout=120,
        )
    finally:
        stop_process(final)
    steps = platform.jobs.steps(job.job_id)
    assert [s.subtask_id for s in steps] == [f"q{i}" for i in range(1, 13)] + ["report"]
    workers = [c[0] for c in platform.jobs.claims(job.job_id)]
    assert workers == ["w0", "w1", "w2", "w-final"]
    executions: dict[str, int] = {}
    for subtask_id, _, _ in platform.jobs.attempts(job.job_id):
        executions[subtask_id] = executions.get(subtask_id, 0) + 1
    # At most the step in flight at each kill can run twice.
    assert sum(n - 1 for n in executions.values()) <= 3
    platform.close()

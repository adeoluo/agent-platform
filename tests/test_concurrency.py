"""Many workers racing for many jobs never double-claim or double-record."""

from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path

from agent_platform.app import Platform
from agent_platform.jobs import SUCCEEDED
from agent_platform.processes import spawn_worker, stop_process
from agent_platform.worker import wait_for
from tests.conftest import make_settings

PAYLOAD = {"questions": ["What is 2 + 3?", "What is 4 * 5?"], "step_delay_seconds": 0.0}


def test_threads_racing_on_claim_get_disjoint_jobs(platform: Platform) -> None:
    job_ids = {platform.jobs.submit("acme", "research", PAYLOAD).job_id for _ in range(60)}
    claimed: list[str] = []
    lock = threading.Lock()

    def race(owner: str) -> None:
        while True:
            result = platform.jobs.claim(owner, lease_seconds=60)
            if result is None:
                return
            with lock:
                claimed.append(result[0].job_id)

    threads = [threading.Thread(target=race, args=(f"t{i}",)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(claimed) == sorted(job_ids)
    assert len(set(claimed)) == len(claimed)


def test_worker_processes_never_double_claim(workdir: Path) -> None:
    platform = Platform(make_settings(workdir, lease_seconds=30.0))
    jobs = [platform.jobs.submit("acme", "research", PAYLOAD).job_id for _ in range(120)]
    workers = [
        spawn_worker(
            platform.settings.db_path,
            worker_id=f"proc-{i}",
            lease_seconds=30.0,
            poll_interval=0.02,
            log_path=workdir / f"proc-{i}.log",
        )
        for i in range(4)
    ]
    try:
        assert wait_for(lambda: platform.jobs.counts()[SUCCEEDED] == len(jobs), timeout=90)
    finally:
        for process in workers:
            stop_process(process)

    claim_rows = platform.db.conn().execute("SELECT job_id, worker_id FROM job_claims").fetchall()
    per_job = Counter(row["job_id"] for row in claim_rows)
    assert set(per_job) == set(jobs)
    assert set(per_job.values()) == {1}, "a job was claimed more than once"

    attempts = (
        platform.db.conn()
        .execute(
            "SELECT job_id, subtask_id, COUNT(*) AS n FROM step_attempts "
            "GROUP BY job_id, subtask_id"
        )
        .fetchall()
    )
    assert len(attempts) == len(jobs) * 3
    assert {row["n"] for row in attempts} == {1}, "a step was executed more than once"

    ledger = platform.db.conn().execute("SELECT COUNT(*) FROM job_steps").fetchone()[0]
    assert ledger == len(jobs) * 3
    # Work was actually spread across processes.
    assert len({row["worker_id"] for row in claim_rows}) >= 2
    platform.close()

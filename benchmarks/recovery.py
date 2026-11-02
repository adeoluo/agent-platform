"""Recovery benchmark: time to recover after a worker is killed mid-job.

For each lease duration, each trial starts a worker process, submits a five-step research
job (four questions and a report, each step taking ``--step-delay`` seconds), waits until
one step is checkpointed, kills the worker with SIGKILL, and immediately starts a
replacement process. It records:

* ``reclaim``: seconds from the kill until the replacement claims the job. The lower bound
  is the time left on the dead worker's lease.
* ``complete``: seconds from the kill until the job succeeds.
* ``re-executed``: checkpointed steps that ran again (expected to be 0).

Usage: uv run python benchmarks/recovery.py [--trials 5] [--leases 0.5,1,2]
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import time

from _common import fresh_dir

from agent_platform.app import Platform
from agent_platform.config import Settings
from agent_platform.corpus import ACME_DOCUMENTS
from agent_platform.knowledge import DocumentInput
from agent_platform.processes import spawn_worker, stop_process
from agent_platform.worker import wait_for

QUESTIONS = [
    "How long does standard shipping take?",
    "How long is the Lumen warranty?",
    "When is the support team available?",
    "How long are raw sensor readings retained?",
]


def trial(lease: float, step_delay: float, index: int) -> dict[str, float]:
    workdir = fresh_dir(f"recovery-{lease}-{index}")
    platform = Platform(Settings(db_path=workdir / "r.db", lease_seconds=lease))
    platform.knowledge.ingest(
        "acme", [DocumentInput(d["id"], d["text"], {}) for d in ACME_DOCUMENTS]
    )
    job = platform.jobs.submit(
        "acme", "research", {"questions": QUESTIONS, "step_delay_seconds": step_delay}
    )
    db_path = platform.settings.db_path
    first = spawn_worker(db_path, worker_id="a", lease_seconds=lease, poll_interval=0.02)
    wait_for(lambda: len(platform.jobs.steps(job.job_id)) >= 1, timeout=60, interval=0.005)
    os.kill(first.pid, signal.SIGKILL)
    killed = time.time()
    first.wait()
    before = {s.subtask_id for s in platform.jobs.steps(job.job_id)}
    second = spawn_worker(db_path, worker_id="b", lease_seconds=lease, poll_interval=0.02)
    try:
        ok = wait_for(
            lambda: (j := platform.jobs.get(job.job_id)) is not None and j.status == "succeeded",
            timeout=120,
            interval=0.005,
        )
    finally:
        stop_process(second)
    assert ok, "job did not complete"
    done = platform.jobs.get(job.job_id)
    assert done is not None and done.completed_at is not None
    claim_b = next(c for c in platform.jobs.claims(job.job_id) if c[0] == "b")
    reexecuted = sum(
        1
        for subtask, worker, _ in platform.jobs.attempts(job.job_id)
        if subtask in before and worker == "b"
    )
    platform.close()
    return {
        "reclaim": claim_b[2] - killed,
        "complete": done.completed_at - killed,
        "reexecuted": float(reexecuted),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--leases", default="0.5,1,2")
    parser.add_argument("--step-delay", type=float, default=0.2)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = []
    for lease in (float(x) for x in args.leases.split(",")):
        results = [trial(lease, args.step_delay, i) for i in range(args.trials)]
        reclaim = [r["reclaim"] for r in results]
        complete = [r["complete"] for r in results]
        rows.append(
            {
                "lease_s": lease,
                "trials": args.trials,
                "reclaim_mean_s": round(statistics.mean(reclaim), 3),
                "reclaim_max_s": round(max(reclaim), 3),
                "complete_mean_s": round(statistics.mean(complete), 3),
                "complete_max_s": round(max(complete), 3),
                "reexecuted_steps": int(sum(r["reexecuted"] for r in results)),
            }
        )
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    print(
        f"{'lease s':>7} {'trials':>6} {'reclaim mean':>12} {'reclaim max':>11} "
        f"{'complete mean':>13} {'complete max':>12} {'re-executed':>11}"
    )
    for r in rows:
        print(
            f"{r['lease_s']:>7} {r['trials']:>6} {r['reclaim_mean_s']:>12} "
            f"{r['reclaim_max_s']:>11} {r['complete_mean_s']:>13} {r['complete_max_s']:>12} "
            f"{r['reexecuted_steps']:>11}"
        )


if __name__ == "__main__":
    main()

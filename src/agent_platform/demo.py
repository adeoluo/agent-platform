"""``agent-platform demo``: an end-to-end scenario on a fresh local database.

The demo starts the API server in-process and a job worker as a child process, then walks
through ingestion, grounded answers, tenant isolation, structured output, authentication
failure, and a multi-step job whose worker is killed with SIGKILL mid-run and replaced. It
prints what happened at each stage, including the step ledger that shows which worker
recorded each step.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent_platform.api import ServerThread
from agent_platform.app import Platform
from agent_platform.client import Client
from agent_platform.config import Settings
from agent_platform.corpus import ACME_DOCUMENTS, GLOBEX_DOCUMENTS
from agent_platform.processes import spawn_worker, stop_process
from agent_platform.worker import wait_for

Printer = Callable[[str], None]


def run_demo(workdir: Path, *, out: Printer = print, lease_seconds: float = 1.5) -> dict[str, Any]:
    """Run the scenario in ``workdir`` (recreated) and return a summary."""
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    db_path = workdir / "demo.db"
    settings = Settings(db_path=db_path, lease_seconds=lease_seconds, poll_interval=0.05)
    platform = Platform(settings)
    summary: dict[str, Any] = {}

    def section(title: str) -> None:
        out("")
        out(f"== {title}")

    acme_key = platform.keys.create("acme", label="demo").secret
    globex_key = platform.keys.create("globex", label="demo").secret

    with ServerThread(platform) as server:
        acme = Client(server.url, acme_key)
        globex = Client(server.url, globex_key)
        section(f"server listening on {server.url}")
        out(f"health: {Client(server.url).request('GET', '/health')[1]}")

        section("ingest (two tenants)")
        out(f"acme:   {acme.ingest(ACME_DOCUMENTS)}")
        out(f"globex: {globex.ingest(GLOBEX_DOCUMENTS)}")

        section("grounded answers for acme")
        for question in (
            "How many days do customers have to request a refund?",
            "How long does the Lumen battery last?",
            "What is 17 * 23?",
        ):
            answer = acme.ask(question)
            out(f"Q: {question}")
            out(f"A: {answer['answer']}  (grounded={answer['grounded']})")
        summary["ask_trace_id"] = answer["trace_id"]

        section("tenant isolation")
        leaked = globex.ask("How many days do customers have to request a refund?")
        out(f"globex asks about acme's refund policy -> {leaked['answer']}")
        out(f"globex sources: {leaked['sources']}")
        summary["isolation_sources"] = leaked["sources"]

        section("structured output (schema-constrained decoding)")
        schema = {
            "properties": [
                {"name": "answer", "type": "string", "max_length": 80},
                {"name": "source", "type": "string", "max_length": 40},
                {"name": "confidence", "type": "enum", "values": ["high", "low"]},
            ]
        }
        structured = acme.ask("How much does express shipping cost?", response_schema=schema)
        out(json.dumps(structured["structured"]))

        section("authentication")
        status, body = Client(server.url, "ap_000000000000_not-a-real-key").request(
            "POST", "/v1/ask", {"question": "hello"}
        )
        out(f"invalid key -> HTTP {status} {body}")

        section("durable job with a worker crash")
        first = spawn_worker(db_path, worker_id="worker-a", lease_seconds=lease_seconds)
        job = acme.submit_job(
            [
                "How long does standard shipping take?",
                "How long is the Lumen warranty?",
                "When is the support team available?",
                "How long are raw sensor readings retained?",
            ],
            step_delay_seconds=0.4,
        )
        job_id = job["job_id"]
        out(f"submitted {job_id}")
        wait_for(lambda: len(platform.jobs.steps(job_id)) >= 2, timeout=30)
        os.kill(first.pid, signal.SIGKILL)
        first.wait()
        killed_at = time.monotonic()
        done_before = [s.subtask_id for s in platform.jobs.steps(job_id)]
        out(f"SIGKILL worker-a after steps {done_before}")
        second = spawn_worker(db_path, worker_id="worker-b", lease_seconds=lease_seconds)
        try:
            wait_for(lambda: acme.job(job_id)["status"] in ("succeeded", "failed"), timeout=60)
        finally:
            stop_process(second)
        recovered_in = time.monotonic() - killed_at
        final = acme.job(job_id)
        out(
            f"status: {final['status']} after {final['attempts']} claims, "
            f"{recovered_in:.2f}s after the kill"
        )
        out("step ledger (seq, subtask, recorded by):")
        for step in final["steps"]:
            out(f"  {step['seq']}  {step['subtask_id']:<7} {step['worker_id']}")
        executions: dict[str, int] = {}
        for subtask_id, _, _ in platform.jobs.attempts(job_id):
            executions[subtask_id] = executions.get(subtask_id, 0) + 1
        out(f"executions per step: {executions}")
        out(f"report: {final['result']['report']['summary']}")
        summary.update(
            job_status=final["status"],
            steps_before_kill=done_before,
            executions=executions,
            recovered_seconds=recovered_in,
        )

        section("job trace (spans from both workers)")
        trace = acme.trace(final["trace_id"])
        workers = sorted({s["attributes"].get("worker", "") for s in trace["spans"]} - {""})
        out(f"{trace['span_count']} spans written by {workers}")

        section("metrics excerpt")
        for line in acme.metrics().splitlines():
            if line.startswith(("agent_platform_jobs{", "agent_platform_answers_total")):
                out(line)
    platform.close()
    return summary

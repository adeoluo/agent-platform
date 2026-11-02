"""Multi-agent jobs built on ``multi_agent_orchestrator``.

A *research* job takes a list of questions. The planner turns each question into a subtask
and adds a final ``report`` subtask that depends on all of them. Each question subtask is
executed by an agent worker that calls :meth:`AgentRuntime.ask` for the job's tenant; the
report worker assembles the answers and their citations. A critic approves a step only if
the worker produced well-formed output.

The orchestrator's blackboard is the job's full state. The platform drives it one subtask
at a time (``Orchestrator.attempt_subtask``) and checkpoints the blackboard snapshot after
each step, which is what makes resumption after a crash possible.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from typing import Any

from llm_observability import Tracer
from multi_agent_orchestrator import Review, Subtask, WorkerResult
from multi_agent_orchestrator.orchestrator import Orchestrator

from agent_platform.agent import AgentRuntime, normalize_question
from agent_platform.model import GatewayBlockedError

RESEARCH = "research"
JOB_KINDS = (RESEARCH,)
MAX_QUESTIONS = 20


def validate_job(
    kind: object, payload: Mapping[str, Any], *, max_step_delay: float
) -> dict[str, Any]:
    """Validate a job submission and return the normalized payload."""
    if kind not in JOB_KINDS:
        raise ValueError(f"'kind' must be one of {list(JOB_KINDS)}")
    questions = payload.get("questions")
    if not isinstance(questions, list) or not questions or len(questions) > MAX_QUESTIONS:
        raise ValueError(f"'questions' must be an array of 1 to {MAX_QUESTIONS} strings")
    normalized = [normalize_question(q) if isinstance(q, str) else "" for q in questions]
    if not all(normalized):
        raise ValueError("every question must be a non-empty string")
    delay = payload.get("step_delay_seconds", 0.0)
    if not isinstance(delay, int | float) or not 0.0 <= float(delay) <= max_step_delay:
        raise ValueError(f"'step_delay_seconds' must be between 0 and {max_step_delay}")
    return {"questions": normalized, "step_delay_seconds": float(delay)}


class ResearchPlanner:
    """Plans one subtask per question plus a dependent report subtask."""

    def __init__(self, questions: Sequence[str]) -> None:
        self._questions = list(questions)

    def plan(self, goal: str) -> Sequence[Subtask]:
        subtasks = [
            Subtask(id=f"q{index}", description=question, metadata={"kind": "ask"})
            for index, question in enumerate(self._questions, start=1)
        ]
        subtasks.append(
            Subtask(
                id="report",
                description=f"Compile a cited report for: {goal}",
                depends_on=tuple(s.id for s in subtasks),
                metadata={"kind": "report"},
            )
        )
        return subtasks


class AgentWorker:
    """Executes question subtasks with the agent and assembles the final report."""

    def __init__(
        self, runtime: AgentRuntime, tenant: str, tracer: Tracer, *, step_delay: float = 0.0
    ) -> None:
        self._runtime = runtime
        self._tenant = tenant
        self._tracer = tracer
        self._step_delay = step_delay

    @property
    def name(self) -> str:
        return "agent-worker"

    def execute(self, subtask: Subtask, context: Mapping[str, WorkerResult]) -> WorkerResult:
        if self._step_delay:
            # Simulates a slow tool or model call; used by tests and the recovery benchmark.
            time.sleep(self._step_delay)
        if subtask.metadata.get("kind") == "report":
            return self._report(subtask, context)
        try:
            result = self._runtime.ask(self._tenant, subtask.description, tracer=self._tracer)
        except GatewayBlockedError as error:
            blocked = {"question": subtask.description, "answer": "", "blocked": error.status}
            return WorkerResult(subtask.id, self.name, json.dumps(blocked), success=False)
        output: dict[str, object] = {
            "question": result.question,
            "answer": result.answer,
            "citations": result.citations,
            "grounded": result.grounded,
        }
        return WorkerResult(subtask.id, self.name, json.dumps(output, sort_keys=True))

    def _report(self, subtask: Subtask, context: Mapping[str, WorkerResult]) -> WorkerResult:
        sections = []
        for dep in subtask.depends_on:
            sections.append(json.loads(context[dep].output))
        grounded = sum(1 for s in sections if s.get("grounded"))
        lines = [f"{i}. {s['question']} -> {s['answer']}" for i, s in enumerate(sections, 1)]
        report = {
            "summary": f"{grounded} of {len(sections)} answers are grounded in cited documents.",
            "lines": lines,
            "citations": sorted({c for s in sections for c in s.get("citations", [])}),
        }
        return WorkerResult(subtask.id, self.name, json.dumps(report, sort_keys=True))


class OutputCritic:
    """Approves a step when the worker succeeded and produced a JSON object."""

    def review(self, subtask: Subtask, result: WorkerResult) -> Review:
        if not result.success:
            return Review(subtask.id, approved=False, feedback="worker reported failure")
        try:
            parsed = json.loads(result.output)
        except json.JSONDecodeError:
            return Review(subtask.id, approved=False, feedback="output is not JSON")
        if not isinstance(parsed, dict):
            return Review(subtask.id, approved=False, feedback="output is not an object")
        return Review(subtask.id, approved=True)


def build_orchestrator(
    runtime: AgentRuntime, tenant: str, payload: Mapping[str, Any], tracer: Tracer
) -> Orchestrator:
    """Assemble the planner, worker, and critic for one research job."""
    return Orchestrator(
        ResearchPlanner(payload["questions"]),
        [AgentWorker(runtime, tenant, tracer, step_delay=float(payload["step_delay_seconds"]))],
        OutputCritic(),
        max_retries=0,
    )

"""Metrics over an orchestrated run.

An orchestration is only improvable if it is measured. :func:`run_metrics` reads a finished
run's blackboard and reports how much work it took: subtask counts, how many attempts the
critic forced (a proxy for wasted effort), which workers were used, and the completion
rate. These are the numbers to watch when tuning a planner, workers, or the critic.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from multi_agent_orchestrator.core.types import Role
from multi_agent_orchestrator.orchestrator import OrchestrationResult


@dataclass(frozen=True, slots=True)
class RunMetrics:
    """Aggregated metrics for one orchestrated run."""

    total_subtasks: int
    completed_subtasks: int
    worker_attempts: int
    retries: int
    critic_rejections: int
    workers_used: tuple[str, ...]

    @property
    def completion_rate(self) -> float:
        if self.total_subtasks == 0:
            return 0.0
        return self.completed_subtasks / self.total_subtasks

    @property
    def attempts_per_subtask(self) -> float:
        if self.total_subtasks == 0:
            return 0.0
        return self.worker_attempts / self.total_subtasks

    def format(self) -> str:
        """Render a one-line summary."""
        return (
            f"{self.completed_subtasks}/{self.total_subtasks} subtasks completed "
            f"({self.completion_rate:.0%}), {self.worker_attempts} attempts, "
            f"{self.retries} retries, {self.critic_rejections} rejections, "
            f"workers: {', '.join(self.workers_used) or 'none'}"
        )


def run_metrics(result: OrchestrationResult) -> RunMetrics:
    """Compute metrics from a finished orchestration result."""
    board = result.blackboard
    worker_messages = [m for m in board.messages if m.role is Role.WORKER]
    worker_attempts = len(worker_messages)
    workers_used = tuple(sorted({m.metadata.get("worker", "") for m in worker_messages} - {""}))
    rejections = sum(
        1 for m in board.messages if m.role is Role.CRITIC and m.content.startswith("rejected")
    )
    total = len(board.subtasks)
    completed = sum(1 for subtask in board.subtasks if board.is_complete(subtask.id))
    # Retries are attempts beyond the first for each subtask that was attempted.
    # Subtracting the subtask count undercounts when some subtasks are never
    # attempted (for example dependents skipped after an upstream failure).
    attempts_per_subtask: Counter[str | None] = Counter(m.subtask_id for m in worker_messages)
    retries = sum(max(0, count - 1) for count in attempts_per_subtask.values())
    return RunMetrics(
        total_subtasks=total,
        completed_subtasks=completed,
        worker_attempts=worker_attempts,
        retries=retries,
        critic_rejections=rejections,
        workers_used=workers_used,
    )

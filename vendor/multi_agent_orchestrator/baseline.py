"""A single-agent baseline for comparison.

The multi-agent design (plan, specialize, review) is only worth its complexity if it beats
doing the whole task in one undifferentiated pass. This module provides that baseline: a
:class:`SingleAgent` that takes the goal and emits one monolithic output, with no
decomposition, specialization, or review. Comparing its output and structure to an
orchestrated run makes the value of coordination concrete.
"""

from __future__ import annotations

from dataclasses import dataclass

from multi_agent_orchestrator.core.agents import Worker
from multi_agent_orchestrator.core.types import Subtask


@dataclass(frozen=True, slots=True)
class SingleAgentResult:
    """The outcome of a single-agent run."""

    goal: str
    output: str
    reviewed: bool = False
    subtasks: int = 1


class SingleAgent:
    """Runs a worker once over the entire goal, with no planning or review."""

    def __init__(self, worker: Worker) -> None:
        self._worker = worker

    def run(self, goal: str) -> SingleAgentResult:
        """Execute the goal in a single undifferentiated pass."""
        subtask = Subtask(id="whole-task", description=goal)
        result = self._worker.execute(subtask, {})
        return SingleAgentResult(goal=goal, output=result.output)


def compare_structure(single_subtasks: int, multi_subtasks: int) -> str:
    """Return a short description contrasting the two approaches' structure."""
    return (
        f"single-agent: {single_subtasks} undifferentiated pass; "
        f"multi-agent: {multi_subtasks} planned, specialized, reviewed subtasks"
    )

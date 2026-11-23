"""Agent roles as protocols.

The three roles are defined structurally so any object with the right shape qualifies -
deterministic stubs for tests, or model-backed implementations in production:

- :class:`Planner` - decomposes a goal into subtasks.
- :class:`Worker` - executes one subtask, optionally seeing prior results.
- :class:`Critic` - reviews a worker's output and approves or rejects it.

Keeping these minimal lets the orchestrator own coordination while the agents own their one
responsibility.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from multi_agent_orchestrator.core.types import (
    Review,
    Subtask,
    WorkerResult,
)


@runtime_checkable
class Planner(Protocol):
    """Decomposes a goal into an ordered set of subtasks."""

    def plan(self, goal: str) -> Sequence[Subtask]:
        """Return the subtasks for ``goal``."""
        ...


@runtime_checkable
class Worker(Protocol):
    """Executes a single subtask."""

    @property
    def name(self) -> str:
        """A short identifier used in results and the audit trail."""
        ...

    def execute(self, subtask: Subtask, context: Mapping[str, WorkerResult]) -> WorkerResult:
        """Produce a result for ``subtask``, given completed dependency results."""
        ...


@runtime_checkable
class Critic(Protocol):
    """Reviews a worker's output for a subtask."""

    def review(self, subtask: Subtask, result: WorkerResult) -> Review:
        """Approve or reject ``result`` for ``subtask``."""
        ...

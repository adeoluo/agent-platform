"""Core types for multi-agent coordination.

The orchestrator decomposes a goal into subtasks, dispatches them to worker agents, and
reviews the results. Four small records carry that work:

- :class:`Subtask` - one unit of work with an id and optional dependencies.
- :class:`Message` - a typed note posted to the shared blackboard (an audit trail).
- :class:`WorkerResult` - a worker's output for a subtask.
- :class:`Review` - a critic's verdict on a worker's output.

All are immutable, so the record of a run is tamper-evident and easy to serialize.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    """Who produced a message."""

    PLANNER = "planner"
    WORKER = "worker"
    CRITIC = "critic"
    ORCHESTRATOR = "orchestrator"


@dataclass(frozen=True, slots=True)
class Subtask:
    """A single unit of work produced by the planner."""

    id: str
    description: str
    depends_on: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Subtask.id must be non-empty")
        if not self.description:
            raise ValueError("Subtask.description must be non-empty")


@dataclass(frozen=True, slots=True)
class Message:
    """A note posted to the blackboard, forming the run's audit trail."""

    role: Role
    content: str
    subtask_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WorkerResult:
    """A worker's output for a subtask."""

    subtask_id: str
    worker: str
    output: str
    success: bool = True


@dataclass(frozen=True, slots=True)
class Review:
    """A critic's verdict on a worker's output."""

    subtask_id: str
    approved: bool
    feedback: str = ""

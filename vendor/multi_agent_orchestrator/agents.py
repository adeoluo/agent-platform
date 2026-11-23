"""Deterministic, offline built-in agents.

These let the whole orchestrator be exercised and demonstrated with no model or network:

- :class:`KeywordPlanner` - splits a goal into subtasks on connective words ("and",
  "then"), optionally chaining them with dependencies.
- :class:`EchoWorker` / :class:`UppercaseWorker` - trivial workers that transform the
  subtask description, standing in for real task execution.
- :class:`LengthCritic` - approves outputs that meet a minimum length, rejecting empty or
  trivial ones so the retry path is exercised.

A model-backed planner, worker, and critic would replace these behind the same protocols.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from multi_agent_orchestrator.core.types import Review, Subtask, WorkerResult

_WHITESPACE = re.compile(r"\s+")
_SPLIT = re.compile(r" (?:and then|then|and|;) ", re.IGNORECASE)


@dataclass(frozen=True)
class KeywordPlanner:
    """Splits a goal into subtasks on connective words."""

    chain_dependencies: bool = False

    def plan(self, goal: str) -> Sequence[Subtask]:
        # Collapse whitespace first so the split runs in linear time on untrusted
        # input (no ambiguous whitespace quantifiers to backtrack over).
        normalized = _WHITESPACE.sub(" ", goal).strip()
        parts = [part.strip() for part in _SPLIT.split(normalized) if part.strip()]
        if not parts:
            parts = [normalized]
        subtasks: list[Subtask] = []
        for index, part in enumerate(parts):
            depends = (f"task-{index - 1}",) if self.chain_dependencies and index > 0 else ()
            subtasks.append(Subtask(id=f"task-{index}", description=part, depends_on=depends))
        return subtasks


@dataclass(frozen=True)
class EchoWorker:
    """Returns the subtask description as its output."""

    name: str = "echo-worker"

    def execute(self, subtask: Subtask, context: Mapping[str, WorkerResult]) -> WorkerResult:
        prefix = ""
        if context:
            prefix = "(given " + ", ".join(sorted(context)) + ") "
        return WorkerResult(
            subtask_id=subtask.id,
            worker=self.name,
            output=f"{prefix}done: {subtask.description}",
        )


@dataclass(frozen=True)
class UppercaseWorker:
    """Returns the subtask description uppercased."""

    name: str = "uppercase-worker"

    def execute(self, subtask: Subtask, context: Mapping[str, WorkerResult]) -> WorkerResult:
        return WorkerResult(
            subtask_id=subtask.id,
            worker=self.name,
            output=subtask.description.upper(),
        )


@dataclass(frozen=True)
class LengthCritic:
    """Approves outputs at least ``min_length`` characters long."""

    min_length: int = 3

    def review(self, subtask: Subtask, result: WorkerResult) -> Review:
        if len(result.output.strip()) >= self.min_length:
            return Review(subtask_id=subtask.id, approved=True)
        return Review(
            subtask_id=subtask.id,
            approved=False,
            feedback="output too short; provide more detail",
        )

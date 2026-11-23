"""The shared blackboard: the single source of truth for a run.

In the blackboard architecture, agents do not call each other directly; they read from and
write to a shared workspace. This decouples the planner, workers, and critic, makes the run
fully inspectable (every post is recorded in order), and gives the orchestrator one place
to check progress and detect being stuck.

The blackboard stores the goal, the planned subtasks, worker results, reviews, and an
ordered message log.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from multi_agent_orchestrator.core.types import (
    Message,
    Review,
    Role,
    Subtask,
    WorkerResult,
)


class Blackboard:
    """A shared, append-mostly workspace coordinating the agents."""

    def __init__(self, goal: str) -> None:
        if not goal:
            raise ValueError("goal must be non-empty")
        self._goal = goal
        self._subtasks: dict[str, Subtask] = {}
        self._results: dict[str, WorkerResult] = {}
        self._reviews: dict[str, Review] = {}
        self._messages: list[Message] = []

    @property
    def goal(self) -> str:
        return self._goal

    # -- messages ----------------------------------------------------------- #
    def post(self, message: Message) -> None:
        """Append a message to the ordered log."""
        self._messages.append(message)

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    # -- subtasks ----------------------------------------------------------- #
    def set_plan(self, subtasks: Sequence[Subtask]) -> None:
        """Record the planner's subtasks, rejecting duplicate ids."""
        seen: dict[str, Subtask] = {}
        for subtask in subtasks:
            if subtask.id in seen:
                raise ValueError(f"duplicate subtask id: {subtask.id}")
            seen[subtask.id] = subtask
        self._subtasks = seen

    @property
    def subtasks(self) -> list[Subtask]:
        return list(self._subtasks.values())

    def subtask(self, subtask_id: str) -> Subtask:
        return self._subtasks[subtask_id]

    # -- results ------------------------------------------------------------ #
    def record_result(self, result: WorkerResult) -> None:
        self._results[result.subtask_id] = result

    def result(self, subtask_id: str) -> WorkerResult | None:
        return self._results.get(subtask_id)

    @property
    def results(self) -> dict[str, WorkerResult]:
        return dict(self._results)

    # -- reviews ------------------------------------------------------------ #
    def record_review(self, review: Review) -> None:
        self._reviews[review.subtask_id] = review

    def review(self, subtask_id: str) -> Review | None:
        return self._reviews.get(subtask_id)

    # -- progress ----------------------------------------------------------- #
    def is_complete(self, subtask_id: str) -> bool:
        """A subtask is complete when it has a result and an approving review."""
        review = self._reviews.get(subtask_id)
        return review is not None and review.approved

    def ready_subtasks(self) -> list[Subtask]:
        """Subtasks whose dependencies are all complete and that are not yet done."""
        ready: list[Subtask] = []
        for subtask in self._subtasks.values():
            if self.is_complete(subtask.id):
                continue
            if all(self.is_complete(dep) for dep in subtask.depends_on):
                ready.append(subtask)
        return ready

    def all_complete(self) -> bool:
        """True when every planned subtask is complete."""
        return bool(self._subtasks) and all(self.is_complete(sid) for sid in self._subtasks)

    # -- durable serialization --------------------------------------------- #
    def snapshot(self) -> dict[str, Any]:
        """Serialize the entire board (plan, results, reviews, log) to a JSON-ready dict."""
        return {
            "goal": self._goal,
            "subtasks": [
                {
                    "id": subtask.id,
                    "description": subtask.description,
                    "depends_on": list(subtask.depends_on),
                    "metadata": subtask.metadata,
                }
                for subtask in self._subtasks.values()
            ],
            "results": [
                {
                    "subtask_id": result.subtask_id,
                    "worker": result.worker,
                    "output": result.output,
                    "success": result.success,
                }
                for result in self._results.values()
            ],
            "reviews": [
                {
                    "subtask_id": review.subtask_id,
                    "approved": review.approved,
                    "feedback": review.feedback,
                }
                for review in self._reviews.values()
            ],
            "messages": [
                {
                    "role": message.role.value,
                    "content": message.content,
                    "subtask_id": message.subtask_id,
                    "metadata": message.metadata,
                }
                for message in self._messages
            ],
        }

    @classmethod
    def restore(cls, snapshot: dict[str, Any]) -> Blackboard:
        """Rebuild a board from :meth:`snapshot`, preserving plan, progress, and log order."""
        board = cls(snapshot["goal"])
        board.set_plan(
            [
                Subtask(
                    id=entry["id"],
                    description=entry["description"],
                    depends_on=tuple(entry.get("depends_on", ())),
                    metadata=entry.get("metadata") or {},
                )
                for entry in snapshot["subtasks"]
            ]
        )
        for entry in snapshot["results"]:
            board.record_result(
                WorkerResult(
                    subtask_id=entry["subtask_id"],
                    worker=entry["worker"],
                    output=entry["output"],
                    success=entry.get("success", True),
                )
            )
        for entry in snapshot["reviews"]:
            board.record_review(
                Review(
                    subtask_id=entry["subtask_id"],
                    approved=entry["approved"],
                    feedback=entry.get("feedback", ""),
                )
            )
        board._messages = [
            Message(
                role=Role(entry["role"]),
                content=entry["content"],
                subtask_id=entry.get("subtask_id"),
                metadata=entry.get("metadata") or {},
            )
            for entry in snapshot["messages"]
        ]
        return board

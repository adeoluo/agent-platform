"""The orchestrator: the coordination loop over planner, workers, and critic.

The orchestrator runs the blackboard control loop:

1. Ask the **planner** to decompose the goal into subtasks and record the plan.
2. Repeatedly pick a **ready** subtask (dependencies complete), dispatch it to a
   **worker**, and have the **critic** review the result.
3. On approval, mark it complete; on rejection, retry up to a limit (feeding the critic's
   feedback back as context), then give up on that subtask.
4. Stop when every subtask is complete, or when no progress can be made (a guard against
   cycles or persistent rejection).

The orchestrator never does the work itself - it only schedules, dispatches, and records,
so the agents stay single-purpose and the whole run is an auditable blackboard history.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

from multi_agent_orchestrator.core.agents import Critic, Planner, Worker
from multi_agent_orchestrator.core.blackboard import Blackboard
from multi_agent_orchestrator.core.types import Message, Role, Subtask, WorkerResult


@dataclass(frozen=True, slots=True)
class OrchestrationResult:
    """The outcome of an orchestrated run."""

    goal: str
    blackboard: Blackboard
    completed: bool

    @property
    def num_subtasks(self) -> int:
        return len(self.blackboard.subtasks)

    def final_outputs(self) -> dict[str, str]:
        """The approved outputs, keyed by subtask id, in plan order."""
        outputs: dict[str, str] = {}
        for subtask in self.blackboard.subtasks:
            if self.blackboard.is_complete(subtask.id):
                result = self.blackboard.result(subtask.id)
                if result is not None:
                    outputs[subtask.id] = result.output
        return outputs


class Orchestrator:
    """Coordinates a planner, a pool of workers, and a critic over a blackboard."""

    def __init__(
        self,
        planner: Planner,
        workers: list[Worker],
        critic: Critic,
        *,
        max_retries: int = 2,
        validate_plan: bool = True,
    ) -> None:
        if not workers:
            raise ValueError("at least one worker is required")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self._planner = planner
        self._workers = list(workers)
        self._critic = critic
        self._max_retries = max_retries
        self._validate_plan = validate_plan

    def run(self, goal: str) -> OrchestrationResult:
        """Orchestrate the agents to accomplish ``goal``."""
        board = self.plan_board(goal)
        self.drive(board)
        return OrchestrationResult(goal=goal, blackboard=board, completed=board.all_complete())

    def plan_board(self, goal: str) -> Blackboard:
        """Create a board, ask the planner to decompose ``goal``, and record the plan."""
        board = Blackboard(goal)
        subtasks = self._planner.plan(goal)
        if self._validate_plan:
            from multi_agent_orchestrator.graph import validate_plan

            validate_plan(subtasks)
        board.set_plan(subtasks)
        board.post(Message(role=Role.PLANNER, content=f"planned {len(board.subtasks)} subtasks"))
        return board

    def drive(self, board: Blackboard) -> None:
        """Drive a (possibly partially completed) board to completion or a stuck state."""
        # Bound total iterations to guarantee termination even on pathological plans.
        max_iterations = len(board.subtasks) * (self._max_retries + 1) + 1
        for _ in range(max_iterations):
            if board.all_complete():
                break
            ready = board.ready_subtasks()
            if not ready:
                break  # no progress possible (unmet dependencies or exhausted retries)
            progressed = False
            for subtask in ready:
                if self.attempt_subtask(board, subtask):
                    progressed = True
            if not progressed:
                break

    def attempt_subtask(self, board: Blackboard, subtask: Subtask) -> bool:
        """Attempt one subtask (with bounded retries); return whether it was approved."""
        return self._attempt(board, subtask)

    def _attempt(self, board: Blackboard, subtask: Subtask) -> bool:
        worker = self._select_worker(subtask)
        context = {
            dep: result for dep in subtask.depends_on if (result := board.result(dep)) is not None
        }

        for attempt in range(self._max_retries + 1):
            result = worker.execute(subtask, context)
            board.record_result(result)
            board.post(
                Message(
                    role=Role.WORKER,
                    content=result.output,
                    subtask_id=subtask.id,
                    metadata={"worker": worker.name, "attempt": attempt},
                )
            )
            review = self._critic.review(subtask, result)
            board.record_review(review)
            board.post(
                Message(
                    role=Role.CRITIC,
                    content=("approved" if review.approved else f"rejected: {review.feedback}"),
                    subtask_id=subtask.id,
                )
            )
            if review.approved:
                return True
        return False  # exhausted retries without approval

    def _select_worker(self, subtask: Subtask) -> Worker:
        # Round-robin by a stable hash of the subtask id. Python's built-in hash
        # of a string is salted per process, so it cannot be used for dispatch
        # that must be deterministic and reproducible across runs; crc32 is stable.
        index = zlib.crc32(subtask.id.encode("utf-8")) % len(self._workers)
        return self._workers[index]


__all__ = ["OrchestrationResult", "Orchestrator", "WorkerResult"]

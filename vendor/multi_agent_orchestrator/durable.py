"""Durable orchestration: checkpoint the blackboard and resume an interrupted run.

An orchestration is a sequence of subtask completions recorded on the blackboard. Because
the blackboard *is* the entire run state -- plan, results, reviews, and the message log --
checkpointing it after every completed subtask makes the run durable: if the process dies,
the checkpoint on disk is a consistent prefix of the work, and a resumed run picks up from
the last completed subtask.

:func:`run_durable` drives the orchestrator one subtask at a time, writing a checkpoint after
each, and :func:`resume_durable` restores the board and finishes the remaining subtasks.
Because the agents are deterministic and scheduling is stable, a resumed run completes to a
state identical to one that was never interrupted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from multi_agent_orchestrator.core.blackboard import Blackboard
from multi_agent_orchestrator.orchestrator import OrchestrationResult, Orchestrator


class CrashError(RuntimeError):
    """Raised by :func:`run_durable` to simulate an interruption after some subtasks complete."""


@dataclass(frozen=True)
class CheckpointStatus:
    """A report on a checkpoint found on disk."""

    goal: str
    completed: int
    total: int

    @property
    def complete(self) -> bool:
        return self.total > 0 and self.completed >= self.total

    @property
    def interrupted(self) -> bool:
        return 0 <= self.completed < self.total

    def format(self) -> str:
        state = "complete" if self.complete else "interrupted"
        return f"{state}: {self.completed}/{self.total} subtasks checkpointed"


def _completed_count(board: Blackboard) -> int:
    return sum(1 for subtask in board.subtasks if board.is_complete(subtask.id))


def save_checkpoint(path: Path, board: Blackboard) -> None:
    """Write the board's full state to ``path`` as JSON."""
    path.write_text(json.dumps(board.snapshot()), encoding="utf-8")


def load_checkpoint(path: Path) -> Blackboard:
    """Restore a board from a checkpoint written by :func:`save_checkpoint`."""
    return Blackboard.restore(json.loads(path.read_text(encoding="utf-8")))


def inspect_checkpoint(path: Path) -> CheckpointStatus:
    """Report whether a checkpoint on disk is complete or interrupted."""
    board = load_checkpoint(path)
    return CheckpointStatus(
        goal=board.goal,
        completed=_completed_count(board),
        total=len(board.subtasks),
    )


def _drive_with_checkpoints(
    orchestrator: Orchestrator,
    board: Blackboard,
    checkpoint_path: Path,
    crash_after: int | None,
) -> None:
    """Complete ready subtasks one at a time, checkpointing after each, honoring a crash point."""
    attempted = 0
    guard = len(board.subtasks) * 8 + 1
    while attempted < guard:
        if board.all_complete():
            break
        ready = board.ready_subtasks()
        if not ready:
            break
        subtask = ready[0]
        orchestrator.attempt_subtask(board, subtask)
        save_checkpoint(checkpoint_path, board)
        attempted += 1
        if crash_after is not None and attempted >= crash_after:
            raise CrashError(
                f"interrupted after {_completed_count(board)} of {len(board.subtasks)} subtasks"
            )


def run_durable(
    orchestrator: Orchestrator,
    goal: str,
    checkpoint_path: Path,
    *,
    crash_after: int | None = None,
) -> OrchestrationResult:
    """Run ``goal`` durably, checkpointing after each completed subtask.

    If ``crash_after`` is set, raise :class:`CrashError` once that many subtasks have been
    attempted, leaving a consistent checkpoint behind for :func:`resume_durable`.
    """
    board = orchestrator.plan_board(goal)
    save_checkpoint(checkpoint_path, board)
    _drive_with_checkpoints(orchestrator, board, checkpoint_path, crash_after)
    return OrchestrationResult(goal=goal, blackboard=board, completed=board.all_complete())


def resume_durable(orchestrator: Orchestrator, checkpoint_path: Path) -> OrchestrationResult:
    """Resume an interrupted run from its checkpoint and finish the remaining subtasks."""
    board = load_checkpoint(checkpoint_path)
    _drive_with_checkpoints(orchestrator, board, checkpoint_path, crash_after=None)
    return OrchestrationResult(goal=board.goal, blackboard=board, completed=board.all_complete())

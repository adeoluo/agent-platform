"""An honest single-agent vs multi-agent comparison over multi-step workloads.

The multi-agent pattern (plan, specialize, review) is only worth its complexity if it buys
something a single undifferentiated pass does not. This module runs the *same* goals through
both and measures the trade-off with real counts, not assertions:

- **worker calls** -- the cost. A single agent makes one call; the orchestrator makes one per
  subtask (plus any critic-forced retries).
- **raw coverage** -- the fraction of a goal's parts whose text appears *somewhere* in the
  produced output. A single pass that echoes the whole goal already mentions every part, so on
  these decomposable goals raw coverage is typically a tie.
- **verified coverage** -- the fraction of parts delivered as an *independently reviewed*,
  approved unit. The single agent has no critic, so none of its output is verified; the
  orchestrator reviews every subtask.

The honest finding this surfaces: on simple goals coordination does **not** improve raw
coverage and costs more worker calls; its value is verification and per-part reliability.
"""

from __future__ import annotations

from dataclasses import dataclass

from multi_agent_orchestrator.agents import EchoWorker, KeywordPlanner, LengthCritic
from multi_agent_orchestrator.baseline import SingleAgent
from multi_agent_orchestrator.core.types import Role
from multi_agent_orchestrator.orchestrator import OrchestrationResult, Orchestrator


def _parts(goal: str) -> list[str]:
    """The deliverable parts of a goal, as the keyword planner would split them."""
    return [subtask.description for subtask in KeywordPlanner().plan(goal)]


def _coverage(parts: list[str], text: str) -> float:
    if not parts:
        return 1.0
    covered = sum(1 for part in parts if part.lower() in text.lower())
    return covered / len(parts)


@dataclass(frozen=True)
class ApproachMetrics:
    """Measured outcome of one approach on one goal."""

    worker_calls: int
    outputs: int
    reviewed_deliverables: int
    raw_coverage: float
    verified_coverage: float
    completed: bool


@dataclass(frozen=True)
class Comparison:
    """A single-agent vs multi-agent comparison on one goal."""

    goal: str
    parts: int
    single: ApproachMetrics
    multi: ApproachMetrics


def _single_metrics(goal: str, parts: list[str]) -> ApproachMetrics:
    result = SingleAgent(EchoWorker()).run(goal)
    raw = _coverage(parts, result.output)
    return ApproachMetrics(
        worker_calls=1,
        outputs=1,
        reviewed_deliverables=0,  # the baseline has no critic
        raw_coverage=raw,
        verified_coverage=0.0,
        completed=bool(result.output),
    )


def _multi_metrics(result: OrchestrationResult, parts: list[str]) -> ApproachMetrics:
    board = result.blackboard
    worker_calls = sum(1 for message in board.messages if message.role is Role.WORKER)
    approved = [s for s in board.subtasks if board.is_complete(s.id)]
    combined = " ".join(r.output for r in board.results.values() if board.is_complete(r.subtask_id))
    return ApproachMetrics(
        worker_calls=worker_calls,
        outputs=len(board.results),
        reviewed_deliverables=len(approved),
        raw_coverage=_coverage(parts, combined),
        verified_coverage=len(approved) / len(parts) if parts else 1.0,
        completed=result.completed,
    )


def _orchestrator() -> Orchestrator:
    return Orchestrator(KeywordPlanner(chain_dependencies=True), [EchoWorker()], LengthCritic())


def compare_goal(goal: str) -> Comparison:
    """Run one goal through both approaches and measure the trade-off."""
    parts = _parts(goal)
    multi_result = _orchestrator().run(goal)
    return Comparison(
        goal=goal,
        parts=len(parts),
        single=_single_metrics(goal, parts),
        multi=_multi_metrics(multi_result, parts),
    )


@dataclass(frozen=True)
class ComparisonSummary:
    """Aggregate of a suite of comparisons."""

    rows: tuple[Comparison, ...]

    @property
    def single_worker_calls(self) -> int:
        return sum(row.single.worker_calls for row in self.rows)

    @property
    def multi_worker_calls(self) -> int:
        return sum(row.multi.worker_calls for row in self.rows)

    @property
    def single_verified(self) -> int:
        return sum(row.single.reviewed_deliverables for row in self.rows)

    @property
    def multi_verified(self) -> int:
        return sum(row.multi.reviewed_deliverables for row in self.rows)

    @property
    def mean_single_raw_coverage(self) -> float:
        return sum(row.single.raw_coverage for row in self.rows) / len(self.rows)

    @property
    def mean_multi_raw_coverage(self) -> float:
        return sum(row.multi.raw_coverage for row in self.rows) / len(self.rows)

    @property
    def cost_ratio(self) -> float:
        if self.single_worker_calls == 0:
            return float("inf")
        return self.multi_worker_calls / self.single_worker_calls

    def verdict(self) -> str:
        """An honest one-paragraph reading of the aggregate numbers."""
        raw_tie = abs(self.mean_single_raw_coverage - self.mean_multi_raw_coverage) < 1e-9
        raw = (
            "identical raw coverage"
            if raw_tie
            else (
                f"raw coverage {self.mean_single_raw_coverage:.2f} vs "
                f"{self.mean_multi_raw_coverage:.2f}"
            )
        )
        return (
            f"Across {len(self.rows)} goals the two approaches have {raw}; coordination does not "
            f"add raw content on these decomposable goals. Its payoff is verification: the "
            f"multi-agent runs delivered {self.multi_verified} independently reviewed sub-"
            f"deliverables versus {self.single_verified} for the single agent, at "
            f"{self.cost_ratio:.1f}x the worker calls "
            f"({self.multi_worker_calls} vs {self.single_worker_calls}). On single-part goals the "
            f"two are equivalent and coordination is pure overhead."
        )


def compare_suite(goals: list[str]) -> ComparisonSummary:
    """Run a suite of goals through both approaches and aggregate the results."""
    return ComparisonSummary(rows=tuple(compare_goal(goal) for goal in goals))


DEFAULT_SUITE: tuple[str, ...] = (
    "write the report",
    "research the topic and then write the report",
    "research the topic and then draft an outline and then write the report",
    "gather sources and then research the topic and then draft an outline "
    "and then write the report and then proofread it",
    "define scope and then gather sources and then research the topic and then "
    "draft an outline and then write the report and then proofread it "
    "and then format the output and then publish it",
)

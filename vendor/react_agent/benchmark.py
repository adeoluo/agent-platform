"""A benchmark harness for measuring agent task success.

A single demonstration proves an agent *can* work; a benchmark measures *how often* it
does. This module defines a :class:`BenchmarkTask` (a question plus a way to judge the
final answer) and a :func:`run_benchmark` function that runs an agent over a suite and
reports the success rate with per-task outcomes.

Judging reuses simple, transparent matchers so the benchmark is deterministic and offline.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from react_agent.core.agent import ReActAgent
from react_agent.core.steps import AgentResult

Judge = Callable[[str], bool]


def contains(expected: str) -> Judge:
    """A judge that passes when the answer contains ``expected`` (case-insensitive)."""
    needle = expected.lower()
    return lambda answer: needle in answer.lower()


def equals(expected: str) -> Judge:
    """A judge that passes on a normalized exact match."""
    target = expected.strip().lower()
    return lambda answer: answer.strip().lower() == target


def every(*expected: str) -> Judge:
    """A judge that passes only when the answer contains *all* of ``expected``.

    Used for multi-part questions, where a correct answer must include every sub-answer.
    """
    needles = tuple(part.lower() for part in expected)
    return lambda answer: all(needle in answer.lower() for needle in needles)


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    """One benchmark item: a question and a judge for the agent's final answer."""

    name: str
    question: str
    judge: Judge
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    """The result of running the agent on one benchmark task."""

    task_name: str
    passed: bool
    finished: bool
    steps: int
    answer: str


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Aggregated benchmark results."""

    outcomes: tuple[TaskOutcome, ...] = field(default_factory=tuple)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def passed(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.passed)

    @property
    def success_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def mean_steps(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(outcome.steps for outcome in self.outcomes) / len(self.outcomes)

    def summary(self) -> str:
        """Render a one-line-per-task summary with a success-rate header."""
        lines = [
            f"Benchmark: {self.passed}/{self.total} passed "
            f"({self.success_rate * 100:.1f}%), mean {self.mean_steps:.1f} steps"
        ]
        for outcome in self.outcomes:
            mark = "PASS" if outcome.passed else "FAIL"
            lines.append(f"  [{mark}] {outcome.task_name} ({outcome.steps} steps)")
        return "\n".join(lines)


def run_benchmark(agent: ReActAgent, tasks: Sequence[BenchmarkTask]) -> BenchmarkReport:
    """Run ``agent`` over ``tasks`` and return a :class:`BenchmarkReport`."""
    outcomes: list[TaskOutcome] = []
    for task in tasks:
        result: AgentResult = agent.run(task.question)
        passed = result.finished and task.judge(result.answer)
        outcomes.append(
            TaskOutcome(
                task_name=task.name,
                passed=passed,
                finished=result.finished,
                steps=result.step_count,
                answer=result.answer,
            )
        )
    return BenchmarkReport(outcomes=tuple(outcomes))


@dataclass(frozen=True, slots=True)
class BaselineComparison:
    """A ReAct report alongside a single-shot baseline run on the same tasks.

    The baseline is the *same* model and tools given a one-step budget, i.e. a single model
    turn with no observe-then-replan loop. Comparing the two isolates the value of the loop
    itself: tasks that need a tool observation can only be solved when the agent is allowed
    to act, read the result, and continue.
    """

    react: BenchmarkReport
    baseline: BenchmarkReport

    @property
    def lift(self) -> float:
        """How much the ReAct loop improves the success rate over the baseline."""
        return self.react.success_rate - self.baseline.success_rate

    def summary(self) -> str:
        """Render a comparison of the two success rates and the loop's lift."""
        react = self.react
        baseline = self.baseline
        return (
            "Loop vs. single-shot baseline:\n"
            f"  ReAct loop     : {react.passed}/{react.total} passed "
            f"({react.success_rate * 100:.1f}%), mean {react.mean_steps:.1f} steps\n"
            f"  single-shot    : {baseline.passed}/{baseline.total} passed "
            f"({baseline.success_rate * 100:.1f}%), mean {baseline.mean_steps:.1f} steps\n"
            f"  loop lift      : {self.lift * 100:+.1f} percentage points"
        )


def compare_to_single_shot(
    agent: ReActAgent,
    baseline_agent: ReActAgent,
    tasks: Sequence[BenchmarkTask],
) -> BaselineComparison:
    """Benchmark the ReAct ``agent`` and a one-step ``baseline_agent`` on the same tasks."""
    return BaselineComparison(
        react=run_benchmark(agent, tasks),
        baseline=run_benchmark(baseline_agent, tasks),
    )

"""Consensus coordination: many workers answer the same task, then aggregate.

The orchestrator is a *division-of-labor* pattern (different workers, different subtasks).
Consensus is the complementary *redundancy* pattern: several workers answer the **same**
task and their outputs are combined, which improves reliability on ambiguous tasks where
one agent might err. Two aggregation strategies are provided:

- :func:`majority_vote` - pick the most common answer (ties broken by first occurrence).
- :func:`unanimous` - accept only when all workers agree.

:class:`ConsensusOrchestrator` runs the workers and applies an aggregator, recording every
vote for auditability.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from multi_agent_orchestrator.core.agents import Worker
from multi_agent_orchestrator.core.types import Subtask, WorkerResult

Aggregator = Callable[[Sequence[str]], str | None]


def majority_vote(answers: Sequence[str]) -> str | None:
    """Return the most common answer, or ``None`` if there are no answers."""
    if not answers:
        return None
    counts = Counter(answers)
    best = max(counts, key=lambda answer: (counts[answer], -list(answers).index(answer)))
    return best


def unanimous(answers: Sequence[str]) -> str | None:
    """Return the answer only if all workers agree, else ``None``."""
    if not answers:
        return None
    first = answers[0]
    return first if all(answer == first for answer in answers) else None


@dataclass(frozen=True, slots=True)
class ConsensusResult:
    """The outcome of a consensus run on one task."""

    subtask: Subtask
    answer: str | None
    votes: tuple[str, ...]
    agreed: bool

    @property
    def num_votes(self) -> int:
        return len(self.votes)


class ConsensusOrchestrator:
    """Runs several workers on the same task and aggregates their answers."""

    def __init__(
        self, workers: Sequence[Worker], *, aggregator: Aggregator = majority_vote
    ) -> None:
        if len(workers) < 2:
            raise ValueError("consensus requires at least two workers")
        self._workers = tuple(workers)
        self._aggregator = aggregator

    def run(self, subtask: Subtask) -> ConsensusResult:
        """Have every worker answer ``subtask`` and aggregate the answers."""
        results: list[WorkerResult] = [worker.execute(subtask, {}) for worker in self._workers]
        votes = tuple(result.output for result in results)
        answer = self._aggregator(votes)
        agreed = answer is not None and votes.count(answer) == len(votes)
        return ConsensusResult(subtask=subtask, answer=answer, votes=votes, agreed=agreed)

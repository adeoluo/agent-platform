"""The platform eval suite, built on ``evaluation_harness``.

The suite drives the *running* HTTP API, not internal functions, so it measures what a
client sees: it ingests a fixed corpus for a dedicated tenant, then asks a fixed question
set through ``POST /v1/ask``. ``evaluation_harness.Runner`` executes the task and grades
answers with ``IncludesGrader``; accuracy carries a bootstrap confidence interval. On top of
that the suite reports citation behaviour (how often retrieval answers cite a source, and
how often the cited source is the expected document) and client-observed latency
percentiles.
"""

from __future__ import annotations

import math
import threading
from dataclasses import asdict, dataclass
from typing import Any

from evaluation_harness import Example, Runner, Task
from evaluation_harness.graders import IncludesGrader

from agent_platform.client import Client
from agent_platform.corpus import ACME_DOCUMENTS, EVAL_CASES, EvalCase


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile of ``values`` (0 for an empty list)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(pct / 100.0 * len(ordered))))
    return ordered[rank - 1]


class PlatformModel:
    """An ``evaluation_harness`` model that answers by calling the platform API."""

    def __init__(self, client: Client) -> None:
        self._client = client
        self._lock = threading.Lock()
        self.responses: dict[str, dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return "agent-platform"

    def generate(self, example: Example) -> str:
        response = self._client.ask(example.input)
        with self._lock:
            self.responses[example.identifier] = response
        return str(response.get("answer", ""))


@dataclass(frozen=True)
class EvalReport:
    """Aggregated results of one eval run."""

    examples: int
    accuracy: float
    accuracy_low: float
    accuracy_high: float
    accuracy_by_kind: dict[str, float]
    citation_rate: float
    citation_precision: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    failures: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format(self) -> str:
        lines = [
            f"examples            {self.examples}",
            f"accuracy            {self.accuracy:.1%} "
            f"(95% CI {self.accuracy_low:.1%} to {self.accuracy_high:.1%})",
        ]
        for kind, value in sorted(self.accuracy_by_kind.items()):
            lines.append(f"  {kind:<17} {value:.1%}")
        lines += [
            f"citation rate       {self.citation_rate:.1%} of retrieval answers cite a source",
            f"citation precision  {self.citation_precision:.1%} cite the expected document",
            f"latency p50/p95/p99 {self.latency_p50_ms:.1f} / {self.latency_p95_ms:.1f} / "
            f"{self.latency_p99_ms:.1f} ms",
        ]
        for failure in self.failures:
            lines.append(f"  miss: {failure['question']!r} -> {failure['answer']!r}")
        return "\n".join(lines)


def build_task(cases: list[EvalCase]) -> Task:
    examples = [
        Example(
            input=case.question,
            target=case.target,
            id=f"case-{index}",
            metadata={"kind": case.kind, "doc_id": case.doc_id},
        )
        for index, case in enumerate(cases, start=1)
    ]
    return Task(
        name="platform-grounded-qa",
        examples=examples,
        grader=IncludesGrader(),
        description="Grounded question answering through the public API.",
    )


def run_eval(
    client: Client,
    *,
    cases: list[EvalCase] | None = None,
    documents: list[dict[str, str]] | None = None,
    ingest: bool = True,
    workers: int = 1,
) -> EvalReport:
    """Ingest the corpus through ``client`` and evaluate the fixed question set."""
    cases = cases if cases is not None else EVAL_CASES
    if ingest:
        client.ingest([dict(doc) for doc in (documents or ACME_DOCUMENTS)])
    task = build_task(cases)
    model = PlatformModel(client)
    result = Runner(seed=0, max_workers=workers).run(model, task)
    aggregate = result.accuracy()

    by_kind: dict[str, list[float]] = {}
    cited = 0
    precise = 0
    retrieval = 0
    failures: list[dict[str, str]] = []
    latencies = [r.latency_seconds * 1000.0 for r in result.example_results]
    for item in result.example_results:
        kind = str(item.example.metadata["kind"])
        by_kind.setdefault(kind, []).append(item.grade.score)
        if not item.grade.passed:
            failures.append({"question": item.example.input, "answer": item.prediction})
        if kind != "retrieval":
            continue
        retrieval += 1
        response = model.responses.get(item.example.identifier, {})
        citations = response.get("citations", [])
        if response.get("grounded"):
            cited += 1
        if item.example.metadata["doc_id"] in citations:
            precise += 1
    return EvalReport(
        examples=result.count,
        accuracy=aggregate.mean,
        accuracy_low=aggregate.low,
        accuracy_high=aggregate.high,
        accuracy_by_kind={k: sum(v) / len(v) for k, v in by_kind.items()},
        citation_rate=cited / retrieval if retrieval else 0.0,
        citation_precision=precise / retrieval if retrieval else 0.0,
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        latency_p99_ms=percentile(latencies, 99),
        failures=failures,
    )

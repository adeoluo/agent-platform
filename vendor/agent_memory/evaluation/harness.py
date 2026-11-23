"""Benchmark harness: memory conditions, retrieval ablations, and scaling measurements.

Every condition answers the same questions with the same deterministic reader, so differences
come only from what each memory condition puts in front of the reader:

* ``no memory``: the reader sees nothing and guesses uniformly among the attribute's values.
* ``full-context truncation``: the reader sees the most recent ``window`` words of the raw
  transcript, the way a fixed context window would.
* retrieval conditions: the reader sees the top ``k`` memories returned by ``AgentMemory.recall``
  with a given scoring configuration.

The reader is told which attribute is being asked about (an oracle question parser) so the
benchmark isolates memory quality from question understanding. It extracts statements about that
attribute from what it sees, orders them by time, and answers with the latest value (``current``)
or the value before the latest change (``previous``).
"""

from __future__ import annotations

import itertools
import random
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from agent_memory.evaluation.dataset import Dataset, Question, UserLog, generate_dataset
from agent_memory.extraction import RuleExtractor
from agent_memory.memory import AgentMemory
from agent_memory.models import Episode, Fact, RetrievedMemory
from agent_memory.settings import MemorySettings, ScoringWeights
from agent_memory.text import normalize_value

ANSWER_K = 5
RECALL_KS = (1, 3, 5, 10)
_EXTRACTOR = RuleExtractor()


@dataclass(frozen=True)
class Evidence:
    text: str
    time: float
    role: str = "user"
    predicate: str | None = None
    value: str | None = None


def statements(item: Evidence, predicate: str) -> list[str]:
    """Values for ``predicate`` that this piece of evidence states."""
    if item.predicate is not None and item.value is not None:
        return [item.value] if item.predicate == predicate else []
    extraction = _EXTRACTOR.extract(item.text, item.role)
    return [fact.object for fact in extraction.facts if fact.predicate == predicate]


def read_answer(evidence: Sequence[Evidence], predicate: str, mode: str) -> str | None:
    timeline: list[tuple[float, str]] = []
    for item in evidence:
        timeline.extend((item.time, value) for value in statements(item, predicate))
    timeline.sort(key=lambda entry: entry[0])
    collapsed: list[str] = []
    for _, value in timeline:
        if not collapsed or normalize_value(collapsed[-1]) != normalize_value(value):
            collapsed.append(value)
    if mode == "current":
        return collapsed[-1] if collapsed else None
    return collapsed[-2] if len(collapsed) >= 2 else None


def supports(item: Evidence, question: Question) -> bool:
    gold = normalize_value(question.answer)
    return any(normalize_value(value) == gold for value in statements(item, question.predicate))


def correct(answer: str | None, question: Question) -> bool:
    return answer is not None and normalize_value(answer) == normalize_value(question.answer)


def to_evidence(hit: RetrievedMemory) -> Evidence:
    memory = hit.memory
    if isinstance(memory, Fact):
        if memory.predicate == "summary":
            return Evidence(memory.object, memory.created_at)
        return Evidence(memory.text, memory.updated_at, "user", memory.predicate, memory.object)
    if isinstance(memory, Episode):
        return Evidence(memory.content, memory.created_at, memory.role)
    return Evidence(memory.text, memory.created_at, "preference")


# --------------------------------------------------------------------------- building stores


def ingest(
    memory: AgentMemory,
    log: UserLog,
    consolidate_every: int | None = None,
) -> float:
    """Load one user's transcript; returns total ingest seconds (consolidation excluded)."""
    view = memory.for_namespace(log.user_id)
    elapsed = 0.0
    last_session = -1
    for turn in log.turns:
        boundary = turn.session_index != last_session and last_session >= 0
        if consolidate_every and boundary and (last_session + 1) % consolidate_every == 0:
            view.consolidate(now=turn.timestamp)
        last_session = turn.session_index
        start = time.perf_counter()
        view.add_turn(turn.content, turn.role, session_id=turn.session_id, timestamp=turn.timestamp)
        elapsed += time.perf_counter() - start
    if consolidate_every:
        view.consolidate(now=log.end_time)
    return elapsed


def build_memory(
    dataset: Dataset,
    *,
    extract_on_write: bool = True,
    consolidate_every: int | None = None,
    settings: MemorySettings | None = None,
) -> AgentMemory:
    base = settings or MemorySettings()
    configured = MemorySettings(
        weights=base.weights,
        recency_half_life_days=base.recency_half_life_days,
        candidate_pool=base.candidate_pool,
        min_semantic_similarity=base.min_semantic_similarity,
        extract_on_write=extract_on_write,
        multi_valued_predicates=base.multi_valued_predicates,
        conflict_margin=base.conflict_margin,
        consolidate_after_days=base.consolidate_after_days,
        summary_sentences=base.summary_sentences,
        forgetting=base.forgetting,
    )
    memory = AgentMemory(namespace="benchmark", settings=configured)
    for log in dataset.users:
        ingest(memory, log, consolidate_every)
    return memory


# -------------------------------------------------------------------------------- conditions


@dataclass
class ConditionResult:
    name: str
    correct_current: int = 0
    total_current: int = 0
    correct_previous: int = 0
    total_previous: int = 0
    correct_paraphrased: int = 0
    total_paraphrased: int = 0
    hits: dict[int, int] | None = None
    retrieval: bool = True

    def record(self, question: Question, is_correct: bool, ranks: list[bool] | None) -> None:
        if question.mode == "current":
            self.total_current += 1
            self.correct_current += is_correct
        else:
            self.total_previous += 1
            self.correct_previous += is_correct
        if question.paraphrased:
            self.total_paraphrased += 1
            self.correct_paraphrased += is_correct
        if ranks is not None:
            if self.hits is None:
                self.hits = dict.fromkeys(RECALL_KS, 0)
            for k in RECALL_KS:
                self.hits[k] += any(ranks[:k])

    def summary(self) -> dict[str, float | None]:
        total = self.total_current + self.total_previous

        def ratio(a: int, b: int) -> float | None:
            return a / b if b else None

        result: dict[str, float | None] = {
            "accuracy": ratio(self.correct_current + self.correct_previous, total),
            "accuracy_current": ratio(self.correct_current, self.total_current),
            "accuracy_previous": ratio(self.correct_previous, self.total_previous),
            "accuracy_paraphrased": ratio(self.correct_paraphrased, self.total_paraphrased),
        }
        for k in RECALL_KS:
            result[f"recall@{k}"] = ratio(self.hits[k], total) if self.hits else None
        return result


def run_no_memory(dataset: Dataset, seed: int) -> ConditionResult:
    rng = random.Random(seed)
    result = ConditionResult("no memory", retrieval=False)
    for question in dataset.questions:
        result.record(question, correct(rng.choice(question.candidates), question), None)
    return result


def run_truncation(dataset: Dataset, window_words: int) -> ConditionResult:
    result = ConditionResult(f"full-context truncation ({window_words} words)", retrieval=False)
    for log in dataset.users:
        window: list[Evidence] = []
        budget = window_words
        for turn in reversed(log.turns):
            words = len(turn.content.split()) + 1
            if words > budget:
                break
            budget -= words
            window.append(Evidence(turn.content, turn.timestamp, turn.role))
        for question in log.questions:
            answer = read_answer(window, question.predicate, question.mode)
            result.record(question, correct(answer, question), None)
    return result


def run_retrieval(
    name: str,
    memory: AgentMemory,
    dataset: Dataset,
    weights: ScoringWeights,
    latencies: list[float] | None = None,
) -> ConditionResult:
    result = ConditionResult(name)
    for log in dataset.users:
        view = memory.for_namespace(log.user_id)
        for question in log.questions:
            start = time.perf_counter()
            hits = view.recall(
                question.text,
                max(RECALL_KS),
                include_history=question.mode == "previous",
                weights=weights,
                now=log.end_time,
                touch=False,
            )
            if latencies is not None:
                latencies.append(time.perf_counter() - start)
            evidence = [to_evidence(hit) for hit in hits]
            answer = read_answer(evidence[:ANSWER_K], question.predicate, question.mode)
            ranks = [supports(item, question) for item in evidence]
            ranks += [False] * (max(RECALL_KS) - len(ranks))
            result.record(question, correct(answer, question), ranks)
    return result


def _single(term: str) -> ScoringWeights:
    values = dict.fromkeys(("lexical", "semantic", "recency", "importance"), 0.0)
    values[term] = 1.0
    return ScoringWeights(**values)


def evaluate_seed(
    seed: int,
    users: int,
    sessions: int,
    windows: Sequence[int],
    tuned: ScoringWeights | None = None,
) -> dict[str, Any]:
    dataset = generate_dataset(users=users, sessions=sessions, seed=seed)
    full = build_memory(dataset)
    episodes_only = build_memory(dataset, extract_on_write=False)
    hybrid = ScoringWeights()
    tuned_rows = [] if tuned is None else [run_retrieval(TUNED_NAME, full, dataset, tuned)]
    rows: list[ConditionResult] = [
        run_no_memory(dataset, seed),
        *(run_truncation(dataset, window) for window in windows),
        run_retrieval("lexical only (BM25)", full, dataset, _single("lexical")),
        run_retrieval("full hybrid", full, dataset, hybrid),
        run_retrieval(
            "full hybrid, episodes only (no fact extraction)", episodes_only, dataset, hybrid
        ),
        *tuned_rows,
    ]
    ablations: list[ConditionResult] = [
        run_retrieval("full hybrid", full, dataset, hybrid),
        *(
            run_retrieval(f"without {term}", full, dataset, hybrid.without(term))
            for term in ("lexical", "semantic", "recency", "importance")
        ),
        run_retrieval("semantic only (embedding)", full, dataset, _single("semantic")),
    ]
    episode_ablations: list[ConditionResult] = [
        run_retrieval("full hybrid", episodes_only, dataset, hybrid),
        *(
            run_retrieval(f"without {term}", episodes_only, dataset, hybrid.without(term))
            for term in ("lexical", "semantic", "recency", "importance")
        ),
    ]
    return {
        "seed": seed,
        "questions": len(dataset.questions),
        "turns": len(dataset.turns),
        "conditions": {row.name: row.summary() for row in rows},
        "ablation": {row.name: row.summary() for row in ablations},
        "ablation_episodes_only": {row.name: row.summary() for row in episode_ablations},
    }


# ------------------------------------------------------------------------------------ tuning

TUNED_NAME = "hybrid, weights tuned on development seeds"


def tune_weights(
    dev_seeds: Sequence[int], users: int, sessions: int, grid: Sequence[float]
) -> tuple[ScoringWeights, float]:
    """Grid-search the four weights on development seeds (disjoint from the test seeds).

    The objective is mean answer accuracy, with mean recall@1 as the tie-breaker.
    """
    stores = []
    for seed in dev_seeds:
        dataset = generate_dataset(users=users, sessions=sessions, seed=seed)
        stores.append((dataset, build_memory(dataset)))
    best: tuple[float, float, ScoringWeights] | None = None
    for combination in itertools.product(grid, repeat=4):
        if not any(combination):
            continue
        weights = ScoringWeights(*combination)
        accuracies, recalls = [], []
        for dataset, memory in stores:
            summary = run_retrieval("tuning", memory, dataset, weights).summary()
            accuracies.append(summary["accuracy"] or 0.0)
            recalls.append(summary["recall@1"] or 0.0)
        candidate = (statistics.fmean(accuracies), statistics.fmean(recalls), weights)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    return best[2], best[0]


# ----------------------------------------------------------------------------------- scaling


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def measure_scaling(
    session_counts: Sequence[int], users: int = 3, seed: int = 11, consolidate_every: int = 10
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sessions in session_counts:
        dataset = generate_dataset(users=users, sessions=sessions, seed=seed)
        for consolidated in (False, True):
            memory = AgentMemory(namespace="benchmark")
            ingest_seconds = 0.0
            for log in dataset.users:
                ingest_seconds += ingest(memory, log, consolidate_every if consolidated else None)
            latencies: list[float] = []
            outcome = run_retrieval("full hybrid", memory, dataset, ScoringWeights(), latencies)
            stored = sum(
                sum(memory.store.counts(log.user_id).values())
                - memory.store.counts(log.user_id).get("episodic.consolidated", 0)
                for log in dataset.users
            )
            rows.append(
                {
                    "sessions": sessions,
                    "consolidation": consolidated,
                    "turns": len(dataset.turns),
                    "memories_stored": stored,
                    "database_bytes": memory.store.size_bytes(compact=True),
                    "ingest_ms_per_turn": 1000.0 * ingest_seconds / max(len(dataset.turns), 1),
                    "recall_p50_ms": 1000.0 * _percentile(latencies, 0.5),
                    "recall_p95_ms": 1000.0 * _percentile(latencies, 0.95),
                    "accuracy": outcome.summary()["accuracy"],
                }
            )
            memory.close()
    return rows


# ------------------------------------------------------------------------------- aggregation


def _aggregate(per_seed: list[dict[str, dict[str, float | None]]]) -> dict[str, dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for name in per_seed[0]:
        metrics: dict[str, Any] = {}
        for metric in per_seed[0][name]:
            values = [run[name][metric] for run in per_seed if run[name][metric] is not None]
            if not values:
                metrics[metric] = None
                continue
            numbers = [float(value) for value in values if value is not None]
            metrics[metric] = {
                "mean": statistics.fmean(numbers),
                "std": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
            }
        aggregated[name] = metrics
    return aggregated


def run_benchmark(
    users: int = 12,
    sessions: int = 20,
    seeds: Sequence[int] | None = None,
    seed: int = 7,
    windows: Sequence[int] = (400, 1600),
    scaling: Sequence[int] = (10, 25, 50, 100, 200),
    progress: Callable[[str], None] | None = None,
    dev_seeds: Sequence[int] = (),
    tuning_grid: Sequence[float] = (0.0, 0.15, 0.3, 0.5),
) -> dict[str, Any]:
    seed_list = list(seeds) if seeds is not None else [seed, seed + 1, seed + 2]
    if set(dev_seeds) & set(seed_list):
        raise ValueError("development seeds must be disjoint from test seeds")
    tuned: ScoringWeights | None = None
    tuning: dict[str, Any] | None = None
    if dev_seeds:
        if progress:
            progress(f"tuning weights on seeds {list(dev_seeds)}")
        tuned, dev_accuracy = tune_weights(dev_seeds, users, sessions, tuning_grid)
        tuning = {
            "dev_seeds": list(dev_seeds),
            "grid": list(tuning_grid),
            "weights": tuned.__dict__,
            "dev_accuracy": dev_accuracy,
        }
    runs = []
    for value in seed_list:
        if progress:
            progress(f"seed {value}")
        runs.append(evaluate_seed(value, users, sessions, windows, tuned))
    if progress:
        progress("scaling")
    return {
        "configuration": {
            "users": users,
            "sessions": sessions,
            "seeds": seed_list,
            "truncation_windows_words": list(windows),
            "answer_k": ANSWER_K,
            "weights": ScoringWeights().__dict__,
        },
        "tuning": tuning,
        "questions_per_seed": [run["questions"] for run in runs],
        "turns_per_seed": [run["turns"] for run in runs],
        "conditions": _aggregate([run["conditions"] for run in runs]),
        "ablation": _aggregate([run["ablation"] for run in runs]),
        "ablation_episodes_only": _aggregate([run["ablation_episodes_only"] for run in runs]),
        "scaling": measure_scaling(scaling) if scaling else [],
    }


# -------------------------------------------------------------------------------- formatting


def _cell(metric: Any) -> str:
    if metric is None:
        return "n/a"
    return f"{metric['mean']:.3f} ± {metric['std']:.3f}"


def _table(section: dict[str, dict[str, Any]]) -> list[str]:
    columns = (
        "accuracy",
        "accuracy_current",
        "accuracy_previous",
        "accuracy_paraphrased",
        "recall@1",
        "recall@5",
    )
    lines = [
        "| Condition | " + " | ".join(columns) + " |",
        "|---|" + "---|" * len(columns),
    ]
    for name, metrics in section.items():
        lines.append(f"| {name} | " + " | ".join(_cell(metrics[c]) for c in columns) + " |")
    return lines


def format_report(report: dict[str, Any]) -> str:
    configuration = report["configuration"]
    lines = [
        f"Seeds {configuration['seeds']}, {configuration['users']} users x "
        f"{configuration['sessions']} sessions, questions per seed {report['questions_per_seed']}",
        "",
        *(
            [
                f"Tuned weights (selected on seeds {report['tuning']['dev_seeds']}): "
                + ", ".join(f"{k}={v}" for k, v in report["tuning"]["weights"].items()),
                "",
            ]
            if report.get("tuning")
            else []
        ),
        "Memory conditions (mean ± standard deviation across seeds):",
        *_table(report["conditions"]),
        "",
        "Scoring-term ablation on the full store:",
        *_table(report["ablation"]),
        "",
        "Scoring-term ablation on an episodes-only store:",
        *_table(report["ablation_episodes_only"]),
    ]
    if report["scaling"]:
        lines += [
            "",
            "Scaling (3 users, consolidation every 10 sessions when enabled):",
            "| Sessions | Consolidation | Turns | Memories stored | Database KB | "
            + "Ingest ms/turn | Recall p50 ms | Recall p95 ms | Accuracy |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for row in report["scaling"]:
            lines.append(
                f"| {row['sessions']} | {'yes' if row['consolidation'] else 'no'} | "
                f"{row['turns']} | {row['memories_stored']} | "
                f"{row['database_bytes'] / 1024:.0f} | {row['ingest_ms_per_turn']:.2f} | "
                f"{row['recall_p50_ms']:.2f} | {row['recall_p95_ms']:.2f} | "
                f"{row['accuracy']:.3f} |"
            )
    return "\n".join(lines)

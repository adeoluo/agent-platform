"""Command-line interface: ``agent-memory add | recall | consolidate | stats | ... | mcp``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agent_memory import __version__
from agent_memory.memory import AgentMemory

DEFAULT_DATABASE = "~/.agent-memory/memory.db"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-memory", description="Long-term memory for AI agents."
    )
    parser.add_argument("--version", action="version", version=f"agent-memory {__version__}")
    parser.add_argument(
        "--database",
        default=os.environ.get("AGENT_MEMORY_DATABASE", DEFAULT_DATABASE),
        help="SQLite database path (environment: AGENT_MEMORY_DATABASE)",
    )
    parser.add_argument(
        "--namespace",
        default=os.environ.get("AGENT_MEMORY_NAMESPACE", "default"),
        help="memory namespace, one per user or agent (environment: AGENT_MEMORY_NAMESPACE)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    add = commands.add_parser("add", help="store a conversation turn and extract facts from it")
    add.add_argument("text")
    add.add_argument("--role", default="user")
    add.add_argument("--session", default="cli")
    add.add_argument("--json", action="store_true")

    recall = commands.add_parser("recall", help="retrieve memories with explanations")
    recall.add_argument("query")
    recall.add_argument("-k", type=int, default=5)
    recall.add_argument("--history", action="store_true", help="include superseded facts")
    recall.add_argument("--json", action="store_true")

    consolidate = commands.add_parser("consolidate", help="summarize old episodes and forget")
    consolidate.add_argument("--json", action="store_true")

    stats = commands.add_parser("stats", help="show memory statistics for the namespace")
    stats.add_argument("--json", action="store_true")

    facts = commands.add_parser("facts", help="list semantic facts")
    facts.add_argument("--subject")
    facts.add_argument("--predicate")
    facts.add_argument("--history", action="store_true")
    facts.add_argument("--json", action="store_true")

    forget = commands.add_parser("forget", help="delete one memory by identifier")
    forget.add_argument("memory_id", type=int)

    export = commands.add_parser("export", help="export the namespace to a JSON file")
    export.add_argument("path")

    load = commands.add_parser("import", help="import a JSON export into the namespace")
    load.add_argument("path")

    commands.add_parser("mcp", help="serve this namespace as a Model Context Protocol server")

    bench = commands.add_parser("benchmark", help="run the evaluation harness")
    bench.add_argument("--users", type=int, default=12)
    bench.add_argument("--sessions", type=int, default=20)
    bench.add_argument("--seed", type=int, default=7)
    bench.add_argument("--scaling", default="10,25,50,100,200")
    bench.add_argument("--output", help="write the JSON report to this path")
    return parser


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "benchmark":
        from agent_memory.evaluation.harness import format_report, run_benchmark

        report = run_benchmark(
            users=args.users,
            sessions=args.sessions,
            seed=args.seed,
            scaling=[int(part) for part in args.scaling.split(",") if part.strip()],
        )
        if args.output:
            Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(format_report(report))
        return 0

    if args.command == "mcp":
        from agent_memory.mcp_server import serve

        serve(args.database, args.namespace)
        return 0

    with AgentMemory(args.database, namespace=args.namespace) as memory:
        return _dispatch(memory, args)


def _dispatch(memory: AgentMemory, args: argparse.Namespace) -> int:
    if args.command == "add":
        result = memory.add_turn(args.text, args.role, session_id=args.session)
        if args.json:
            _print_json(
                {
                    "episode_id": result.episode.id,
                    "facts": [fact.text for fact in result.facts],
                    "preferences": [item.text for item in result.preferences],
                    "actions": list(result.actions),
                }
            )
        else:
            print(f"stored episode {result.episode.id}")
            for action in result.actions:
                print(f"  {action}")
        return 0

    if args.command == "recall":
        hits = memory.recall(args.query, args.k, include_history=args.history)
        if args.json:
            _print_json([hit.to_dict() for hit in hits])
        elif not hits:
            print("no matching memories")
        else:
            for rank, hit in enumerate(hits, 1):
                print(f"{rank}. [{hit.memory.kind.value} #{hit.id}] {hit.text}")
                print(f"   score {hit.score.total:.3f}: {hit.explanation}")
        return 0

    if args.command == "consolidate":
        report = memory.consolidate()
        if args.json:
            _print_json(report.__dict__)
        else:
            print(
                f"summarized {report.sessions_summarized} sessions "
                f"({report.episodes_consolidated} episodes), wrote {report.facts_written} facts, "
                f"forgot {report.episodes_forgotten} episodes and {report.facts_forgotten} facts"
            )
        return 0

    if args.command == "stats":
        statistics = memory.stats()
        if args.json:
            _print_json(statistics)
        else:
            for key, value in statistics.items():
                print(f"{key:>22}: {value}")
        return 0

    if args.command == "facts":
        facts = memory.facts(args.subject, args.predicate, include_history=args.history)
        if args.json:
            _print_json(
                [
                    {
                        "id": fact.id,
                        "subject": fact.subject,
                        "predicate": fact.predicate,
                        "object": fact.object,
                        "confidence": round(fact.confidence, 3),
                        "status": fact.status.value,
                    }
                    for fact in facts
                ]
            )
        else:
            for fact in facts:
                print(f"#{fact.id} [{fact.status.value}, {fact.confidence:.2f}] {fact.text}")
        return 0

    if args.command == "forget":
        if memory.forget(args.memory_id):
            print(f"forgot memory {args.memory_id}")
            return 0
        print(f"no memory {args.memory_id} in namespace {memory.namespace!r}", file=sys.stderr)
        return 1

    if args.command == "export":
        count = memory.export_to(args.path)
        print(f"exported {count} memories to {args.path}")
        return 0

    count = memory.import_from(args.path)
    print(f"imported {count} memories into namespace {memory.namespace!r}")
    return 0

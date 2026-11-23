"""Command-line interface for the multi-agent orchestrator.

Subcommands:

- ``run`` - orchestrate planner, workers, and critic over a goal and print the blackboard
  log and final outputs.
- ``compare`` - run the same goal through the single-agent baseline and the multi-agent
  orchestrator and contrast their structure, making the value of coordination concrete.

The CLI wires the agents and the orchestrator together; it holds no coordination logic.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from multi_agent_orchestrator import __version__
from multi_agent_orchestrator.agents import (
    EchoWorker,
    KeywordPlanner,
    LengthCritic,
    UppercaseWorker,
)
from multi_agent_orchestrator.baseline import SingleAgent, compare_structure
from multi_agent_orchestrator.orchestrator import Orchestrator

_DEFAULT_GOAL = "research the subject and then draft an outline and then write the report"


def _orchestrator() -> Orchestrator:
    return Orchestrator(
        KeywordPlanner(chain_dependencies=True),
        [EchoWorker(), UppercaseWorker()],
        LengthCritic(),
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="multi-agent-orchestrator",
        description="A planner/workers/critic multi-agent orchestrator built from scratch.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Orchestrate agents over a goal.")
    run_parser.add_argument("goal", nargs="?", default=_DEFAULT_GOAL, help="The goal.")
    run_parser.add_argument("--metrics", action="store_true", help="Print run metrics.")
    run_parser.add_argument(
        "--json", type=Path, default=None, metavar="PATH", help="Write the run as JSON."
    )
    run_parser.set_defaults(handler=_handle_run)

    compare_parser = subparsers.add_parser(
        "compare", help="Contrast single-agent and multi-agent on a goal."
    )
    compare_parser.add_argument("goal", nargs="?", default=_DEFAULT_GOAL, help="The goal.")
    compare_parser.set_defaults(handler=_handle_compare)

    measure_parser = subparsers.add_parser(
        "measure", help="Measure single-agent vs multi-agent over a suite (cost and coverage)."
    )
    measure_parser.set_defaults(handler=_handle_measure)

    durable_parser = subparsers.add_parser(
        "durable", help="Interrupt a run, then resume it from its checkpoint."
    )
    durable_parser.add_argument("goal", nargs="?", default=_DEFAULT_GOAL, help="The goal.")
    durable_parser.set_defaults(handler=_handle_durable)

    serve_parser = subparsers.add_parser("serve", help="Start the HTTP orchestration service.")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    serve_parser.set_defaults(handler=_handle_serve)
    return parser


def _handle_measure(args: argparse.Namespace) -> int:
    from multi_agent_orchestrator.comparison import DEFAULT_SUITE, compare_suite

    summary = compare_suite(list(DEFAULT_SUITE))
    print("Single-agent vs multi-agent (worker calls | raw coverage | verified coverage):")
    for row in summary.rows:
        print(
            f"  {row.parts}-part goal: "
            f"calls {row.single.worker_calls} vs {row.multi.worker_calls}, "
            f"raw {row.single.raw_coverage:.2f} vs {row.multi.raw_coverage:.2f}, "
            f"verified {row.single.verified_coverage:.2f} vs {row.multi.verified_coverage:.2f}"
        )
    print(f"\n{summary.verdict()}")
    return 0


def _handle_durable(args: argparse.Namespace) -> int:
    import tempfile
    from pathlib import Path

    from multi_agent_orchestrator.durable import (
        CrashError,
        inspect_checkpoint,
        resume_durable,
        run_durable,
    )

    orchestrator = _orchestrator()
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "run.checkpoint"
        try:
            run_durable(orchestrator, args.goal, checkpoint, crash_after=1)
        except CrashError as crash:
            print(f"Interrupted: {crash}")
        print(f"Checkpoint on disk: {inspect_checkpoint(checkpoint).format()}")
        resumed = resume_durable(orchestrator, checkpoint)
        uninterrupted = _orchestrator().run(args.goal)
        identical = resumed.final_outputs() == uninterrupted.final_outputs()
        print(f"Resumed: completed={resumed.completed} ({resumed.num_subtasks} subtasks)")
        print(f"Resumed outputs identical to uninterrupted run: {identical}")
    return 0 if resumed.completed else 1


def _handle_serve(args: argparse.Namespace) -> int:
    from multi_agent_orchestrator.service import serve

    serve(host=args.host, port=args.port)
    return 0


def _handle_run(args: argparse.Namespace) -> int:
    result = _orchestrator().run(args.goal)
    print(f"Goal: {args.goal}")
    print(f"Completed: {result.completed} ({result.num_subtasks} subtasks)\n")
    for message in result.blackboard.messages:
        print(f"  [{message.role}] {message.content}")
    if args.metrics:
        from multi_agent_orchestrator.metrics import run_metrics

        print(f"\n{run_metrics(result).format()}")
    if args.json is not None:
        from multi_agent_orchestrator.trace import run_to_json

        args.json.write_text(run_to_json(result), encoding="utf-8")
        print(f"\nWrote run to {args.json}")
    return 0 if result.completed else 1


def _handle_compare(args: argparse.Namespace) -> int:
    single = SingleAgent(EchoWorker()).run(args.goal)
    multi = _orchestrator().run(args.goal)
    print(f"Goal: {args.goal}\n")
    print("Single-agent output:")
    print(f"  {single.output}\n")
    print("Multi-agent outputs:")
    for subtask_id, output in multi.final_outputs().items():
        print(f"  [{subtask_id}] {output}")
    print()
    print(compare_structure(single.subtasks, multi.num_subtasks))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = args.handler
    exit_code: int = handler(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

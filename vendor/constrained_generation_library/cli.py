"""Command line interface for the constrained generation library."""

from __future__ import annotations

import argparse

from constrained_generation_library.constrainer import constrained_generate
from constrained_generation_library.data import default_vocab, random_model
from constrained_generation_library.evaluation import validity_report
from constrained_generation_library.schema import Property, compile_schema, schema_to_pattern

_SCHEMA = [
    Property("name", "string", max_length=10),
    Property("age", "integer", max_length=3),
    Property("active", "boolean"),
    Property("role", "enum", ("admin", "user", "guest")),
]


def _cmd_pattern(_: argparse.Namespace) -> None:
    print(schema_to_pattern(_SCHEMA))


def _cmd_generate(args: argparse.Namespace) -> None:
    regex = compile_schema(_SCHEMA)
    vocab = default_vocab()
    result = constrained_generate(regex, vocab, random_model(args.seed), max_tokens=120)
    print(f"generated: {result.text}")
    print(f"valid:     {regex.fullmatch(result.text)}")
    print(f"mean allowed fraction of the vocabulary per step: {result.mean_allowed_fraction:.3f}")


def _cmd_report(args: argparse.Namespace) -> None:
    regex = compile_schema(_SCHEMA)
    report = validity_report(regex, trials=args.trials)
    print(f"Over {args.trials} random models:")
    print(f"  {report.format()}")
    print(f"  mean allowed fraction of the vocabulary per step: {report.mean_allowed_fraction:.3f}")


def _cmd_serve(args: argparse.Namespace) -> None:
    from constrained_generation_library.service import serve

    serve(host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="constrained-generation-library", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("pattern", help="print the regular expression compiled from the schema")

    generate = sub.add_parser("generate", help="generate a schema-valid document from a seed")
    generate.add_argument("--seed", type=int, default=0)

    report = sub.add_parser("report", help="validity rate with and without the constraint")
    report.add_argument("--trials", type=int, default=100)

    serve = sub.add_parser("serve", help="start the HTTP service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    handlers = {
        "pattern": _cmd_pattern,
        "generate": _cmd_generate,
        "report": _cmd_report,
        "serve": _cmd_serve,
    }
    args = parser.parse_args(argv)
    handlers[args.command](args)


if __name__ == "__main__":
    main()

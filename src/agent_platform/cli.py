"""Command-line interface: ``agent-platform <command>``."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import FrameType
from typing import Any

from agent_platform import __version__
from agent_platform.config import Settings

DEFAULT_URL = "http://127.0.0.1:8080"


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    if getattr(args, "db", None):
        settings = replace(settings, db_path=Path(args.db))
    return settings


def _client(args: argparse.Namespace) -> Any:
    from agent_platform.client import Client

    key = args.api_key or os.environ.get("AGENT_PLATFORM_API_KEY")
    if not key:
        raise SystemExit("an API key is required (--api-key or AGENT_PLATFORM_API_KEY)")
    return Client(args.url, key)


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def _install_stop_handlers(stop: threading.Event) -> None:
    def handler(signum: int, frame: FrameType | None) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


# -- commands ------------------------------------------------------------------------ #
def cmd_serve(args: argparse.Namespace) -> int:
    from agent_platform.api import make_server
    from agent_platform.app import Platform

    settings = _settings(args)
    settings = replace(settings, host=args.host or settings.host, port=args.port or settings.port)
    platform = Platform(settings)
    server = make_server(platform, settings.host, settings.port)
    stop = threading.Event()
    _install_stop_handlers(stop)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    print(
        f"agent-platform listening on http://{host!s}:{port} (db: {settings.db_path})", flush=True
    )
    stop.wait()
    server.shutdown()
    server.server_close()
    platform.close()
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    from agent_platform.app import Platform

    settings = _settings(args)
    if args.lease_seconds is not None:
        settings = replace(settings, lease_seconds=args.lease_seconds)
    if args.poll is not None:
        settings = replace(settings, poll_interval=args.poll)
    platform = Platform(settings)
    worker = platform.worker(worker_id=args.worker_id)
    _install_stop_handlers(worker.stop_event)
    print(f"worker {worker.worker_id} polling {settings.db_path}", file=sys.stderr, flush=True)
    processed = worker.run(max_jobs=args.max_jobs, exit_when_idle=args.exit_when_idle)
    print(f"worker {worker.worker_id} processed {processed} jobs", file=sys.stderr, flush=True)
    platform.close()
    return 0


def cmd_keys(args: argparse.Namespace) -> int:
    from agent_platform.auth import KeyStore
    from agent_platform.db import Database

    settings = _settings(args)
    db = Database(settings.db_path)
    store = KeyStore(db, pepper=settings.key_pepper)
    try:
        if args.keys_command == "create":
            try:
                issued = store.create(args.tenant, label=args.label)
            except ValueError as error:
                print(f"error: {error}", file=sys.stderr)
                return 2
            print(f"tenant:  {issued.tenant}")
            print(f"key id:  {issued.key_id}")
            # Shown once, on purpose: only an HMAC of the secret is stored.
            print(f"api key: {issued.secret}")
            print("Store the key now. Only its hash is kept, so it cannot be shown again.")
        elif args.keys_command == "list":
            for info in store.list(args.tenant):
                state = "revoked" if info.revoked else "active"
                print(f"{info.key_id}  {info.tenant:<20} {state:<8} {info.label}")
        else:
            if not store.revoke(args.key_id):
                print(f"no active key {args.key_id!r}", file=sys.stderr)
                return 1
            print(f"revoked {args.key_id}")
    finally:
        db.close()
    return 0


def _documents_from_paths(paths: list[str]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for file in files:
            if file.suffix == ".json":
                loaded = json.loads(file.read_text(encoding="utf-8"))
                documents.extend(loaded if isinstance(loaded, list) else loaded["documents"])
            elif file.suffix in (".txt", ".md"):
                documents.append({"id": file.stem, "text": file.read_text(encoding="utf-8")})
    return documents


def cmd_ingest(args: argparse.Namespace) -> int:
    documents = _documents_from_paths(args.paths)
    if not documents:
        print("no .txt, .md, or .json documents found", file=sys.stderr)
        return 2
    _print_json(_client(args).ingest(documents))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    schema = json.loads(args.schema) if args.schema else None
    _print_json(_client(args).ask(args.question, response_schema=schema))
    return 0


def _render_event(event: str, data: dict[str, Any], out: Any) -> None:
    if event == "step":
        print(f"  [{data['index']}] {data['tool']}: {data['input']}", file=out)
    elif event == "observation":
        preview = data["observation"]
        print(f"      {preview[:160]}{'...' if len(preview) > 160 else ''}", file=out)
    elif event == "answer":
        flag = "grounded" if data["grounded"] else "not grounded"
        print(f"\n{data['answer']}\n  ({flag}; trace {data['trace_id']})\n", file=out)
    elif event == "error":
        print(f"error {data['status']} {data['error']}: {data['message']}", file=out)


def cmd_chat(args: argparse.Namespace) -> int:
    """Interactive loop over the streaming endpoint; steps print as the agent takes them."""
    from agent_platform.client import APIError

    client = _client(args)
    interactive = sys.stdin.isatty()
    if interactive:
        print("Ask a question (Ctrl-D to exit).")
    status = 0
    while True:
        if interactive:
            print("> ", end="", flush=True)
        line = sys.stdin.readline()
        if not line:
            break
        question = line.strip()
        if not question:
            continue
        try:
            for event, data in client.ask_stream(question):
                _render_event(event, data, sys.stdout)
                if event == "error":
                    status = 1
        except APIError as error:
            print(f"error {error.status}: {error.payload}", file=sys.stderr)
            status = 1
    return status


def cmd_mcp(args: argparse.Namespace) -> int:
    from agent_platform.mcp_server import serve

    serve(_client(args))
    return 0


def cmd_jobs(args: argparse.Namespace) -> int:
    client = _client(args)
    if args.jobs_command == "submit":
        _print_json(client.submit_job(args.question, step_delay_seconds=args.step_delay))
    else:
        _print_json(client.job(args.job_id))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from agent_platform.client import Client
    from agent_platform.evaluation import run_eval

    if args.url:
        report = run_eval(_client(args), workers=args.workers)
    else:
        from agent_platform.api import ServerThread
        from agent_platform.app import Platform

        workdir = Path(args.workdir)
        if workdir.exists():
            shutil.rmtree(workdir)
        settings = replace(Settings.from_env(), db_path=workdir / "eval.db")
        platform = Platform(settings)
        key = platform.keys.create("eval", label="eval suite").secret
        with ServerThread(platform) as server:
            report = run_eval(Client(server.url, key), workers=args.workers)
        platform.close()
    if args.json:
        _print_json(report.to_dict())
    else:
        print(report.format())
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from agent_platform.demo import run_demo

    summary = run_demo(Path(args.workdir))
    return 0 if summary.get("job_status") == "succeeded" else 1


# -- parser -------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-platform",
        description="Authenticated multi-tenant agent platform with durable jobs.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_db(p: argparse.ArgumentParser) -> None:
        p.add_argument("--db", help="SQLite database path (default: AGENT_PLATFORM_DB)")

    def add_remote(p: argparse.ArgumentParser) -> None:
        p.add_argument("--url", default=os.environ.get("AGENT_PLATFORM_URL", DEFAULT_URL))
        p.add_argument("--api-key", help="API key (default: AGENT_PLATFORM_API_KEY)")

    serve = sub.add_parser("serve", help="run the HTTP API server")
    add_db(serve)
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.set_defaults(func=cmd_serve)

    worker = sub.add_parser("worker", help="run a durable job worker")
    add_db(worker)
    worker.add_argument("--worker-id")
    worker.add_argument("--lease-seconds", type=float)
    worker.add_argument("--poll", type=float, help="idle poll interval in seconds")
    worker.add_argument("--max-jobs", type=int)
    worker.add_argument("--exit-when-idle", action="store_true")
    worker.set_defaults(func=cmd_worker)

    keys = sub.add_parser("keys", help="manage API keys")
    keys_sub = keys.add_subparsers(dest="keys_command", required=True)
    create = keys_sub.add_parser("create", help="issue a key for a tenant")
    add_db(create)
    create.add_argument("--tenant", required=True)
    create.add_argument("--label", default="")
    listing = keys_sub.add_parser("list", help="list keys (never secrets)")
    add_db(listing)
    listing.add_argument("--tenant")
    revoke = keys_sub.add_parser("revoke", help="revoke a key by id")
    add_db(revoke)
    revoke.add_argument("key_id")
    keys.set_defaults(func=cmd_keys)

    ingest = sub.add_parser("ingest", help="ingest .txt/.md/.json documents over the API")
    add_remote(ingest)
    ingest.add_argument("paths", nargs="+")
    ingest.set_defaults(func=cmd_ingest)

    ask = sub.add_parser("ask", help="ask a question over the API")
    add_remote(ask)
    ask.add_argument("question")
    ask.add_argument("--schema", help="response schema as JSON")
    ask.set_defaults(func=cmd_ask)

    chat = sub.add_parser("chat", help="interactive questions, streaming the agent's steps")
    add_remote(chat)
    chat.set_defaults(func=cmd_chat)

    mcp = sub.add_parser("mcp", help="serve the platform to MCP clients over stdio")
    add_remote(mcp)
    mcp.set_defaults(func=cmd_mcp)

    jobs = sub.add_parser("jobs", help="submit or inspect research jobs")
    jobs_sub = jobs.add_subparsers(dest="jobs_command", required=True)
    submit = jobs_sub.add_parser("submit")
    add_remote(submit)
    submit.add_argument("-q", "--question", action="append", required=True)
    submit.add_argument("--step-delay", type=float, default=0.0)
    get = jobs_sub.add_parser("get")
    add_remote(get)
    get.add_argument("job_id")
    jobs.set_defaults(func=cmd_jobs)

    evaluate = sub.add_parser("eval", help="run the eval suite against the API")
    evaluate.add_argument("--url", help="evaluate a running server instead of an in-process one")
    evaluate.add_argument("--api-key")
    evaluate.add_argument("--workdir", default=".platform/eval")
    evaluate.add_argument("--workers", type=int, default=1)
    evaluate.add_argument("--json", action="store_true")
    evaluate.set_defaults(func=cmd_eval)

    demo = sub.add_parser("demo", help="run the end-to-end local scenario")
    demo.add_argument("--workdir", default=".platform/demo")
    demo.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result

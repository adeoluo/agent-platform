# AGENTS.md

Working notes for coding agents and contributors in **Agent Platform**. The [README](README.md)
explains what the project does and why; this file explains how to change it safely.

## Project

An authenticated, multi-tenant AI service that answers questions from each tenant's own documents and runs multi-step research jobs that survive worker crashes.

## Layout

- `benchmarks/`: benchmark scripts and recorded results
- `docs/`: documentation, architecture diagram source and render, design records
- `scripts/`: maintenance scripts (release, assets, data)
- `src/`: library and service source
- `tests/`: test suite

Key modules:

- `src/agent_platform/__init__.py`: Agent platform: an authenticated, multi-tenant AI service composed from small libraries.
- `src/agent_platform/agent.py`: The agent runtime: one place that answers a tenant's question.
- `src/agent_platform/api.py`: The HTTP API: a threaded standard-library server in front of the platform.
- `src/agent_platform/app.py`: The composition root: one object that wires every component to one database.
- `src/agent_platform/auth.py`: API keys: issuance, hashed storage, verification, and revocation.
- `src/agent_platform/cli.py`: Command-line interface: ``agent-platform <command>``.
- `src/agent_platform/client.py`: A minimal standard-library HTTP client for the platform API.
- `src/agent_platform/config.py`: Runtime settings, read from explicit arguments or ``AGENT_PLATFORM_*`` variables.
- `src/agent_platform/corpus.py`: A small fixed corpus and question set used by the demo, the eval suite, and tests.
- `src/agent_platform/db.py`: SQLite storage: schema, per-thread connections, and explicit write transactions.
- `src/agent_platform/demo.py`: ``agent-platform demo``: an end-to-end scenario on a fresh local database.
- `src/agent_platform/evaluation.py`: The platform eval suite, built on ``evaluation_harness``.
- `src/agent_platform/jobs.py`: Durable job storage: submission, lease-based claiming, checkpoints, and a step ledger.
- `src/agent_platform/knowledge.py`: Per-tenant knowledge bases: durable documents and cached hybrid retrievers.
- `src/agent_platform/metrics.py`: A small thread-safe metrics registry rendered in the Prometheus text format.
- `src/agent_platform/model.py`: The language-model path: a deterministic offline provider behind the safety gateway.
- `src/agent_platform/orchestration.py`: Multi-agent jobs built on ``multi_agent_orchestrator``.
- `src/agent_platform/processes.py`: Helpers for running workers as separate operating-system processes.
- `src/agent_platform/structured.py`: Structured answers via schema-constrained decoding.
- `src/agent_platform/tracing.py`: Trace persistence on top of ``llm_observability``.
- `src/agent_platform/vectors.py`: A vector retriever: hashed embeddings indexed in an HNSW graph.
- `src/agent_platform/worker.py`: The job worker: claim, heartbeat, execute step by step, checkpoint, finish.

## Commands

| Command | Purpose |
|---|---|
| `make help` | Show this help |
| `make setup` | Install dependencies (including the git-sourced libraries) |
| `make format` | Format automatically |
| `make lint` | Lint (no fixes) |
| `make type` | Type-check |
| `make test` | Run the CI test selection |
| `make test-all` | Run every test, including slow soak tests |
| `make coverage` | Tests with the coverage gate used in CI |
| `make eval` | Run the eval suite against an in-process server |
| `make demo` | Run the end-to-end demo (server, worker crash, recovery) |
| `make serve` | Run the API server |
| `make worker` | Run a job worker |
| `make bench-load` | Load benchmark against a server process |
| `make bench-recovery` | Worker-kill recovery benchmark |
| `make docker-build` | Build the image |
| `make clean` | Remove caches and local data |
| `make all` | Full local gate (mirror of continuous integration) |
| `make changelog` | Regenerate CHANGELOG.md from Conventional Commits (git-cliff) |
| `make cut-release` | Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional) |

Run the full local gate (`make all`) before committing; CI runs the same checks.

## Conventions

- Commits follow Conventional Commits (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `ci:`, `chore:`).
  The changelog and the next version number are derived from them.
- Every number quoted in the README is produced by a command in this repository. When behavior
  changes, rerun that command and update the number in the same commit.
- When a component is added, removed or renamed, update `docs/architecture/architecture.mmd`
  and re-render `architecture.svg` so the diagram matches the code.
- README sections keep their fixed order. Prose avoids em and en dashes and promotional wording.
- Tests accompany behavior changes; a bug fix comes with a test that fails without it.
- Releases are cut by the maintainer with `make cut-release`, never from CI.

## Python notes

- Python is managed with `uv`; run commands through `uv run` so the locked environment is used.
- Lint and format with `ruff`; type-check with `mypy`; tests use `pytest`.
- Public functions carry type hints and a docstring that states the contract, not the steps.

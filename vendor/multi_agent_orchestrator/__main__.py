"""Enable ``python -m multi_agent_orchestrator`` to invoke the CLI."""

from __future__ import annotations

from multi_agent_orchestrator.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

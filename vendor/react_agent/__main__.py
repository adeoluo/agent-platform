"""Enable ``python -m react_agent`` to invoke the command-line interface."""

from __future__ import annotations

from react_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

"""Allow ``python -m agent_platform``."""

from __future__ import annotations

from agent_platform.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

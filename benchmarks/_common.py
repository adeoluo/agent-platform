"""Shared helpers for the benchmarks."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def fresh_dir(name: str) -> Path:
    path = ROOT / ".platform" / "bench" / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def start_server(
    db_path: Path, env: dict[str, str] | None = None
) -> tuple[subprocess.Popen[str], str]:
    """Start ``agent-platform serve`` on a free port in a child process; return it and its URL."""
    process = subprocess.Popen(
        [sys.executable, "-m", "agent_platform", "serve", "--db", str(db_path), "--port", "0"],
        env={**os.environ, **(env or {})},
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert process.stdout is not None
    line = process.stdout.readline()
    url = line.split("listening on ", 1)[1].split()[0]
    return process, url

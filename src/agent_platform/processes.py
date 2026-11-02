"""Helpers for running workers as separate operating-system processes."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def spawn_worker(
    db_path: Path,
    *,
    worker_id: str,
    lease_seconds: float,
    poll_interval: float = 0.05,
    log_path: Path | None = None,
) -> subprocess.Popen[bytes]:
    """Start ``agent-platform worker`` in a child process against ``db_path``.

    The child's stderr goes to ``log_path`` when given, otherwise it is discarded.
    """
    command = [
        sys.executable,
        "-m",
        "agent_platform",
        "worker",
        "--db",
        str(db_path),
        "--worker-id",
        worker_id,
        "--lease-seconds",
        str(lease_seconds),
        "--poll",
        str(poll_interval),
    ]
    if log_path is None:
        return subprocess.Popen(
            command,
            env=dict(os.environ),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    with log_path.open("ab") as log:
        return subprocess.Popen(
            command,
            env=dict(os.environ),
            stdout=subprocess.DEVNULL,
            stderr=log,
            start_new_session=True,
        )


def stop_process(process: subprocess.Popen[bytes], *, timeout: float = 5.0) -> None:
    """Terminate ``process`` gracefully, then forcibly if it does not exit."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout)

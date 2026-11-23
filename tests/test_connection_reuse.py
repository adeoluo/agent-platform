"""Database connections are recycled, so a long-running server does not leak file descriptors."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from agent_platform.api import ServerThread
from agent_platform.app import Platform
from agent_platform.client import Client
from agent_platform.db import Database
from tests.conftest import make_settings

FD_DIR = Path("/dev/fd")


def open_fds() -> int:
    return len(os.listdir(FD_DIR))


def test_a_server_holds_a_bounded_number_of_connections(workdir: Path) -> None:
    """Every request runs on a new thread; 300 of them must not cost 300 connections."""
    platform = Platform(make_settings(workdir, rate_capacity=1e9, rate_refill_per_second=1e9))
    try:
        with ServerThread(platform) as server:
            client = Client(server.url, platform.keys.create("acme").secret)
            client.ask("warm up")
            before = open_fds() if FD_DIR.exists() else 0
            for index in range(300):
                client.ask(f"What is {index} + 1?")
            assert platform.db.connection_count() <= Database.MAX_IDLE + 2
            if FD_DIR.exists():
                assert open_fds() - before <= 8, "file descriptors grew with the request count"
    finally:
        platform.close()


def test_connections_of_finished_threads_are_reused(workdir: Path) -> None:
    """A thread that ends without calling release() still gives its connection back."""
    db = Database(workdir / "reuse.db")
    seen: set[int] = set()

    def use() -> None:
        seen.add(id(db.conn()))
        assert db.ping()

    for _ in range(40):
        thread = threading.Thread(target=use)
        thread.start()
        thread.join()
    assert db.connection_count() <= Database.MAX_IDLE + 1
    assert len(seen) < 40
    db.close()
    assert db.connection_count() == 0


def test_a_thread_that_dies_mid_transaction_does_not_poison_its_connection(workdir: Path) -> None:
    db = Database(workdir / "rollback.db")
    db.conn().execute("CREATE TABLE t (x INTEGER)")
    db.release()

    def die_in_transaction() -> None:
        db.conn().execute("BEGIN IMMEDIATE")
        db.conn().execute("INSERT INTO t VALUES (1)")

    thread = threading.Thread(target=die_in_transaction)
    thread.start()
    thread.join()
    with db.transaction() as conn:
        conn.execute("INSERT INTO t VALUES (2)")
    assert [row[0] for row in db.conn().execute("SELECT x FROM t")] == [2]
    db.close()


def test_more_idle_connections_than_the_cap_are_closed(workdir: Path) -> None:
    db = Database(workdir / "cap.db")
    barrier = threading.Barrier(Database.MAX_IDLE + 6)

    def hold() -> None:
        db.conn()
        barrier.wait(timeout=10)
        db.release()

    threads = [threading.Thread(target=hold) for _ in range(Database.MAX_IDLE + 6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert db.connection_count() <= Database.MAX_IDLE + 1
    db.close()


@pytest.mark.parametrize("calls", [1, 3])
def test_release_is_safe_to_repeat(workdir: Path, calls: int) -> None:
    db = Database(workdir / "repeat.db")
    for _ in range(calls):
        db.release()
    assert db.ping()
    db.close()

"""Shared fixtures.

Test data lives under ``.pytest-work/`` inside the repository (ignored by git) rather than
a system temporary directory, and each test gets a fresh subdirectory.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from agent_platform.api import ServerThread
from agent_platform.app import Platform
from agent_platform.client import Client
from agent_platform.config import Settings

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / ".pytest-work"


@pytest.fixture
def workdir(request: pytest.FixtureRequest) -> Iterator[Path]:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.nodeid)[-120:]
    path = WORK / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def make_settings(workdir: Path, **overrides: object) -> Settings:
    base = Settings(db_path=workdir / "platform.db", lease_seconds=1.0, poll_interval=0.02)
    return Settings(**{**base.__dict__, **overrides})  # type: ignore[arg-type]


@pytest.fixture
def platform(workdir: Path) -> Iterator[Platform]:
    instance = Platform(make_settings(workdir))
    yield instance
    instance.close()


@dataclass
class Running:
    platform: Platform
    url: str

    def client(self, tenant: str) -> Client:
        return Client(self.url, self.platform.keys.create(tenant).secret)


@pytest.fixture
def running(platform: Platform) -> Iterator[Running]:
    with ServerThread(platform) as server:
        yield Running(platform, server.url)

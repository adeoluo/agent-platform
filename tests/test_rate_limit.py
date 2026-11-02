from __future__ import annotations

from pathlib import Path

from agent_platform.api import ServerThread
from agent_platform.app import Platform
from agent_platform.client import Client
from tests.conftest import make_settings


def test_rate_limit_is_per_tenant(workdir: Path) -> None:
    platform = Platform(make_settings(workdir, rate_capacity=3.0, rate_refill_per_second=0.0))
    with ServerThread(platform) as server:
        acme = Client(server.url, platform.keys.create("acme").secret)
        globex = Client(server.url, platform.keys.create("globex").secret)
        statuses = [
            acme.request("POST", "/v1/ask", {"question": "What is 1 + 1?"})[0] for _ in range(5)
        ]
        assert statuses == [200, 200, 200, 429, 429]
        assert globex.request("POST", "/v1/ask", {"question": "What is 2 + 2?"})[0] == 200
        status, body = acme.request("GET", "/v1/jobs/job-x")
        assert (status, body["error"]) == (429, "rate_limited")
        assert 'agent_platform_rate_limited_total{tenant="acme"} 3' in acme.metrics()
    platform.close()


def test_health_is_not_rate_limited(workdir: Path) -> None:
    platform = Platform(make_settings(workdir, rate_capacity=1.0, rate_refill_per_second=0.0))
    with ServerThread(platform) as server:
        client = Client(server.url)
        assert all(client.request("GET", "/health")[0] == 200 for _ in range(5))
    platform.close()

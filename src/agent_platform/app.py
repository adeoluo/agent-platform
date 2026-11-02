"""The composition root: one object that wires every component to one database."""

from __future__ import annotations

import threading

from llm_gateway import Provider, TenantRateLimiter

from agent_platform.agent import AgentRuntime
from agent_platform.auth import KeyStore
from agent_platform.config import Settings
from agent_platform.db import Database
from agent_platform.jobs import JobStore
from agent_platform.knowledge import KnowledgeBase
from agent_platform.metrics import Metrics
from agent_platform.model import TenantGateways
from agent_platform.providers import build_provider
from agent_platform.tracing import TraceStore
from agent_platform.worker import JobWorker


class Platform:
    """Holds the shared services used by the API server, workers, and the CLI."""

    def __init__(
        self, settings: Settings | None = None, *, provider: Provider | None = None
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.db = Database(self.settings.db_path)
        self.keys = KeyStore(self.db, pepper=self.settings.key_pepper)
        self.knowledge = KnowledgeBase(self.db)
        self.gateways = TenantGateways(provider or build_provider())
        self.runtime = AgentRuntime(self.knowledge, self.gateways)
        self.jobs = JobStore(self.db, max_attempts=self.settings.max_attempts)
        self.traces = TraceStore(self.db)
        self.metrics = Metrics()
        self._rate_limiter = TenantRateLimiter(
            self.settings.rate_capacity, self.settings.rate_refill_per_second
        )
        self._rate_lock = threading.Lock()

    def allow_request(self, tenant: str) -> bool:
        """Consume one unit of ``tenant``'s request quota, if available."""
        with self._rate_lock:
            return self._rate_limiter.allow(tenant)

    def worker(self, *, worker_id: str | None = None) -> JobWorker:
        """Create a job worker bound to this platform's stores."""
        return JobWorker(
            self.jobs,
            self.runtime,
            self.traces,
            worker_id=worker_id,
            lease_seconds=self.settings.lease_seconds,
            poll_interval=self.settings.poll_interval,
        )

    def close(self) -> None:
        self.db.close()

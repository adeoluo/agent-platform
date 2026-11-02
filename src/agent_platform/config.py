"""Runtime settings, read from explicit arguments or ``AGENT_PLATFORM_*`` variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

DEFAULT_DB_PATH = Path(".platform") / "platform.db"
# Development-only default. Production deployments must set AGENT_PLATFORM_KEY_PEPPER.
DEFAULT_PEPPER = "agent-platform-development-pepper"


@dataclass(frozen=True)
class Settings:
    """Configuration shared by the API server, workers, and the CLI."""

    db_path: Path = field(default_factory=lambda: DEFAULT_DB_PATH)
    host: str = "127.0.0.1"
    port: int = 8080
    key_pepper: str = DEFAULT_PEPPER
    rate_capacity: float = 120.0
    rate_refill_per_second: float = 60.0
    lease_seconds: float = 10.0
    poll_interval: float = 0.2
    max_attempts: int = 5
    max_body_bytes: int = 2_000_000
    max_step_delay_seconds: float = 10.0

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> Settings:
        """Build settings from environment variables, falling back to defaults."""
        env = dict(os.environ) if environ is None else environ
        base = cls()
        return replace(
            base,
            db_path=Path(env.get("AGENT_PLATFORM_DB", str(base.db_path))),
            host=env.get("AGENT_PLATFORM_HOST", base.host),
            port=int(env.get("AGENT_PLATFORM_PORT", base.port)),
            key_pepper=env.get("AGENT_PLATFORM_KEY_PEPPER", base.key_pepper),
            rate_capacity=float(env.get("AGENT_PLATFORM_RATE_CAPACITY", base.rate_capacity)),
            rate_refill_per_second=float(
                env.get("AGENT_PLATFORM_RATE_REFILL", base.rate_refill_per_second)
            ),
            lease_seconds=float(env.get("AGENT_PLATFORM_LEASE_SECONDS", base.lease_seconds)),
            poll_interval=float(env.get("AGENT_PLATFORM_POLL_INTERVAL", base.poll_interval)),
            max_attempts=int(env.get("AGENT_PLATFORM_MAX_ATTEMPTS", base.max_attempts)),
        )

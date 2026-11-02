"""Hosted model providers (OpenAI, Anthropic) and the factory that selects one.

Both providers implement the ``llm_gateway`` provider protocol (``complete(system, prompt)``)
with the standard-library HTTP client, so every call still passes through the tenant's
gateway for injection screening, PII redaction, output policy, and caching.

Errors are classified at the boundary, because retrying the wrong error is either wasteful
or dangerous:

* rate limits (429), server errors (5xx), timeouts, and connection failures raise
  ``llm_gateway.ProviderError``, which ``ResilientProvider`` retries and then fails over;
* any other 4xx (bad key, unknown model, malformed request) raises
  :class:`ProviderRequestError`, which is not retried: repeating it cannot succeed.

``build_provider("anthropic,openai,offline")`` returns a failover chain in that order.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from llm_gateway import Provider, ProviderError, ResilientProvider

from agent_platform.model import GroundedPolicyProvider

OPENAI_URL = "https://api.openai.com/v1"
ANTHROPIC_URL = "https://api.anthropic.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-haiku-latest"
PROVIDER_NAMES = ("offline", "openai", "anthropic")


class ProviderRequestError(RuntimeError):
    """A provider rejected the request itself (4xx other than 429). Never retried."""

    def __init__(self, provider: str, status: int, detail: str) -> None:
        super().__init__(f"{provider}: HTTP {status}: {detail}")
        self.provider = provider
        self.status = status


def _post_json(
    provider: str, url: str, headers: Mapping[str, str], body: dict[str, Any], timeout: float
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = _error_detail(error)
        if error.code == 429 or error.code >= 500:
            raise ProviderError(f"{provider}: HTTP {error.code}: {detail}") from None
        raise ProviderRequestError(provider, error.code, detail) from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        raise ProviderError(f"{provider}: {type(error).__name__}: {error}") from None
    except json.JSONDecodeError:
        raise ProviderError(f"{provider}: response was not JSON") from None
    if not isinstance(payload, dict):
        raise ProviderError(f"{provider}: response was not a JSON object")
    return payload


def _error_detail(error: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(error.read().decode("utf-8"))
    except (ValueError, OSError):
        return error.reason or "error"
    message = payload.get("error", payload) if isinstance(payload, dict) else payload
    if isinstance(message, dict):
        message = message.get("message", message)
    return str(message)[:300]


@dataclass
class UsageCounter:
    """Token usage reported by the provider, summed across calls."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class OpenAIProvider:
    """OpenAI Chat Completions over HTTPS."""

    model: str = DEFAULT_OPENAI_MODEL
    api_key: str = ""
    base_url: str = OPENAI_URL
    timeout: float = 30.0
    max_tokens: int = 512
    usage: UsageCounter = field(default_factory=UsageCounter)

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("OpenAI provider needs an API key (OPENAI_API_KEY)")

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def complete(self, system: str, prompt: str) -> str:
        payload = _post_json(
            "openai",
            f"{self.base_url.rstrip('/')}/chat/completions",
            {"authorization": f"Bearer {self.api_key}"},
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
            self.timeout,
        )
        try:
            text = payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise ProviderError("openai: response had no message content") from None
        usage = payload.get("usage") or {}
        self.usage.calls += 1
        self.usage.input_tokens += int(usage.get("prompt_tokens", 0))
        self.usage.output_tokens += int(usage.get("completion_tokens", 0))
        return str(text)


@dataclass
class AnthropicProvider:
    """Anthropic Messages API over HTTPS."""

    model: str = DEFAULT_ANTHROPIC_MODEL
    api_key: str = ""
    base_url: str = ANTHROPIC_URL
    timeout: float = 30.0
    max_tokens: int = 512
    usage: UsageCounter = field(default_factory=UsageCounter)

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("Anthropic provider needs an API key (ANTHROPIC_API_KEY)")

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    def complete(self, system: str, prompt: str) -> str:
        payload = _post_json(
            "anthropic",
            f"{self.base_url.rstrip('/')}/messages",
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
            self.timeout,
        )
        blocks = payload.get("content")
        if not isinstance(blocks, list):
            raise ProviderError("anthropic: response had no content blocks")
        usage = payload.get("usage") or {}
        self.usage.calls += 1
        self.usage.input_tokens += int(usage.get("input_tokens", 0))
        self.usage.output_tokens += int(usage.get("output_tokens", 0))
        return "".join(
            str(b.get("text", ""))
            for b in blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )


@dataclass(frozen=True)
class ProviderSettings:
    """Provider selection, read from ``AGENT_PLATFORM_PROVIDER`` and related variables."""

    chain: tuple[str, ...] = ("offline",)
    openai_model: str = DEFAULT_OPENAI_MODEL
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL
    openai_base_url: str = OPENAI_URL
    anthropic_base_url: str = ANTHROPIC_URL
    timeout: float = 30.0
    max_attempts: int = 3

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ProviderSettings:
        env = os.environ if environ is None else environ
        base = cls()
        return cls(
            chain=parse_chain(env.get("AGENT_PLATFORM_PROVIDER", ",".join(base.chain))),
            openai_model=env.get("AGENT_PLATFORM_OPENAI_MODEL", base.openai_model),
            anthropic_model=env.get("AGENT_PLATFORM_ANTHROPIC_MODEL", base.anthropic_model),
            openai_base_url=env.get("OPENAI_BASE_URL", base.openai_base_url),
            anthropic_base_url=env.get("ANTHROPIC_BASE_URL", base.anthropic_base_url),
            timeout=float(env.get("AGENT_PLATFORM_PROVIDER_TIMEOUT", base.timeout)),
            max_attempts=int(env.get("AGENT_PLATFORM_PROVIDER_ATTEMPTS", base.max_attempts)),
        )


def parse_chain(spec: str) -> tuple[str, ...]:
    """Parse ``"anthropic, openai"`` into a validated, de-duplicated provider chain."""
    names = [part.strip().lower() for part in spec.split(",") if part.strip()]
    if not names:
        raise ValueError("provider chain is empty")
    unknown = [n for n in names if n not in PROVIDER_NAMES]
    if unknown:
        raise ValueError(f"unknown provider(s): {', '.join(unknown)}; choose from {PROVIDER_NAMES}")
    return tuple(dict.fromkeys(names))


@dataclass
class NamedResilientProvider(ResilientProvider):
    """``ResilientProvider`` with a readable name for metrics and traces."""

    @property
    def name(self) -> str:
        return " > ".join(getattr(p, "name", type(p).__name__) for p in self.providers)


def build_provider(
    settings: ProviderSettings | None = None, environ: Mapping[str, str] | None = None
) -> Provider:
    """Create the configured provider; more than one name builds a failover chain."""
    settings = settings or ProviderSettings.from_env(environ)
    env = os.environ if environ is None else environ
    providers: list[Provider] = []
    for name in settings.chain:
        if name == "offline":
            providers.append(GroundedPolicyProvider())
        elif name == "openai":
            providers.append(
                OpenAIProvider(
                    model=settings.openai_model,
                    api_key=env.get("OPENAI_API_KEY", ""),
                    base_url=settings.openai_base_url,
                    timeout=settings.timeout,
                )
            )
        else:
            providers.append(
                AnthropicProvider(
                    model=settings.anthropic_model,
                    api_key=env.get("ANTHROPIC_API_KEY", ""),
                    base_url=settings.anthropic_base_url,
                    timeout=settings.timeout,
                )
            )
    if len(providers) == 1 and settings.chain == ("offline",):
        return providers[0]
    return NamedResilientProvider(
        providers,
        max_attempts=settings.max_attempts,
        backoff_base=0.5,
        max_backoff=8.0,
        sleep=time.sleep,
    )

"""Language-model provider interface and a Claude implementation."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Protocol


class Provider(Protocol):
    """A text-in, text-out language model."""

    def complete(self, system: str, prompt: str) -> str:
        """Return the model's reply to ``prompt`` under the ``system`` instructions."""


class AnthropicProvider:
    """Claude via the Anthropic Messages API, using only the standard library."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 2048,
        timeout: float = 30.0,
    ) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("set ANTHROPIC_API_KEY to use the Anthropic provider")
        self.api_key = key
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(self, system: str, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                # 2026-10-21 rejects sampling parameters; the body sends none.
                "anthropic-version": "2026-10-21",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return "".join(
            block.get("text", "")
            for block in payload.get("content", [])
            if block.get("type") == "text"
        )

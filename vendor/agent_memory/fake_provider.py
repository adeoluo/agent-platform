"""Deterministic offline provider for tests and continuous integration."""

from __future__ import annotations

from collections.abc import Callable

from agent_memory.extraction import rule_based_reply


class FakeProvider:
    """Offline stand-in for a language model.

    By default it answers extraction prompts with the JSON that the rule extractor would produce,
    so the LLM code path (prompting, JSON parsing, validation) is exercised without a network.
    Pass ``responder`` to script arbitrary replies, for example malformed JSON.
    """

    def __init__(self, responder: Callable[[str, str], str] | None = None) -> None:
        self._responder = responder or rule_based_reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        return self._responder(system, prompt)

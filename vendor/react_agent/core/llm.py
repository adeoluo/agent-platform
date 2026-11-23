"""The language model interface the agent reasons with.

``LanguageModel`` is a minimal :class:`~typing.Protocol`: given a prompt (and optional
stop sequences) it returns completion text. Keeping the interface this small means the
agent is decoupled from any provider - a deterministic scripted model used in tests and
a frontier API are interchangeable.

Stop sequences are part of the contract because the ReAct loop asks the model to emit a
single reasoning-and-action step at a time, halting generation at ``Observation:`` so the
tool result (not a hallucinated one) is appended next.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class LanguageModel(Protocol):
    """A text-completion model."""

    @property
    def name(self) -> str:
        """Short identifier used in traces and reports."""
        ...

    def complete(self, prompt: str, *, stop: Sequence[str] | None = None) -> str:
        """Return the completion for ``prompt``, truncated at any ``stop`` sequence."""
        ...

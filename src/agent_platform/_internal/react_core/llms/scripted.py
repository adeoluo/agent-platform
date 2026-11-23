"""A scripted language model for offline demonstrations and tests.

It replays a fixed list of completions in order. This lets the agent's behavior be
specified up front and run with no network access, which is what powers the hermetic
command-line demo and the test suite.
"""

from __future__ import annotations

from collections.abc import Sequence


class ScriptedModel:
    """Replays a fixed list of completions, one per :meth:`complete` call."""

    def __init__(self, completions: Sequence[str], *, name: str = "scripted") -> None:
        if not completions:
            raise ValueError("ScriptedModel needs at least one completion")
        self._completions = list(completions)
        self._index = 0
        self.name = name

    def complete(self, prompt: str, *, stop: Sequence[str] | None = None) -> str:
        if self._index >= len(self._completions):
            raise RuntimeError("ScriptedModel exhausted its completions")
        completion = self._completions[self._index]
        self._index += 1
        return completion

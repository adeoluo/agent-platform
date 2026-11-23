"""Tools: the actions a ReAct agent can take in the world.

A tool is any named, described callable that maps a string input to a string
observation. ``Tool`` is a :class:`~typing.Protocol`, so a plain function wrapped in
:class:`FunctionTool`, or any object with the right shape, is a first-class tool. The
``description`` matters as much as the code: it is what the model reads to decide when to
use the tool, so it is part of the public contract.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class ToolError(Exception):
    """Raised by a tool when it cannot complete its action.

    The agent catches this and feeds the message back to the model as an observation,
    giving the model a chance to recover rather than crashing the run.
    """


class RetryableToolError(ToolError):
    """A :class:`ToolError` the agent should retry before surfacing as an observation.

    Some failures are transient - a flaky lookup, a momentarily unavailable resource. A
    tool raises this (instead of a plain :class:`ToolError`) to tell the agent the action
    is worth retrying a bounded number of times. If every retry still fails, the agent
    turns the final error into an observation like any other, so the run still terminates.
    """


@runtime_checkable
class Tool(Protocol):
    """A named, described action mapping a string input to a string observation."""

    @property
    def name(self) -> str:
        """Short identifier the model uses to select the tool."""
        ...

    @property
    def description(self) -> str:
        """One-line explanation of what the tool does and its expected input."""
        ...

    def __call__(self, tool_input: str) -> str:
        """Execute the tool and return an observation."""
        ...


@dataclass(frozen=True)
class FunctionTool:
    """Adapts a plain ``str -> str`` function into a :class:`Tool`."""

    name: str
    description: str
    function: Callable[[str], str]

    def __call__(self, tool_input: str) -> str:
        return self.function(tool_input)


class ToolRegistry:
    """An ordered, name-indexed collection of tools.

    The registry rejects duplicate names so tool selection is always unambiguous, and it
    preserves insertion order so prompts list tools deterministically.
    """

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.add(tool)

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name!r}")
        self._tools[tool.name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(
                f"Unknown tool {name!r}. Available: {', '.join(self._tools) or '(none)'}"
            ) from None

    @property
    def names(self) -> list[str]:
        return list(self._tools)

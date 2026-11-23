"""A simple scratch-memory tool the agent can write to and read from.

Long-horizon tasks benefit from a place to stash intermediate facts. This tool is a
small key-value store the agent controls through two commands:

    set <key> = <value>
    get <key>

The state lives on the tool instance, so a fresh tool means a fresh memory; sharing one
instance across a run gives the agent persistent working memory for that run.
"""

from __future__ import annotations

from react_agent.core.tool import ToolError


class MemoryTool:
    """A per-instance key-value scratchpad."""

    name = "memory"
    description = (
        "Store and recall facts. Use 'set <key> = <value>' to store, or 'get <key>' to recall."
    )

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    @property
    def store(self) -> dict[str, str]:
        """A read-only view of the current contents (copy)."""
        return dict(self._store)

    def __call__(self, tool_input: str) -> str:
        command = tool_input.strip()
        lowered = command.lower()
        if lowered.startswith("set "):
            return self._set(command[4:])
        if lowered.startswith("get "):
            return self._get(command[4:])
        raise ToolError("memory expects 'set <key> = <value>' or 'get <key>'")

    def _set(self, body: str) -> str:
        if "=" not in body:
            raise ToolError("memory 'set' requires '<key> = <value>'")
        key, value = body.split("=", 1)
        key = key.strip()
        if not key:
            raise ToolError("memory 'set' requires a non-empty key")
        self._store[key] = value.strip()
        return f"stored {key!r}"

    def _get(self, body: str) -> str:
        key = body.strip()
        if key in self._store:
            return self._store[key]
        return f"no value stored for {key!r}"

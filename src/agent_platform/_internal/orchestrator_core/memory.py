"""Shared memory for agents.

The blackboard records the structured plan, results, and reviews. Agents sometimes also
need a free-form scratchpad to pass intermediate facts that are not themselves subtask
outputs - a running list, a computed constant, a decision. :class:`SharedMemory` is a small
key-value store with namespacing, so a worker can stash and recall values across subtasks
without encoding them into its output string.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


class SharedMemory:
    """A namespaced key-value scratchpad shared across agents in a run."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def set(self, key: str, value: Any, *, namespace: str = "") -> None:
        """Store ``value`` under ``key`` in an optional ``namespace``."""
        self._store[self._key(namespace, key)] = value

    def get(self, key: str, *, namespace: str = "", default: Any = None) -> Any:
        """Return the value for ``key`` in ``namespace``, or ``default`` if absent."""
        return self._store.get(self._key(namespace, key), default)

    def has(self, key: str, *, namespace: str = "") -> bool:
        """Return whether ``key`` exists in ``namespace``."""
        return self._key(namespace, key) in self._store

    def append(self, key: str, value: Any, *, namespace: str = "") -> None:
        """Append ``value`` to a list stored under ``key`` (creating it if needed)."""
        full = self._key(namespace, key)
        current = self._store.get(full)
        if current is None:
            self._store[full] = [value]
        elif isinstance(current, list):
            current.append(value)
        else:
            raise TypeError(f"value at {key!r} is not a list")

    def keys(self, *, namespace: str = "") -> Iterator[str]:
        """Yield the keys in ``namespace`` (without the namespace prefix)."""
        prefix = f"{namespace}:" if namespace else ""
        for stored in self._store:
            if namespace:
                if stored.startswith(prefix):
                    yield stored[len(prefix) :]
            elif ":" not in stored:
                yield stored

    def __len__(self) -> int:
        return len(self._store)

    @staticmethod
    def _key(namespace: str, key: str) -> str:
        return f"{namespace}:{key}" if namespace else key

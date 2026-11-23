"""Observability hooks for an agent run.

A run emits events as it progresses: a step begins, an action is chosen, an observation
returns, the agent finishes. :class:`Callbacks` is a small set of optional functions that
receive these events, enabling logging, metrics, tracing, or progress display without
touching the control loop. Every hook defaults to a no-op, so callers wire only what they
need.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from react_agent.core.steps import AgentAction, AgentFinish


def _noop_step(index: int) -> None:  # pragma: no cover - trivial default
    return None


def _noop_action(index: int, action: AgentAction) -> None:  # pragma: no cover
    return None


def _noop_observation(index: int, observation: str) -> None:  # pragma: no cover
    return None


def _noop_tool_retry(index: int, attempt: int, error: str) -> None:  # pragma: no cover
    return None


def _noop_finish(finish: AgentFinish) -> None:  # pragma: no cover
    return None


@dataclass(frozen=True)
class Callbacks:
    """Optional hooks invoked during a run.

    Attributes:
        on_step: Called at the start of each step with its 1-based index.
        on_action: Called when the model chooses a tool action.
        on_observation: Called with each tool observation.
        on_tool_retry: Called when a retryable tool failure is about to be retried, with
            the step index, the (1-based) failed attempt number, and the error message.
        on_finish: Called once when the agent produces a final answer.
    """

    on_step: Callable[[int], None] = _noop_step
    on_action: Callable[[int, AgentAction], None] = _noop_action
    on_observation: Callable[[int, str], None] = _noop_observation
    on_tool_retry: Callable[[int, int, str], None] = _noop_tool_retry
    on_finish: Callable[[AgentFinish], None] = _noop_finish

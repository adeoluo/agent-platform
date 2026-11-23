"""A logging-callbacks factory.

Turns the agent's observability hooks into human-readable log lines. This is the simplest
useful consumer of :class:`~react_agent.core.callbacks.Callbacks` and doubles as a verbose
mode for the command-line interface: pass the returned callbacks to an agent and each
step, action, and observation is printed as it happens.
"""

from __future__ import annotations

from collections.abc import Callable

from react_agent.core.callbacks import Callbacks
from react_agent.core.steps import AgentAction, AgentFinish


def logging_callbacks(sink: Callable[[str], None] = print) -> Callbacks:
    """Return :class:`Callbacks` that log each event to ``sink``.

    Args:
        sink: Where to write log lines; defaults to :func:`print`.
    """

    def on_step(index: int) -> None:
        sink(f"[step {index}] thinking...")

    def on_action(index: int, action: AgentAction) -> None:
        sink(f"[step {index}] action: {action.tool}({action.tool_input!r})")

    def on_observation(index: int, observation: str) -> None:
        sink(f"[step {index}] observation: {observation}")

    def on_finish(finish: AgentFinish) -> None:
        sink(f"[finish] {finish.answer}")

    return Callbacks(
        on_step=on_step,
        on_action=on_action,
        on_observation=on_observation,
        on_finish=on_finish,
    )

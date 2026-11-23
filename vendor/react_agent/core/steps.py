"""Typed records for the agent's reasoning trace.

A core goal of this project is that every run is fully inspectable. The agent emits
structured steps - a decision to call a tool (:class:`AgentAction`) or to stop
(:class:`AgentFinish`) - and records each action together with its observation in a
:class:`TraceStep`. The complete :class:`AgentResult` is therefore an auditable account
of exactly how the agent reached its answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class AgentAction:
    """A decision to invoke a tool."""

    thought: str
    tool: str
    tool_input: str
    raw: str = ""


@dataclass(frozen=True, slots=True)
class AgentFinish:
    """A decision to stop and return a final answer."""

    thought: str
    answer: str
    raw: str = ""


@dataclass(frozen=True, slots=True)
class TraceStep:
    """One completed action together with the observation it produced."""

    action: AgentAction
    observation: str


@dataclass(frozen=True, slots=True)
class AgentResult:
    """The outcome of an agent run.

    Attributes:
        question: The original question.
        answer: The final answer, or an empty string if the step limit was hit.
        steps: The ordered action/observation steps taken.
        finished: ``True`` if the agent produced a final answer within the step limit.
        model_name: The language model used, for reporting.
    """

    question: str
    answer: str
    steps: Sequence[TraceStep] = field(default_factory=tuple)
    finished: bool = True
    model_name: str = ""

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def transcript(self) -> str:
        """Render the run as a readable ReAct transcript."""
        lines: list[str] = [f"Question: {self.question}"]
        for step in self.steps:
            lines.append(f"Thought: {step.action.thought}")
            lines.append(f"Action: {step.action.tool}")
            lines.append(f"Action Input: {step.action.tool_input}")
            lines.append(f"Observation: {step.observation}")
        if self.finished:
            lines.append(f"Final Answer: {self.answer}")
        else:
            lines.append("(stopped: step limit reached)")
        return "\n".join(lines)

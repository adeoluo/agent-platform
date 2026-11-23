"""A deterministic rule-based policy model for offline benchmarking.

To report a real agent success rate without a network, we need a model that actually
drives the ReAct loop to correct answers on the benchmark suite. :class:`RulePolicyModel`
inspects the running prompt and emits the next ReAct step: it builds a small ordered *plan*
of tool calls from the question (arithmetic, capital-of, unit conversion), issues them one
at a time, and once every planned observation is in, reads them back to produce a final
answer. Because the plan can contain more than one step, the model solves genuinely
multi-step questions (for example "the capital of France *and* 21 times 2"), which a
single-shot answer cannot.

It is intentionally narrow and transparent - a stand-in for a capable model that keeps the
benchmark deterministic and offline. A frontier model would replace it unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

_CAPITAL = re.compile(r"capital of[ \t]+([A-Za-z]+)", re.IGNORECASE)
_CONVERT = re.compile(r"convert[ \t]+([^?\n]+)", re.IGNORECASE)
_LAST_OBSERVATION = re.compile(r"Observation:[ \t]*([^\n]*)")
_ARITHMETIC = re.compile(r"^[0-9+\-*/()\s]+$")


@dataclass(frozen=True, slots=True)
class _PlanStep:
    """One planned tool call: which tool and what input to pass it."""

    tool: str
    tool_input: str


def _iter_calculations(question: str) -> Iterator[tuple[int, str]]:
    """Yield ``(position, expression)`` for each "what is <expression>" arithmetic clause.

    Implemented with linear string scanning (no backtracking regex) because the question
    is untrusted input. Each clause is cut at a following "and"/comma so a compound
    question does not swallow the next sub-question into one expression.
    """
    lowered = question.lower()
    marker = "what is"
    start = 0
    while True:
        index = lowered.find(marker, start)
        if index < 0:
            return
        start = index + len(marker)
        rest = question[start:]
        if not rest[:1].isspace():
            continue
        end = rest.find("?")
        segment = rest[:end] if end >= 0 else rest
        segment = re.split(r",?\s+and\s+", segment, maxsplit=1)[0]
        expression = segment.strip()
        if expression and any(ch.isdigit() for ch in expression) and _ARITHMETIC.match(expression):
            yield index, expression


def _plan(question: str) -> list[_PlanStep]:
    """Build an ordered list of tool calls that answer ``question``."""
    found: list[tuple[int, _PlanStep]] = []
    for match in _CAPITAL.finditer(question):
        found.append((match.start(), _PlanStep("knowledge_base", f"capital of {match.group(1)}")))
    for position, expression in _iter_calculations(question):
        found.append((position, _PlanStep("calculator", expression)))
    for match in _CONVERT.finditer(question):
        found.append((match.start(), _PlanStep("convert_units", match.group(1).strip())))
    found.sort(key=lambda item: item[0])
    return [step for _, step in found]


@dataclass
class RulePolicyModel:
    """Emits ReAct steps by planning tool calls and reading back their observations."""

    name: str = "rule-policy"

    def complete(self, prompt: str, *, stop: Sequence[str] | None = None) -> str:
        question = _extract_question(prompt)
        observations = _LAST_OBSERVATION.findall(prompt)
        plan = _plan(question)

        if not plan:
            return "Thought: I am not sure.\nFinal Answer: unknown"

        # Issue the next planned tool call that has not produced an observation yet.
        if len(observations) < len(plan):
            step = plan[len(observations)]
            ordinal = len(observations) + 1
            return (
                f"Thought: step {ordinal} of {len(plan)}.\n"
                f"Action: {step.tool}\nAction Input: {step.tool_input}\n"
            )

        # Every planned step is done: combine the observations into the final answer.
        answer = "; ".join(observation.strip() for observation in observations)
        return f"Thought: I have the results.\nFinal Answer: {answer}"


def _extract_question(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.startswith("Question:"):
            return line.split(":", 1)[1].strip()
    return ""

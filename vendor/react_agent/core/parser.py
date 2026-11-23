"""Parsing the model's ReAct output into a structured decision.

The agent prompts the model to emit one of two shapes:

    Thought: <reasoning>
    Action: <tool name>
    Action Input: <input>

or:

    Thought: <reasoning>
    Final Answer: <answer>

The parser is deliberately lenient about surrounding whitespace and casing of the field
labels, because models vary, but strict about the structure so malformed output is caught
and surfaced (the agent turns a :class:`ParseError` into corrective feedback rather than
crashing).
"""

from __future__ import annotations

import re

from react_agent.core.steps import AgentAction, AgentFinish

_THOUGHT = re.compile(
    r"Thought:[ \t]*([^\n]*(?:\n(?![ \t]*(?:Action|Final Answer)[ \t]*:)[^\n]*)*)",
    re.IGNORECASE,
)
_ACTION = re.compile(r"Action:[ \t]*([^\n]*)", re.IGNORECASE)
_ACTION_INPUT = re.compile(
    r"Action Input:[ \t]*([^\n]*(?:\n(?![ \t]*Observation[ \t]*:)[^\n]*)*)", re.IGNORECASE
)
_FINAL = re.compile(r"Final Answer:[ \t]*(.*)", re.IGNORECASE | re.DOTALL)


class ParseError(ValueError):
    """Raised when the model output matches neither an action nor a final answer."""


def parse_output(text: str) -> AgentAction | AgentFinish:
    """Parse one ReAct step from ``text``.

    Returns:
        An :class:`AgentFinish` if a final answer is present, otherwise an
        :class:`AgentAction`.

    Raises:
        ParseError: If neither a final answer nor a complete action is found.
    """
    thought_match = _THOUGHT.search(text)
    thought = thought_match.group(1).strip() if thought_match else ""

    final_match = _FINAL.search(text)
    if final_match:
        return AgentFinish(thought=thought, answer=final_match.group(1).strip(), raw=text)

    action_match = _ACTION.search(text if text.endswith("\n") else text + "\n")
    input_match = _ACTION_INPUT.search(text)
    if action_match and input_match:
        return AgentAction(
            thought=thought,
            tool=action_match.group(1).strip(),
            tool_input=input_match.group(1).strip(),
            raw=text,
        )

    raise ParseError(
        "Expected an 'Action:' with 'Action Input:' or a 'Final Answer:'. "
        f"Got: {text.strip()[:200]!r}"
    )

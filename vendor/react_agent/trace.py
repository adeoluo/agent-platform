"""Serializing an agent run to JSON and back.

A trace is only auditable if it can be stored and shared. This module serializes an
:class:`~react_agent.core.steps.AgentResult` - question, every thought/action/observation,
the final answer, and metadata - to a JSON-ready structure and reconstructs it losslessly.
This is the substrate for trace viewers, regression comparisons, and offline analysis.
"""

from __future__ import annotations

import json
from typing import Any

from react_agent.core.steps import AgentAction, AgentResult, TraceStep


def trace_to_dict(result: AgentResult) -> dict[str, Any]:
    """Serialize an :class:`AgentResult` to a JSON-ready dictionary."""
    return {
        "question": result.question,
        "answer": result.answer,
        "finished": result.finished,
        "model_name": result.model_name,
        "steps": [
            {
                "thought": step.action.thought,
                "tool": step.action.tool,
                "tool_input": step.action.tool_input,
                "raw": step.action.raw,
                "observation": step.observation,
            }
            for step in result.steps
        ],
    }


def trace_from_dict(payload: dict[str, Any]) -> AgentResult:
    """Reconstruct an :class:`AgentResult` from :func:`trace_to_dict` output."""
    steps = tuple(
        TraceStep(
            action=AgentAction(
                thought=item["thought"],
                tool=item["tool"],
                tool_input=item["tool_input"],
                raw=item.get("raw", ""),
            ),
            observation=item["observation"],
        )
        for item in payload["steps"]
    )
    return AgentResult(
        question=payload["question"],
        answer=payload["answer"],
        steps=steps,
        finished=payload["finished"],
        model_name=payload.get("model_name", ""),
    )


def trace_to_json(result: AgentResult, *, indent: int = 2) -> str:
    """Serialize an :class:`AgentResult` to a JSON string."""
    return json.dumps(trace_to_dict(result), indent=indent)


def trace_from_json(text: str) -> AgentResult:
    """Parse an :class:`AgentResult` from a JSON string."""
    return trace_from_dict(json.loads(text))

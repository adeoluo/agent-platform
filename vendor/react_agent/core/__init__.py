"""Core abstractions for the ReAct agent.

The agent is built from small, independently testable pieces:

- :class:`~react_agent.core.tool.Tool` and :class:`~react_agent.core.tool.ToolRegistry` -
  the actions available to the agent.
- :class:`~react_agent.core.llm.LanguageModel` - the reasoning model, a thin Protocol.
- :mod:`react_agent.core.parser` - turns model output into a structured decision.
- :mod:`react_agent.core.prompt` - assembles the ReAct prompt.
- :class:`~react_agent.core.agent.ReActAgent` - the control loop tying them together.
"""

from __future__ import annotations

from react_agent.core.agent import ReActAgent
from react_agent.core.callbacks import Callbacks
from react_agent.core.llm import LanguageModel
from react_agent.core.parser import ParseError, parse_output
from react_agent.core.steps import AgentAction, AgentFinish, AgentResult, TraceStep
from react_agent.core.tool import (
    FunctionTool,
    RetryableToolError,
    Tool,
    ToolError,
    ToolRegistry,
)

__all__ = [
    "AgentAction",
    "AgentFinish",
    "AgentResult",
    "Callbacks",
    "FunctionTool",
    "LanguageModel",
    "ParseError",
    "ReActAgent",
    "RetryableToolError",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "TraceStep",
    "parse_output",
]

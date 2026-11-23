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

from agent_platform._internal.react_core.core.agent import ReActAgent
from agent_platform._internal.react_core.core.callbacks import Callbacks
from agent_platform._internal.react_core.core.llm import LanguageModel
from agent_platform._internal.react_core.core.parser import ParseError, parse_output
from agent_platform._internal.react_core.core.steps import (
    AgentAction,
    AgentFinish,
    AgentResult,
    TraceStep,
)
from agent_platform._internal.react_core.core.tool import (
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

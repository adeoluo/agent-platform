"""Built-in tools and ready-made toolsets.

The tools are deliberately safe and offline:

- **calculator** - arithmetic via an AST allow-list (never ``eval``).
- **search** - fuzzy lookup over a bundled knowledge base.
- **knowledge_base** - structured attribute lookup (``capital of France``).
- **memory** - a per-run key-value scratchpad.
- **date_difference** / **weekday** - deterministic date reasoning.
- **convert_units** - length, mass, and temperature conversion.
- **text_length** / **text_transform** / **count_occurrences** - string utilities.

Together they are enough to demonstrate rich multi-step reasoning with no network access.
"""

from __future__ import annotations

from agent_platform._internal.react_core.core.tool import Tool, ToolRegistry
from agent_platform._internal.react_core.tools.calculator import CalculatorTool
from agent_platform._internal.react_core.tools.datetime_tools import DateDifferenceTool, WeekdayTool
from agent_platform._internal.react_core.tools.knowledge_base import KnowledgeBaseTool
from agent_platform._internal.react_core.tools.memory import MemoryTool
from agent_platform._internal.react_core.tools.search import SearchTool
from agent_platform._internal.react_core.tools.text import (
    CountOccurrencesTool,
    TextLengthTool,
    TextTransformTool,
)
from agent_platform._internal.react_core.tools.units import UnitConversionTool


def default_toolset() -> ToolRegistry:
    """The core toolset: calculator, search, and a fresh per-run memory."""
    return ToolRegistry([CalculatorTool(), SearchTool(), MemoryTool()])


def full_toolset() -> ToolRegistry:
    """Every built-in tool, with a fresh per-run memory."""
    tools: list[Tool] = [
        CalculatorTool(),
        SearchTool(),
        KnowledgeBaseTool(),
        MemoryTool(),
        DateDifferenceTool(),
        WeekdayTool(),
        UnitConversionTool(),
        TextLengthTool(),
        TextTransformTool(),
        CountOccurrencesTool(),
    ]
    return ToolRegistry(tools)


__all__ = [
    "CalculatorTool",
    "CountOccurrencesTool",
    "DateDifferenceTool",
    "KnowledgeBaseTool",
    "MemoryTool",
    "SearchTool",
    "TextLengthTool",
    "TextTransformTool",
    "UnitConversionTool",
    "WeekdayTool",
    "default_toolset",
    "full_toolset",
]

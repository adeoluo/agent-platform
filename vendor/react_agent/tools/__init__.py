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

from react_agent.core.tool import Tool, ToolRegistry
from react_agent.tools.calculator import CalculatorTool
from react_agent.tools.datetime_tools import DateDifferenceTool, WeekdayTool
from react_agent.tools.knowledge_base import KnowledgeBaseTool
from react_agent.tools.memory import MemoryTool
from react_agent.tools.search import SearchTool
from react_agent.tools.text import (
    CountOccurrencesTool,
    TextLengthTool,
    TextTransformTool,
)
from react_agent.tools.units import UnitConversionTool


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

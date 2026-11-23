"""String utility tools.

Small, composable text operations an agent can chain: measuring length, reversing,
changing case, and counting word or character occurrences. Each is a single, predictable
transformation, which keeps the agent's reasoning about them simple.
"""

from __future__ import annotations

from dataclasses import dataclass

from react_agent.core.tool import ToolError


@dataclass(frozen=True)
class TextLengthTool:
    """Count the characters and words in the input."""

    name: str = "text_length"
    description: str = "Count characters and words. Input: any text."

    def __call__(self, tool_input: str) -> str:
        characters = len(tool_input)
        words = len(tool_input.split())
        return f"{characters} characters, {words} words"


@dataclass(frozen=True)
class TextTransformTool:
    """Transform text: reverse, upper, lower, or title case.

    Input format: '<operation>: <text>' where operation is one of reverse, upper,
    lower, title.
    """

    name: str = "text_transform"
    description: str = (
        "Transform text. Input: '<op>: <text>' where op is reverse, upper, lower, or title."
    )

    def __call__(self, tool_input: str) -> str:
        if ":" not in tool_input:
            raise ToolError("text_transform expects '<op>: <text>'")
        operation, text = tool_input.split(":", 1)
        operation = operation.strip().lower()
        text = text.strip()
        if operation == "reverse":
            return text[::-1]
        if operation == "upper":
            return text.upper()
        if operation == "lower":
            return text.lower()
        if operation == "title":
            return text.title()
        raise ToolError(f"unknown operation {operation!r}")


@dataclass(frozen=True)
class CountOccurrencesTool:
    """Count how many times a substring appears in text.

    Input format: '<needle> in <haystack>'.
    """

    name: str = "count_occurrences"
    description: str = "Count substring occurrences. Input: '<needle> in <haystack>'."

    def __call__(self, tool_input: str) -> str:
        if " in " not in tool_input:
            raise ToolError("count_occurrences expects '<needle> in <haystack>'")
        needle, haystack = tool_input.split(" in ", 1)
        needle = needle.strip()
        if not needle:
            raise ToolError("needle must be non-empty")
        return str(haystack.count(needle))

"""Date and time tools.

Agents frequently need to reason about dates - "what day of the week is 2026-01-01",
"how many days between two dates". These tools are deterministic given an injected
reference date, so tests and demonstrations are reproducible rather than depending on the
wall clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from react_agent.core.tool import ToolError


def _parse_date(text: str) -> date:
    try:
        return datetime.strptime(text.strip(), "%Y-%m-%d").date()
    except ValueError:
        raise ToolError(f"expected a date as YYYY-MM-DD, got {text.strip()!r}") from None


@dataclass(frozen=True)
class DateDifferenceTool:
    """Compute the number of days between two ISO dates."""

    name: str = "date_difference"
    description: str = (
        "Return the number of days between two dates. Input: 'YYYY-MM-DD to YYYY-MM-DD'."
    )

    def __call__(self, tool_input: str) -> str:
        if " to " not in tool_input:
            raise ToolError("date_difference expects 'YYYY-MM-DD to YYYY-MM-DD'")
        left, right = tool_input.split(" to ", 1)
        start = _parse_date(left)
        end = _parse_date(right)
        return str((end - start).days)


@dataclass(frozen=True)
class WeekdayTool:
    """Return the weekday name for an ISO date."""

    name: str = "weekday"
    description: str = "Return the day of the week for a date. Input: 'YYYY-MM-DD'."

    def __call__(self, tool_input: str) -> str:
        return _parse_date(tool_input).strftime("%A")

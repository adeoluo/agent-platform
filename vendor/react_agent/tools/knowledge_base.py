"""A structured knowledge-base tool with exact-key lookup.

Where :mod:`react_agent.tools.search` does fuzzy keyword retrieval, this tool exposes a
small structured encyclopedia with precise, attribute-level lookups ("population of
France", "capital of Japan"). It demonstrates an agent composing structured queries and
is fully deterministic and offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from react_agent.core.tool import ToolError

_FACTS: dict[str, dict[str, str]] = {
    "france": {"capital": "Paris", "population": "67 million", "continent": "Europe"},
    "japan": {"capital": "Tokyo", "population": "125 million", "continent": "Asia"},
    "brazil": {"capital": "Brasilia", "population": "214 million", "continent": "South America"},
    "egypt": {"capital": "Cairo", "population": "104 million", "continent": "Africa"},
    "canada": {"capital": "Ottawa", "population": "38 million", "continent": "North America"},
    "australia": {"capital": "Canberra", "population": "26 million", "continent": "Oceania"},
}


@dataclass(frozen=True)
class KnowledgeBaseTool:
    """Look up an attribute of an entity. Input: '<attribute> of <entity>'."""

    name: str = "knowledge_base"
    description: str = (
        "Look up a structured fact. Input: '<attribute> of <entity>', for example "
        "'capital of France'. Attributes: capital, population, continent."
    )
    _facts: dict[str, dict[str, str]] = field(default_factory=lambda: _FACTS, repr=False)

    def __call__(self, tool_input: str) -> str:
        if " of " not in tool_input:
            raise ToolError("knowledge_base expects '<attribute> of <entity>'")
        attribute, entity = tool_input.split(" of ", 1)
        attribute = attribute.strip().lower()
        entity = entity.strip().lower().rstrip("?.")
        if entity not in self._facts:
            return f"No entry for {entity!r}."
        record = self._facts[entity]
        if attribute not in record:
            known = ", ".join(sorted(record))
            return f"No attribute {attribute!r} for {entity!r}. Known: {known}."
        return record[attribute]

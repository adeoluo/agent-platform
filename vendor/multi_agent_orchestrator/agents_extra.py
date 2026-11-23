"""A tool-using worker and additional planners.

- :class:`ToolWorker` - a worker that runs a registered tool (a named ``str -> str``
  callable) chosen by the subtask's metadata, demonstrating that workers can take real
  actions, not just transform text.
- :class:`ExplicitPlanner` - returns a fixed, pre-authored plan, useful when the
  decomposition is known and the planning step should be deterministic.
- :class:`TemplatePlanner` - expands a goal into a fixed sequence of role-prefixed
  subtasks ("research:", "draft:", "review:"), a common real-world planning template.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from multi_agent_orchestrator.core.types import Subtask, WorkerResult

Tool = Callable[[str], str]


@dataclass(frozen=True)
class ToolWorker:
    """Executes a subtask by invoking a named tool from a registry.

    The subtask names its tool and argument via metadata keys ``tool`` and ``arg``.
    """

    tools: Mapping[str, Tool]
    name: str = "tool-worker"

    def execute(self, subtask: Subtask, context: Mapping[str, WorkerResult]) -> WorkerResult:
        tool_name = str(subtask.metadata.get("tool", ""))
        argument = str(subtask.metadata.get("arg", subtask.description))
        tool = self.tools.get(tool_name)
        if tool is None:
            return WorkerResult(
                subtask_id=subtask.id,
                worker=self.name,
                output=f"error: unknown tool {tool_name!r}",
                success=False,
            )
        return WorkerResult(subtask_id=subtask.id, worker=self.name, output=tool(argument))


@dataclass(frozen=True)
class ExplicitPlanner:
    """Returns a fixed, pre-authored list of subtasks regardless of the goal."""

    subtasks: Sequence[Subtask]

    def plan(self, goal: str) -> Sequence[Subtask]:
        return list(self.subtasks)


@dataclass(frozen=True)
class TemplatePlanner:
    """Expands a goal into a fixed sequence of role-prefixed, chained subtasks."""

    roles: Sequence[str] = field(default=("research", "draft", "review"))

    def plan(self, goal: str) -> Sequence[Subtask]:
        subtasks: list[Subtask] = []
        for index, role in enumerate(self.roles):
            depends = (f"task-{index - 1}",) if index > 0 else ()
            subtasks.append(
                Subtask(
                    id=f"task-{index}",
                    description=f"{role}: {goal}",
                    depends_on=depends,
                    metadata={"role": role},
                )
            )
        return subtasks

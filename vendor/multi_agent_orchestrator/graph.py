"""Dependency-graph validation.

A plan is a directed graph: subtasks are nodes and dependencies are edges. Before executing
it, the orchestrator should know the graph is well-formed - every dependency refers to a
real subtask, and there are no cycles (which would deadlock the scheduler). This module
validates the plan and computes a topological order, so planning errors surface immediately
rather than as a stuck run.
"""

from __future__ import annotations

from collections.abc import Sequence

from multi_agent_orchestrator.core.types import Subtask


class PlanError(ValueError):
    """Raised when a plan's dependency graph is malformed."""


def validate_plan(subtasks: Sequence[Subtask]) -> None:
    """Validate that dependencies reference real subtasks and the graph is acyclic.

    Raises:
        PlanError: On a missing dependency or a dependency cycle.
    """
    ids = {subtask.id for subtask in subtasks}
    if len(ids) != len(subtasks):
        raise PlanError("duplicate subtask ids in plan")
    for subtask in subtasks:
        for dependency in subtask.depends_on:
            if dependency not in ids:
                raise PlanError(f"subtask {subtask.id!r} depends on unknown subtask {dependency!r}")
    # Cycle detection via topological sort; if it cannot order every node, there is a cycle.
    topological_order(subtasks)  # raises on cycle


def topological_order(subtasks: Sequence[Subtask]) -> list[str]:
    """Return subtask ids in a dependency-respecting order (Kahn's algorithm).

    Raises:
        PlanError: If the graph contains a cycle.
    """
    by_id = {subtask.id: subtask for subtask in subtasks}
    in_degree = {subtask.id: 0 for subtask in subtasks}
    dependents: dict[str, list[str]] = {subtask.id: [] for subtask in subtasks}
    for subtask in subtasks:
        for dependency in subtask.depends_on:
            if dependency in by_id:
                in_degree[subtask.id] += 1
                dependents[dependency].append(subtask.id)

    # Start with zero-dependency nodes, in stable id order for determinism.
    ready = sorted(sid for sid, degree in in_degree.items() if degree == 0)
    order: list[str] = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for dependent in dependents[current]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                ready.append(dependent)
        ready.sort()

    if len(order) != len(subtasks):
        raise PlanError("plan contains a dependency cycle")
    return order

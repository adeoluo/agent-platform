"""Serializing an orchestrated run to JSON.

A run's blackboard - its plan, results, reviews, and message log - is a complete, auditable
record. :func:`run_to_dict` serializes it losslessly so a run can be stored, shared, or
replayed for analysis, and :func:`run_from_messages` rebuilds the ordered message log for a
viewer. This is the orchestration analogue of a trace.
"""

from __future__ import annotations

import json
from typing import Any

from multi_agent_orchestrator.orchestrator import OrchestrationResult


def run_to_dict(result: OrchestrationResult) -> dict[str, Any]:
    """Serialize an orchestration result to a JSON-ready dictionary."""
    board = result.blackboard
    return {
        "goal": result.goal,
        "completed": result.completed,
        "subtasks": [
            {
                "id": subtask.id,
                "description": subtask.description,
                "depends_on": list(subtask.depends_on),
                "metadata": subtask.metadata,
                "complete": board.is_complete(subtask.id),
            }
            for subtask in board.subtasks
        ],
        "results": {
            subtask_id: {
                "worker": worker_result.worker,
                "output": worker_result.output,
                "success": worker_result.success,
            }
            for subtask_id, worker_result in board.results.items()
        },
        "messages": [
            {
                "role": message.role.value,
                "content": message.content,
                "subtask_id": message.subtask_id,
                "metadata": message.metadata,
            }
            for message in board.messages
        ],
    }


def run_to_json(result: OrchestrationResult, *, indent: int = 2) -> str:
    """Serialize an orchestration result to a JSON string."""
    return json.dumps(run_to_dict(result), indent=indent)

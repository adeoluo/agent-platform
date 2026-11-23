"""Multi-Agent Orchestrator - planner, workers, and a critic over a blackboard.

Public API: the core types, the blackboard, and the agent protocols. The orchestrator
loop and built-in agents live in their own modules.
"""

from __future__ import annotations

from multi_agent_orchestrator.comparison import (
    ApproachMetrics,
    Comparison,
    ComparisonSummary,
    compare_goal,
    compare_suite,
)
from multi_agent_orchestrator.core import (
    Blackboard,
    Critic,
    Message,
    Planner,
    Review,
    Role,
    Subtask,
    Worker,
    WorkerResult,
)
from multi_agent_orchestrator.durable import (
    CheckpointStatus,
    CrashError,
    inspect_checkpoint,
    load_checkpoint,
    resume_durable,
    run_durable,
    save_checkpoint,
)

__version__ = "0.1.0"

__all__ = [
    "ApproachMetrics",
    "Blackboard",
    "CheckpointStatus",
    "Comparison",
    "ComparisonSummary",
    "CrashError",
    "Critic",
    "Message",
    "Planner",
    "Review",
    "Role",
    "Subtask",
    "Worker",
    "WorkerResult",
    "__version__",
    "compare_goal",
    "compare_suite",
    "inspect_checkpoint",
    "load_checkpoint",
    "resume_durable",
    "run_durable",
    "save_checkpoint",
]

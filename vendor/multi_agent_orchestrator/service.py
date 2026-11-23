"""A standard-library HTTP service exposing the orchestrator.

The service runs the planner/workers/critic orchestration over a goal and
returns the structured run (subtasks, outputs, blackboard messages) plus
whether it completed. A pure handler is separated from the socket binding for
testing.

Endpoints
---------
``GET  /health``   liveness, worker names, and planner name.
``POST /run``      body ``{"goal": str}``; returns the orchestrated run.
``POST /compare``  body ``{"goal": str}``; compares single-agent and multi-agent.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from multi_agent_orchestrator.agents import (
    EchoWorker,
    KeywordPlanner,
    LengthCritic,
    UppercaseWorker,
)
from multi_agent_orchestrator.baseline import SingleAgent
from multi_agent_orchestrator.observability import ServiceMetrics
from multi_agent_orchestrator.orchestrator import Orchestrator
from multi_agent_orchestrator.trace import run_to_dict

JsonObject = dict[str, Any]
_MAX_GOAL_CHARS = 2000


def _build_orchestrator() -> Orchestrator:
    return Orchestrator(
        KeywordPlanner(chain_dependencies=True),
        [EchoWorker(), UppercaseWorker()],
        LengthCritic(),
    )


@dataclass(frozen=True)
class OrchestrationService:
    """Runs orchestration and single-agent comparison over goals."""

    worker_names: tuple[str, ...]
    planner_name: str
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)

    @classmethod
    def build(cls) -> OrchestrationService:
        return cls(
            worker_names=(EchoWorker().name, UppercaseWorker().name),
            planner_name="keyword-planner",
        )

    def run(self, goal: str) -> JsonObject:
        return run_to_dict(_build_orchestrator().run(goal))

    def compare(self, goal: str) -> JsonObject:
        single = SingleAgent(EchoWorker()).run(goal)
        multi = _build_orchestrator().run(goal)
        return {
            "goal": goal,
            "single_agent": {"output": single.output, "subtasks": single.subtasks},
            "multi_agent": {
                "outputs": multi.final_outputs(),
                "subtasks": multi.num_subtasks,
                "completed": multi.completed,
            },
        }


def _validate_goal(body: Mapping[str, Any]) -> str:
    goal = body.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("'goal' must be a non-empty string")
    if len(goal) > _MAX_GOAL_CHARS:
        raise ValueError(f"'goal' exceeds {_MAX_GOAL_CHARS} characters")
    return goal


def _route(
    method: str, path: str, body: Mapping[str, Any], service: OrchestrationService
) -> tuple[int, JsonObject]:
    if method == "GET" and path == "/health":
        return 200, {
            "status": "ok",
            "workers": list(service.worker_names),
            "planner": service.planner_name,
        }
    if method == "POST" and path in ("/run", "/compare"):
        try:
            goal = _validate_goal(body)
        except ValueError as error:
            return 400, {"error": str(error)}
        return 200, (service.run(goal) if path == "/run" else service.compare(goal))
    return 404, {"error": f"no route for {method} {path}"}


def handle(
    method: str, path: str, body: Mapping[str, Any], service: OrchestrationService
) -> tuple[int, JsonObject]:
    """Route a request, recording per-route metrics, and return status and payload."""

    if method == "GET" and path == "/ready":
        return 200, {"status": "ready"}
    start = time.perf_counter()
    status, payload = _route(method, path, body, service)
    service.metrics.observe(
        f"{method} {path}", (time.perf_counter() - start) * 1000.0, error=status >= 400
    )
    return status, payload


class _Handler(BaseHTTPRequestHandler):  # pragma: no cover - exercised over sockets
    service: OrchestrationService

    def _respond(self, status: int, payload: JsonObject) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/metrics":
            body = self.service.metrics.render_prometheus().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        status, payload = handle("GET", self.path, {}, self.service)
        self._respond(status, payload)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._respond(400, {"error": "request body is not valid JSON"})
            return
        status, payload = handle("POST", self.path, body, self.service)
        self._respond(status, payload)

    def log_message(self, *_: Any) -> None:
        return


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover
    """Start the HTTP service (blocking)."""

    handler = type("_BoundHandler", (_Handler,), {"service": OrchestrationService.build()})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"multi-agent-orchestrator service listening on http://{host}:{port}")
    server.serve_forever()

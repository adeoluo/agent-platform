"""A standard-library HTTP service exposing the ReAct agent.

The service runs the agent with the offline, deterministic rule-policy model and
the full toolset, so it answers questions and returns a complete, auditable
trace without any network dependency. A pure handler is separated from the socket
binding for testing.

Endpoints
---------
``GET  /health``   liveness, tool names, and model name.
``POST /run``      body ``{"question": str, "max_steps"?: int}``; returns the
                   answer, whether the agent finished, and the full trace.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from react_agent.core import ReActAgent
from react_agent.llms import RulePolicyModel
from react_agent.observability import ServiceMetrics
from react_agent.tools import full_toolset
from react_agent.trace import trace_to_dict

JsonObject = dict[str, Any]
_MAX_QUESTION_CHARS = 2000
_MAX_STEPS_LIMIT = 12


@dataclass(frozen=True)
class AgentService:
    """Runs the ReAct agent over requests with a deterministic offline model."""

    tool_names: tuple[str, ...]
    model_name: str
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)

    @classmethod
    def build(cls) -> AgentService:
        toolset = full_toolset()
        return cls(tool_names=tuple(toolset.names), model_name=RulePolicyModel().name)

    def run(self, question: str, *, max_steps: int) -> JsonObject:
        agent = ReActAgent(RulePolicyModel(), full_toolset(), max_steps=max_steps)
        result = agent.run(question)
        return trace_to_dict(result)


def _validate_run(body: Mapping[str, Any]) -> tuple[str, int]:
    question = body.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("'question' must be a non-empty string")
    if len(question) > _MAX_QUESTION_CHARS:
        raise ValueError(f"'question' exceeds {_MAX_QUESTION_CHARS} characters")
    max_steps = body.get("max_steps", 4)
    if not isinstance(max_steps, int) or not 1 <= max_steps <= _MAX_STEPS_LIMIT:
        raise ValueError(f"'max_steps' must be an integer in [1, {_MAX_STEPS_LIMIT}]")
    return question, max_steps


def _route(
    method: str, path: str, body: Mapping[str, Any], service: AgentService
) -> tuple[int, JsonObject]:
    if method == "GET" and path == "/health":
        return 200, {
            "status": "ok",
            "tools": list(service.tool_names),
            "model": service.model_name,
        }
    if method == "POST" and path == "/run":
        try:
            question, max_steps = _validate_run(body)
        except ValueError as error:
            return 400, {"error": str(error)}
        return 200, service.run(question, max_steps=max_steps)
    return 404, {"error": f"no route for {method} {path}"}


def handle(
    method: str, path: str, body: Mapping[str, Any], service: AgentService
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
    service: AgentService

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

    handler = type("_BoundHandler", (_Handler,), {"service": AgentService.build()})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"react-agent service listening on http://{host}:{port}")
    server.serve_forever()

"""A standard-library HTTP service for schema-constrained generation.

The service compiles a default schema at startup. It validates arbitrary JSON
against the schema's automaton, generates a schema-valid document from a seeded
model, and reports the validity rate with and without the constraint. The routing
logic is a pure function tested without a socket.

Endpoints
---------
``GET  /health``    liveness and the active schema pattern.
``GET  /ready``     readiness once the schema is compiled.
``GET  /metrics``   per-route request, error, and latency metrics (Prometheus text).
``POST /validate``  body ``{"text": str}``; whether the text matches the schema.
``POST /generate``  body ``{"seed": int?}``; a schema-valid document from a seeded model.
``GET  /report``    validity rate with and without the constraint over random models.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from constrained_generation_library.constrainer import constrained_generate
from constrained_generation_library.data import default_vocab, random_model
from constrained_generation_library.evaluation import validity_report
from constrained_generation_library.observability import ServiceMetrics
from constrained_generation_library.regex import Regex
from constrained_generation_library.schema import Property, compile_schema, schema_to_pattern

JsonObject = dict[str, Any]
_MAX_TEXT = 4000

_DEFAULT_SCHEMA = [
    Property("name", "string", max_length=10),
    Property("age", "integer", max_length=3),
    Property("active", "boolean"),
    Property("role", "enum", ("admin", "user", "guest")),
]


@dataclass(frozen=True)
class ConstraintService:
    """Holds the compiled schema and answers validate, generate, and report requests."""

    regex: Regex
    pattern: str
    vocab: list[str]
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)

    @classmethod
    def built(cls, properties: list[Property] | None = None) -> ConstraintService:
        props = properties or _DEFAULT_SCHEMA
        return cls(
            regex=compile_schema(props), pattern=schema_to_pattern(props), vocab=default_vocab()
        )

    def validate(self, text: str) -> JsonObject:
        return {"valid": self.regex.fullmatch(text)}

    def generate(self, seed: int) -> JsonObject:
        result = constrained_generate(self.regex, self.vocab, random_model(seed), max_tokens=120)
        return {
            "text": result.text,
            "valid": self.regex.fullmatch(result.text),
            "steps": result.steps,
        }

    def report(self) -> JsonObject:
        report = validity_report(self.regex, trials=50, vocab=self.vocab)
        return {
            "trials": report.trials,
            "constrained_rate": round(report.constrained_rate, 6),
            "unconstrained_rate": round(report.unconstrained_rate, 6),
            "mean_allowed_fraction": round(report.mean_allowed_fraction, 6),
        }


def _route(
    method: str, path: str, body: Mapping[str, Any], service: ConstraintService
) -> tuple[int, JsonObject]:
    if method == "GET" and path == "/health":
        return 200, {"status": "ok", "pattern": service.pattern}
    if method == "GET" and path == "/report":
        return 200, service.report()
    if method == "POST" and path == "/validate":
        text = body.get("text")
        if not isinstance(text, str) or len(text) > _MAX_TEXT:
            return 400, {"error": f"'text' must be a string of at most {_MAX_TEXT} characters"}
        return 200, service.validate(text)
    if method == "POST" and path == "/generate":
        seed = body.get("seed", 0)
        if not isinstance(seed, int) or isinstance(seed, bool) or not (0 <= seed < 10**9):
            return 400, {"error": "'seed' must be a non-negative integer"}
        return 200, service.generate(seed)
    return 404, {"error": f"no route for {method} {path}"}


def handle(
    method: str, path: str, body: Mapping[str, Any], service: ConstraintService
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
    service: ConstraintService

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

    handler = type("_BoundHandler", (_Handler,), {"service": ConstraintService.built()})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"constrained-generation-library service listening on http://{host}:{port}")
    server.serve_forever()

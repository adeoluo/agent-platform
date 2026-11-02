"""The HTTP API: a threaded standard-library server in front of the platform.

Every authenticated request follows the same path: authenticate the API key, resolve the
tenant from the key row, apply the tenant's rate limit, open a trace, run the handler, and
persist the trace under the tenant. Request bodies are bounded, errors are JSON, and
``/metrics`` exposes request, auth, rate-limit, gateway, and job counters.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from llm_observability import Tracer

from agent_platform.app import Platform
from agent_platform.auth import AuthError, Principal
from agent_platform.knowledge import parse_documents
from agent_platform.model import GatewayBlockedError
from agent_platform.orchestration import validate_job
from agent_platform.structured import parse_schema


@dataclass
class Response:
    status: int
    payload: Any
    headers: dict[str, str] = field(default_factory=dict)
    content_type: str = "application/json"


class HTTPError(Exception):
    def __init__(self, status: int, code: str, message: str, **headers: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers


Handler = Callable[[Platform, Principal, dict[str, Any], str, Tracer], Response]


def _require_json(body: dict[str, Any], key: str) -> Any:
    if key not in body:
        raise HTTPError(400, "invalid_request", f"missing field '{key}'")
    return body[key]


def handle_ingest(
    platform: Platform, principal: Principal, body: dict[str, Any], _: str, tracer: Tracer
) -> Response:
    try:
        documents = parse_documents(_require_json(body, "documents"))
    except ValueError as error:
        raise HTTPError(400, "invalid_request", str(error)) from None
    with tracer.span("knowledge.ingest", documents=len(documents)):
        result = platform.knowledge.ingest(principal.tenant, documents)
    platform.metrics.inc(
        "agent_platform_ingested_documents_total",
        "Documents ingested, by tenant.",
        float(result.documents),
        tenant=principal.tenant,
    )
    return Response(200, {"ingested": result.documents, "kb_version": result.kb_version})


def handle_ask(
    platform: Platform, principal: Principal, body: dict[str, Any], _: str, tracer: Tracer
) -> Response:
    question = _require_json(body, "question")
    if not isinstance(question, str):
        raise HTTPError(400, "invalid_request", "'question' must be a string")
    schema = None
    if body.get("response_schema") is not None:
        try:
            schema = parse_schema(body["response_schema"])
        except ValueError as error:
            raise HTTPError(400, "invalid_schema", str(error)) from None
    try:
        result = platform.runtime.ask(principal.tenant, question, tracer=tracer, schema=schema)
    except ValueError as error:
        raise HTTPError(400, "invalid_request", str(error)) from None
    except GatewayBlockedError as error:
        platform.metrics.inc(
            "agent_platform_gateway_blocked_total",
            "Requests refused by the gateway, by reason.",
            tenant=principal.tenant,
            status=error.status,
        )
        reasons = ", ".join(error.reasons)
        raise HTTPError(422, error.status, f"refused by gateway: {reasons}") from None
    platform.metrics.inc(
        "agent_platform_answers_total",
        "Answers produced, by tenant and whether they were grounded.",
        tenant=principal.tenant,
        grounded=str(result.grounded).lower(),
    )
    return Response(200, result.to_dict())


def handle_submit_job(
    platform: Platform, principal: Principal, body: dict[str, Any], _: str, tracer: Tracer
) -> Response:
    try:
        payload = validate_job(
            body.get("kind", "research"),
            body,
            max_step_delay=platform.settings.max_step_delay_seconds,
        )
    except ValueError as error:
        raise HTTPError(400, "invalid_request", str(error)) from None
    with tracer.span("jobs.submit") as span:
        job = platform.jobs.submit(principal.tenant, "research", payload)
        span.set_attribute("job_id", job.job_id)
    platform.metrics.inc(
        "agent_platform_jobs_submitted_total", "Jobs submitted, by tenant.", tenant=job.tenant
    )
    return Response(
        202,
        {"job_id": job.job_id, "status": job.status, "trace_id": job.trace_id},
        headers={"Location": f"/v1/jobs/{job.job_id}"},
    )


def handle_get_job(
    platform: Platform, principal: Principal, _: dict[str, Any], job_id: str, tracer: Tracer
) -> Response:
    job = platform.jobs.get(job_id, tenant=principal.tenant)
    if job is None:
        raise HTTPError(404, "not_found", "job not found")
    steps = platform.jobs.steps(job_id)
    total = len(job.board["subtasks"]) if job.board else None
    data = job.to_dict(steps_completed=len(steps), steps_total=total)
    data["steps"] = [
        {"seq": s.seq, "subtask_id": s.subtask_id, "worker_id": s.worker_id} for s in steps
    ]
    return Response(200, data)


def handle_get_trace(
    platform: Platform, principal: Principal, _: dict[str, Any], trace_id: str, tracer: Tracer
) -> Response:
    trace = platform.traces.get(principal.tenant, trace_id)
    if trace is None:
        raise HTTPError(404, "not_found", "trace not found")
    return Response(200, platform.traces.to_payload(trace))


_ID = r"([A-Za-z0-9_.-]{1,128})"
ROUTES: list[tuple[str, re.Pattern[str], str, Handler]] = [
    ("POST", re.compile(r"^/v1/ingest$"), "/v1/ingest", handle_ingest),
    ("POST", re.compile(r"^/v1/ask$"), "/v1/ask", handle_ask),
    ("POST", re.compile(r"^/v1/jobs$"), "/v1/jobs", handle_submit_job),
    ("GET", re.compile(rf"^/v1/jobs/{_ID}$"), "/v1/jobs/{id}", handle_get_job),
    ("GET", re.compile(rf"^/v1/traces/{_ID}$"), "/v1/traces/{id}", handle_get_trace),
]
PUBLIC = {"/health", "/ready", "/metrics"}


def route_template(path: str) -> str:
    """Return the route label used in metrics (bounded cardinality)."""
    if path in PUBLIC:
        return path
    for _, pattern, template, _ in ROUTES:
        if pattern.match(path):
            return template
    return "unmatched"


class PlatformHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128

    def __init__(self, address: tuple[str, int], platform: Platform) -> None:
        self.platform = platform
        super().__init__(address, _RequestHandler)


class _RequestHandler(BaseHTTPRequestHandler):
    server: PlatformHTTPServer
    protocol_version = "HTTP/1.1"
    server_version = "agent-platform/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        return None

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PUT(self) -> None:
        self._dispatch("PUT")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    # ------------------------------------------------------------------------------- #
    def _dispatch(self, method: str) -> None:
        platform = self.server.platform
        start = time.perf_counter()
        path = self.path.split("?", 1)[0]
        route = route_template(path)
        try:
            response = self._route(platform, method, path)
        except HTTPError as error:
            response = Response(
                error.status,
                {"error": error.code, "message": error.message},
                headers=dict(error.headers),
            )
            if error.status in (401, 413, 429):
                response.headers["Connection"] = "close"
                self.close_connection = True
        except Exception as error:  # pragma: no cover - defensive
            response = Response(500, {"error": "internal", "message": type(error).__name__})
        self._send(response)
        elapsed = time.perf_counter() - start
        platform.metrics.inc(
            "agent_platform_http_requests_total",
            "HTTP requests, by method, route, and status.",
            method=method,
            route=route,
            status=str(response.status),
        )
        platform.metrics.observe(
            "agent_platform_http_request_duration_seconds",
            "HTTP request latency in seconds, by route.",
            elapsed,
            route=route,
        )

    def _route(self, platform: Platform, method: str, path: str) -> Response:
        if path == "/health":
            self._only(method, "GET")
            return Response(200, {"status": "ok"})
        if path == "/ready":
            self._only(method, "GET")
            if platform.db.ping():
                return Response(200, {"status": "ready"})
            return Response(503, {"status": "unavailable"})
        if path == "/metrics":
            self._only(method, "GET")
            return Response(200, render_metrics(platform), content_type="text/plain")
        matched_path = False
        for route_method, pattern, template, handler in ROUTES:
            match = pattern.match(path)
            if match is None:
                continue
            matched_path = True
            if route_method != method:
                continue
            return self._authenticated(platform, method, template, handler, match)
        if matched_path:
            raise HTTPError(405, "method_not_allowed", f"{method} is not allowed here")
        raise HTTPError(404, "not_found", "no such route")

    @staticmethod
    def _only(method: str, allowed: str) -> None:
        if method != allowed:
            raise HTTPError(405, "method_not_allowed", f"{method} is not allowed here")

    def _presented_key(self) -> str | None:
        header = self.headers.get("Authorization", "")
        if header.lower().startswith("bearer "):
            return header[7:].strip()
        return self.headers.get("X-API-Key")

    def _authenticated(
        self,
        platform: Platform,
        method: str,
        template: str,
        handler: Handler,
        match: re.Match[str],
    ) -> Response:
        try:
            principal = platform.keys.verify(self._presented_key())
        except AuthError as error:
            platform.metrics.inc(
                "agent_platform_auth_failures_total",
                "Rejected API keys, by reason.",
                reason=error.reason,
            )
            raise HTTPError(
                401, "unauthorized", error.reason, **{"WWW-Authenticate": "Bearer"}
            ) from None
        if not platform.allow_request(principal.tenant):
            platform.metrics.inc(
                "agent_platform_rate_limited_total",
                "Requests rejected by the per-tenant rate limit.",
                tenant=principal.tenant,
            )
            raise HTTPError(
                429, "rate_limited", "tenant rate limit exceeded", **{"Retry-After": "1"}
            )
        body = self._read_json(platform) if method == "POST" else {}
        tracer = Tracer()
        param = match.group(1) if match.groups() else ""
        trace_id = ""
        try:
            with tracer.span(
                "http.request", method=method, route=template, tenant=principal.tenant
            ) as root:
                trace_id = root.trace_id
                try:
                    response = handler(platform, principal, body, param, tracer)
                except HTTPError as error:
                    root.set_attribute("status", error.status)
                    error.headers["X-Trace-Id"] = trace_id
                    raise
                root.set_attribute("status", response.status)
        finally:
            platform.traces.save_tracer(principal.tenant, tracer)
        response.headers["X-Trace-Id"] = trace_id
        if isinstance(response.payload, dict) and template == "/v1/ask":
            response.payload["trace_id"] = trace_id
        return response

    def _read_json(self, platform: Platform) -> dict[str, Any]:
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            raise HTTPError(411, "length_required", "Content-Length is required")
        try:
            length = int(length_header)
        except ValueError:
            raise HTTPError(400, "invalid_request", "bad Content-Length") from None
        if length < 0 or length > platform.settings.max_body_bytes:
            raise HTTPError(413, "payload_too_large", "request body too large")
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise HTTPError(400, "invalid_json", "request body is not valid JSON") from None
        if not isinstance(body, dict):
            raise HTTPError(400, "invalid_json", "request body must be a JSON object")
        return body

    def _send(self, response: Response) -> None:
        if response.content_type == "application/json":
            data = json.dumps(response.payload).encode("utf-8")
        else:
            data = str(response.payload).encode("utf-8")
        self.send_response(response.status)
        self.send_header("Content-Type", f"{response.content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        for key, value in response.headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)


def render_metrics(platform: Platform) -> str:
    """Refresh scrape-time gauges and render all metrics."""
    for status, count in platform.jobs.counts().items():
        platform.metrics.set_gauge(
            "agent_platform_jobs", "Jobs currently in each status.", float(count), status=status
        )
    for tenant, stats in platform.gateways.stats().items():
        for name, value in (
            ("requests", stats.requests),
            ("model_calls", stats.model_calls),
            ("cache_hits", stats.cache_hits),
            ("blocked_injection", stats.blocked_injection),
            ("blocked_policy", stats.blocked_policy),
            ("pii_redacted", stats.pii_redacted),
        ):
            platform.metrics.set_counter(
                f"agent_platform_gateway_{name}_total",
                f"Gateway {name.replace('_', ' ')} in this server process, by tenant.",
                float(value),
                tenant=tenant,
            )
    return platform.metrics.render()


def make_server(platform: Platform, host: str, port: int) -> PlatformHTTPServer:
    """Bind a server for ``platform``; port 0 picks a free port."""
    return PlatformHTTPServer((host, port), platform)


class ServerThread:
    """Runs the API server on a background thread (tests, demo, eval, benchmarks)."""

    def __init__(self, platform: Platform, *, host: str = "127.0.0.1", port: int = 0) -> None:
        self.server = make_server(platform, host, port)
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host!s}:{port}"

    def __enter__(self) -> ServerThread:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self._thread.join(timeout=5)

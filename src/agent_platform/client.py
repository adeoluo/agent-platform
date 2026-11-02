"""A minimal standard-library HTTP client for the platform API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


class APIError(Exception):
    """Raised for non-2xx responses."""

    def __init__(self, status: int, payload: Any) -> None:
        super().__init__(f"HTTP {status}: {payload}")
        self.status = status
        self.payload = payload


@dataclass
class Client:
    """Calls the platform API with an API key."""

    base_url: str
    api_key: str | None = None
    timeout: float = 30.0

    def request(self, method: str, path: str, body: Any | None = None) -> tuple[int, Any]:
        """Send a request and return ``(status, parsed body)``; never raises on HTTP errors."""
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = self._headers()
        req = urllib.request.Request(
            self.base_url.rstrip("/") + path, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return response.status, _decode(response.read(), response.headers)
        except urllib.error.HTTPError as error:
            return error.code, _decode(error.read(), error.headers)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _ok(self, method: str, path: str, body: Any | None = None) -> Any:
        status, payload = self.request(method, path, body)
        if not 200 <= status < 300:
            raise APIError(status, payload)
        return payload

    def ingest(self, documents: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = self._ok("POST", "/v1/ingest", {"documents": documents})
        return result

    def ask(
        self,
        question: str,
        *,
        response_schema: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> Any:
        body: dict[str, Any] = {"question": question}
        if session_id is not None:
            body["session_id"] = session_id
        if response_schema is not None:
            body["response_schema"] = response_schema
        return self._ok("POST", "/v1/ask", body)

    def ask_stream(self, question: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield ``(event, data)`` pairs from ``POST /v1/ask/stream`` as they arrive."""
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/v1/ask/stream",
            data=json.dumps({"question": question}).encode("utf-8"),
            headers={**self._headers(), "Accept": "text/event-stream"},
            method="POST",
        )
        try:
            response = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            raise APIError(error.code, _decode(error.read(), error.headers)) from None
        with response:
            yield from parse_sse(response)

    def remember(self, text: str, *, session_id: str | None = None) -> Any:
        body: dict[str, Any] = {"text": text}
        if session_id is not None:
            body["session_id"] = session_id
        return self._ok("POST", "/v1/memory", body)

    def recall(self, query: str, *, k: int = 5) -> Any:
        return self._ok("POST", "/v1/memory/recall", {"query": query, "k": k})

    def session(self, session_id: str) -> Any:
        return self._ok("GET", f"/v1/sessions/{session_id}")

    def submit_job(self, questions: list[str], *, step_delay_seconds: float = 0.0) -> Any:
        return self._ok(
            "POST",
            "/v1/jobs",
            {"kind": "research", "questions": questions, "step_delay_seconds": step_delay_seconds},
        )

    def job(self, job_id: str) -> Any:
        return self._ok("GET", f"/v1/jobs/{job_id}")

    def trace(self, trace_id: str) -> Any:
        return self._ok("GET", f"/v1/traces/{trace_id}")

    def metrics(self) -> str:
        return str(self._ok("GET", "/metrics"))


def _decode(raw: bytes, headers: Any) -> Any:
    text = raw.decode("utf-8")
    content_type = headers.get("Content-Type", "") if headers is not None else ""
    if "application/json" in content_type:
        return json.loads(text) if text else None
    return text


def parse_sse(lines: Any) -> Iterator[tuple[str, dict[str, Any]]]:
    """Parse a server-sent event stream (an iterable of byte lines) into ``(event, data)``."""
    event, data = "message", list[str]()
    for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield event, json.loads("\n".join(data))
            event, data = "message", list[str]()
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())

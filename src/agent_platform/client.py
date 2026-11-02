"""A minimal standard-library HTTP client for the platform API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
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
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.base_url.rstrip("/") + path, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return response.status, _decode(response.read(), response.headers)
        except urllib.error.HTTPError as error:
            return error.code, _decode(error.read(), error.headers)

    def _ok(self, method: str, path: str, body: Any | None = None) -> Any:
        status, payload = self.request(method, path, body)
        if not 200 <= status < 300:
            raise APIError(status, payload)
        return payload

    def ingest(self, documents: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = self._ok("POST", "/v1/ingest", {"documents": documents})
        return result

    def ask(self, question: str, *, response_schema: dict[str, Any] | None = None) -> Any:
        body: dict[str, Any] = {"question": question}
        if response_schema is not None:
            body["response_schema"] = response_schema
        return self._ok("POST", "/v1/ask", body)

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

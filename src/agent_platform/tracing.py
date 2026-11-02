"""Trace persistence on top of ``llm_observability``.

Spans are produced in-process by an ``llm_observability.Tracer`` (one per request or per job
step) and persisted to SQLite as JSON using the library's own exporter format, tagged with
the tenant that owns them. A trace can therefore be assembled from spans written by several
processes: a job's trace contains spans from every worker that touched it, including a
worker that was killed and the one that finished the job.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Any

from llm_observability import Span, Trace, Tracer
from llm_observability.exporters import render_tree, trace_from_dict, trace_to_dict

from agent_platform.db import Database


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


class TraceStore:
    """Writes and reads tenant-scoped spans."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def save(self, tenant: str, spans: list[Span], *, trace_id: str | None = None) -> None:
        """Persist ``spans`` for ``tenant``, optionally re-homing them under ``trace_id``."""
        if not spans:
            return
        # Tracer timestamps come from perf_counter, which is per-process. Shift them onto
        # the wall clock so spans written by different workers line up in one trace.
        offset = time.time() - time.perf_counter()
        rehomed = [
            replace(
                span,
                trace_id=trace_id or span.trace_id,
                start_time=span.start_time + offset,
                end_time=None if span.end_time is None else span.end_time + offset,
            )
            for span in spans
        ]
        for span in rehomed:
            span.attributes = {key: _json_safe(value) for key, value in span.attributes.items()}
        payload = trace_to_dict(Trace(rehomed[0].trace_id, rehomed))
        now = time.time()
        with self._db.transaction() as conn:
            for item in payload["spans"]:
                conn.execute(
                    "INSERT OR REPLACE INTO spans (trace_id, span_id, tenant, payload, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (item["trace_id"], item["span_id"], tenant, json.dumps(item), now),
                )

    def save_tracer(self, tenant: str, tracer: Tracer, *, trace_id: str | None = None) -> None:
        """Persist every finished span collected by ``tracer``."""
        self.save(tenant, tracer.finished_spans, trace_id=trace_id)

    def get(self, tenant: str, trace_id: str) -> Trace | None:
        """Return the trace if it exists and belongs to ``tenant``."""
        rows = (
            self._db.conn()
            .execute(
                "SELECT payload FROM spans WHERE tenant = ? AND trace_id = ? "
                "ORDER BY created_at, rowid",
                (tenant, trace_id),
            )
            .fetchall()
        )
        if not rows:
            return None
        return trace_from_dict(
            {"trace_id": trace_id, "spans": [json.loads(row["payload"]) for row in rows]}
        )

    @staticmethod
    def to_payload(trace: Trace) -> dict[str, Any]:
        """Render a trace as JSON-ready data with a readable tree."""
        data = trace_to_dict(trace)
        data["span_count"] = len(trace)
        data["has_error"] = trace.has_error
        data["tree"] = render_tree(trace)
        return data

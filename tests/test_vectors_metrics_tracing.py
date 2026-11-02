from __future__ import annotations

import math

import pytest
from hybrid_rag import Chunk
from llm_observability import Tracer

from agent_platform.app import Platform
from agent_platform.config import Settings
from agent_platform.evaluation import percentile
from agent_platform.metrics import Metrics
from agent_platform.vectors import HashingEmbedder, HnswRetriever


def test_embeddings_are_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(dim=64)
    first = embedder.embed("refunds within thirty days")
    assert first == HashingEmbedder(dim=64).embed("refunds within thirty days")
    assert math.isclose(math.sqrt(sum(x * x for x in first)), 1.0)
    assert not any(embedder.embed("the of and"))
    with pytest.raises(ValueError):
        HashingEmbedder(dim=4)


def test_hnsw_retriever_ranks_morphological_variants() -> None:
    chunks = [
        Chunk(id="a#0", document_id="a", text="Refunds are processed within 30 days."),
        Chunk(id="b#0", document_id="b", text="The sensor measures humidity."),
        Chunk(id="c#0", document_id="c", text="Shipping is free over 75 dollars."),
    ]
    retriever = HnswRetriever()
    assert retriever.retrieve("refund") == []
    retriever.index(chunks)
    assert retriever.retrieve("refund policy", k=1)[0].chunk.document_id == "a"
    assert retriever.retrieve("the of", k=3) == []


def test_metrics_render_prometheus_text() -> None:
    metrics = Metrics()
    metrics.inc("requests_total", "Requests.", route="/x")
    metrics.inc("requests_total", "Requests.", 2.0, route="/x")
    metrics.set_gauge("queue", "Queue depth.", 1.5)
    metrics.set_counter("mirrored_total", "Mirrored.", 7.0, tenant='a"b')
    metrics.observe("latency_seconds", "Latency.", 0.003, route="/x")
    metrics.observe("latency_seconds", "Latency.", 9.0, route="/x")
    text = metrics.render()
    assert 'requests_total{route="/x"} 3' in text
    assert "queue 1.5" in text
    assert 'mirrored_total{tenant="a\\"b"} 7' in text
    assert 'latency_seconds_bucket{route="/x",le="0.005"} 1' in text
    assert 'latency_seconds_bucket{route="/x",le="+Inf"} 2' in text
    assert 'latency_seconds_count{route="/x"} 2' in text
    assert "# TYPE latency_seconds histogram" in text
    assert metrics.counter_value("requests_total", route="/x") == 3.0


def test_trace_store_is_tenant_scoped_and_uses_wall_clock(platform: Platform) -> None:
    tracer = Tracer()
    with tracer.span("outer", payload=object()), tracer.span("inner"):
        pass
    platform.traces.save_tracer("acme", tracer)
    trace_id = tracer.finished_spans[0].trace_id
    trace = platform.traces.get("acme", trace_id)
    assert trace is not None and len(trace) == 2
    assert platform.traces.get("globex", trace_id) is None
    outer = trace.spans_named("outer")[0]
    assert outer.start_time > 1_600_000_000  # epoch seconds, comparable across processes
    assert isinstance(outer.attributes["payload"], str)
    payload = platform.traces.to_payload(trace)
    assert payload["span_count"] == 2 and "inner" in payload["tree"]
    platform.traces.save("acme", [])


def test_percentile() -> None:
    assert percentile([], 50) == 0.0
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 50) == 50.0
    assert percentile(values, 99) == 99.0
    assert percentile([5.0], 95) == 5.0


def test_settings_from_env() -> None:
    settings = Settings.from_env(
        {
            "AGENT_PLATFORM_DB": "x.db",
            "AGENT_PLATFORM_PORT": "9000",
            "AGENT_PLATFORM_RATE_CAPACITY": "5",
            "AGENT_PLATFORM_LEASE_SECONDS": "2.5",
            "AGENT_PLATFORM_MAX_ATTEMPTS": "3",
        }
    )
    assert (settings.port, settings.rate_capacity, settings.lease_seconds) == (9000, 5.0, 2.5)
    assert settings.max_attempts == 3 and str(settings.db_path) == "x.db"
    assert Settings.from_env({}).port == 8080


def test_database_ping_and_close(platform: Platform) -> None:
    assert platform.db.ping()
    platform.db.close()
    assert platform.db.ping()  # reopens a fresh connection lazily

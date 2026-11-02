"""Per-tenant conversation memory through the API."""

from __future__ import annotations

from agent_platform.app import memory_path
from agent_platform.corpus import ACME_DOCUMENTS
from agent_platform.memory import validate_session_id
from tests.conftest import Running

FACT = "The quarterly planning meeting is held in the Juniper room on the third floor."
QUESTION = "Where is the quarterly planning meeting held?"


def test_answer_from_memory_is_cited_and_grounded(running: Running) -> None:
    client = running.client("acme")
    client.ingest(ACME_DOCUMENTS)
    stored = client.remember(FACT)
    assert stored["episode_id"] > 0

    without = client.ask(QUESTION)
    assert without["grounded"] is False  # documents alone do not know this

    answer = client.ask(QUESTION, session_id="s-1")
    assert "Juniper" in answer["answer"]
    assert answer["citations"][0].startswith("memory:")
    assert answer["grounded"] is True
    assert answer["session_id"] == "s-1"
    memory_sources = [s for s in answer["sources"] if "memory" in s]
    assert memory_sources and memory_sources[0]["memory"]["explanation"]


def test_session_records_question_and_answer(running: Running) -> None:
    client = running.client("acme")
    client.ingest(ACME_DOCUMENTS)
    client.ask("How many days do customers have to request a refund?", session_id="s-2")
    turns = client.session("s-2")["turns"]
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert "30 days" in turns[1]["text"]
    status, body = client.request("GET", "/v1/sessions/nope")
    assert (status, body["error"]) == (404, "not_found")


def test_recall_explains_results(running: Running) -> None:
    client = running.client("acme")
    client.remember(FACT)
    client.remember("Invoices are approved by the finance lead.")
    hits = client.recall("planning meeting room", k=2)["memories"]
    assert hits[0]["text"] == FACT
    assert hits[0]["doc_id"] == f"memory:{hits[0]['id']}"
    assert hits[0]["explanation"]


def test_memory_is_isolated_between_tenants(running: Running) -> None:
    acme, globex = running.client("acme"), running.client("globex")
    acme.remember(FACT)
    assert globex.recall("quarterly planning meeting")["memories"] == []
    globex.ingest(ACME_DOCUMENTS)
    answer = globex.ask(QUESTION, session_id="s-1")
    assert not any(c.startswith("memory:") for c in answer["citations"])
    status, _ = globex.request("GET", "/v1/sessions/s-1")
    assert status == 200  # globex's own session, with only globex's turns
    assert all("Juniper" not in t["text"] for t in globex.session("s-1")["turns"])


def test_validation(running: Running) -> None:
    client = running.client("acme")
    for body, path in [
        ({"text": ""}, "/v1/memory"),
        ({"text": "x" * 4001}, "/v1/memory"),
        ({"query": " "}, "/v1/memory/recall"),
        ({"query": "x", "k": 0}, "/v1/memory/recall"),
        ({"question": "hi", "session_id": "bad id!"}, "/v1/ask"),
    ]:
        status, payload = client.request("POST", path, body)
        assert (status, payload["error"]) == (400, "invalid_request"), (path, body)
    assert validate_session_id("user:42@web") == "user:42@web"


def test_memory_file_sits_next_to_the_database(running: Running) -> None:
    path = memory_path(running.platform.settings.db_path)
    assert path.name == "platform.memory.db"
    assert path.exists()

"""Tenants can never see each other's documents, answers, jobs, or traces."""

from __future__ import annotations

from agent_platform.app import Platform
from agent_platform.corpus import ACME_DOCUMENTS, EVAL_CASES
from agent_platform.knowledge import DocumentInput
from tests.conftest import Running


def test_retrieval_never_crosses_tenants(platform: Platform) -> None:
    kb = platform.knowledge
    kb.ingest("acme", [DocumentInput(d["id"], d["text"], {}) for d in ACME_DOCUMENTS])
    kb.ingest("globex", [DocumentInput("handbook", "Globex vacation policy is 25 days.", {})])
    for case in EVAL_CASES:
        for passage in kb.search("globex", case.question, k=10):
            assert passage.doc_id == "handbook"
    assert kb.search("initech", "refund") == []


def test_same_document_id_in_two_tenants_stays_separate(platform: Platform) -> None:
    kb = platform.knowledge
    kb.ingest("acme", [DocumentInput("policy", "Acme refunds take 30 days.", {})])
    kb.ingest("globex", [DocumentInput("policy", "Globex refunds take 90 days.", {})])
    [acme] = kb.search("acme", "refunds", k=5)
    [globex] = kb.search("globex", "refunds", k=5)
    assert "30 days" in acme.text and "90 days" in globex.text
    assert kb.document_count("acme") == kb.document_count("globex") == 1


def test_api_isolation_for_answers_jobs_and_traces(running: Running) -> None:
    acme = running.client("acme")
    globex = running.client("globex")
    acme.ingest(ACME_DOCUMENTS)
    globex.ingest([{"id": "globex-handbook", "text": "Globex staff get 25 vacation days."}])

    question = "How many days do customers have to request a refund?"
    acme_answer = acme.ask(question)
    globex_answer = globex.ask(question)
    assert acme_answer["citations"] == ["refund-policy"]
    assert "could not find" in globex_answer["answer"]
    assert all(s["doc_id"] == "globex-handbook" for s in globex_answer["sources"])

    status, _ = globex.request("GET", f"/v1/traces/{acme_answer['trace_id']}")
    assert status == 404
    job = acme.submit_job([question])
    status, _ = globex.request("GET", f"/v1/jobs/{job['job_id']}")
    assert status == 404
    assert acme.job(job["job_id"])["status"] == "queued"


def test_reingest_updates_the_index(platform: Platform) -> None:
    kb = platform.knowledge
    kb.ingest("acme", [DocumentInput("policy", "Refunds take 30 days.", {})])
    assert "30 days" in kb.search("acme", "refunds")[0].text
    result = kb.ingest("acme", [DocumentInput("policy", "Refunds take 45 days.", {})])
    assert result.kb_version == 2
    assert "45 days" in kb.search("acme", "refunds")[0].text


def test_document_validation() -> None:
    import pytest

    from agent_platform.knowledge import MAX_DOCUMENT_CHARS, parse_documents

    good = parse_documents([{"id": " a ", "text": "x", "metadata": {"k": 1}}])
    assert good[0].doc_id == "a" and good[0].metadata == {"k": 1}
    bad_inputs: list[object] = [
        "nope",
        [],
        ["x"],
        [{"id": "", "text": "x"}],
        [{"id": "a[1]", "text": "x"}],
        [{"id": "a", "text": " "}],
        [{"id": "a", "text": "x" * (MAX_DOCUMENT_CHARS + 1)}],
        [{"id": "a", "text": "x", "metadata": []}],
        [{"id": str(i), "text": "x"} for i in range(501)],
    ]
    for raw in bad_inputs:
        with pytest.raises(ValueError):
            parse_documents(raw)

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_platform.cli import main
from agent_platform.client import APIError, Client
from agent_platform.corpus import EVAL_CASES
from agent_platform.demo import run_demo
from agent_platform.evaluation import run_eval
from tests.conftest import Running


def test_eval_suite_against_the_running_api(running: Running) -> None:
    report = run_eval(running.client("eval"), workers=2)
    assert report.examples == len(EVAL_CASES)
    assert report.accuracy >= 0.9
    assert report.accuracy_by_kind["arithmetic"] == 1.0
    assert report.accuracy_by_kind["unanswerable"] == 1.0
    assert report.citation_rate >= 0.9
    assert report.citation_precision >= 0.9
    assert report.latency_p50_ms <= report.latency_p95_ms <= report.latency_p99_ms
    text = report.format()
    assert "accuracy" in text and "citation precision" in text
    assert json.loads(json.dumps(report.to_dict()))["examples"] == len(EVAL_CASES)


def test_client_raises_api_error(running: Running) -> None:
    with pytest.raises(APIError) as caught:
        Client(running.url, "ap_x_y").ask("hi")
    assert caught.value.status == 401


def test_cli_keys_lifecycle(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = str(workdir / "cli.db")
    assert main(["keys", "create", "--db", db, "--tenant", "acme", "--label", "ops"]) == 0
    out = capsys.readouterr().out
    key_id = next(line.split()[-1] for line in out.splitlines() if line.startswith("key id"))
    assert "api key: ap_" in out
    assert main(["keys", "list", "--db", db]) == 0
    assert "active" in capsys.readouterr().out
    assert main(["keys", "revoke", "--db", db, key_id]) == 0
    assert main(["keys", "revoke", "--db", db, key_id]) == 1
    assert main(["keys", "list", "--db", db, "--tenant", "acme"]) == 0
    assert "revoked" in capsys.readouterr().out
    assert main(["keys", "create", "--db", db, "--tenant", "Bad Name"]) == 2


def test_cli_remote_commands(
    running: Running, workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    key = running.platform.keys.create("acme").secret
    remote = ["--url", running.url, "--api-key", key]
    docs = workdir / "docs"
    docs.mkdir()
    (docs / "warranty.md").write_text("Every Lumen sensor includes a two year warranty.")
    (docs / "extra.json").write_text(
        json.dumps([{"id": "battery", "text": "The Lumen battery lasts 18 months."}])
    )
    (docs / "ignored.bin").write_bytes(b"\x00")
    assert main(["ingest", *remote, str(docs)]) == 0
    assert json.loads(capsys.readouterr().out)["ingested"] == 2

    assert main(["ask", *remote, "How long is the Lumen warranty?"]) == 0
    assert json.loads(capsys.readouterr().out)["citations"] == ["warranty"]
    schema = json.dumps({"properties": [{"name": "answer", "type": "string"}]})
    assert main(["ask", *remote, "How long does the Lumen battery last?", "--schema", schema]) == 0
    assert "18 months" in json.loads(capsys.readouterr().out)["structured"]["answer"]

    assert main(["jobs", "submit", *remote, "-q", "How long is the Lumen warranty?"]) == 0
    job_id = json.loads(capsys.readouterr().out)["job_id"]
    db = str(running.platform.settings.db_path)
    assert main(["worker", "--db", db, "--exit-when-idle", "--lease-seconds", "5"]) == 0
    capsys.readouterr()
    assert main(["jobs", "get", *remote, job_id]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "succeeded"

    empty = workdir / "empty"
    empty.mkdir()
    assert main(["ingest", *remote, str(empty)]) == 2


def test_cli_requires_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_PLATFORM_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        main(["ask", "--url", "http://127.0.0.1:9", "hello"])


def test_cli_eval_in_process(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["eval", "--workdir", str(workdir / "eval"), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["examples"] == len(EVAL_CASES)
    assert report["accuracy"] >= 0.9


def test_cli_eval_against_a_url(running: Running, capsys: pytest.CaptureFixture[str]) -> None:
    key = running.platform.keys.create("eval").secret
    assert main(["eval", "--url", running.url, "--api-key", key]) == 0
    assert "latency p50/p95/p99" in capsys.readouterr().out


def test_demo_runs_end_to_end(workdir: Path) -> None:
    lines: list[str] = []
    summary = run_demo(workdir / "demo", out=lines.append, lease_seconds=1.0)
    assert summary["job_status"] == "succeeded"
    assert all(s["doc_id"] == "globex-handbook" for s in summary["isolation_sources"])
    for subtask_id in summary["steps_before_kill"]:
        assert summary["executions"][subtask_id] == 1
    assert any("SIGKILL" in line for line in lines)

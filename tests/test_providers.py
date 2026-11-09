"""Hosted providers against a local fake of the OpenAI and Anthropic HTTP APIs."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from llm_gateway import ProviderError

from agent_platform.api import ServerThread
from agent_platform.app import Platform
from agent_platform.client import Client
from agent_platform.corpus import ACME_DOCUMENTS
from agent_platform.model import GroundedPolicyProvider
from agent_platform.providers import (
    AnthropicProvider,
    NamedResilientProvider,
    OpenAIProvider,
    ProviderRequestError,
    ProviderSettings,
    build_provider,
    parse_chain,
)
from tests.conftest import make_settings


@dataclass
class FakeLLM:
    """Scripted responses per path; records every request it receives."""

    script: dict[str, list[tuple[int, Any]]] = field(default_factory=dict)
    requests: list[tuple[str, dict[str, str], dict[str, Any]]] = field(default_factory=list)
    url: str = ""

    def reply(self, path: str, status: int, body: Any) -> None:
        self.script.setdefault(path, []).append((status, body))


@pytest.fixture
def fake() -> Iterator[FakeLLM]:
    state = FakeLLM()
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return None

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                state.requests.append(
                    (self.path, {k.lower(): v for k, v in self.headers.items()}, body)
                )
                queue = state.script.get(self.path, [])
                status, payload = queue.pop(0) if queue else (500, {"error": "no script"})
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    yield state
    server.shutdown()
    server.server_close()


def openai_reply(text: str) -> dict[str, Any]:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 10},
    }


def anthropic_reply(text: str) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 40, "output_tokens": 8},
    }


def test_openai_request_shape_and_usage(fake: FakeLLM) -> None:
    fake.reply("/v1/chat/completions", 200, openai_reply("hello"))
    provider = OpenAIProvider(model="m1", api_key="sk-test", base_url=fake.url)
    assert provider.complete("be brief", "hi") == "hello"
    path, headers, body = fake.requests[0]
    assert path == "/v1/chat/completions"
    assert headers["authorization"] == "Bearer sk-test"
    assert body["model"] == "m1"
    assert body["temperature"] == 0
    assert body["reasoning_effort"] == "none"
    assert body["max_completion_tokens"] == 512
    assert "max_tokens" not in body
    assert body["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    assert (provider.usage.calls, provider.usage.input_tokens) == (1, 50)


def test_anthropic_request_shape_and_text_blocks(fake: FakeLLM) -> None:
    fake.reply(
        "/v1/messages",
        200,
        {
            "content": [
                {"type": "text", "text": "a"},
                {"type": "tool_use"},
                {"type": "text", "text": "b"},
            ]
        },
    )
    provider = AnthropicProvider(model="c1", api_key="ak-test", base_url=fake.url)
    assert provider.complete("sys", "hi") == "ab"
    _, headers, body = fake.requests[0]
    assert headers["x-api-key"] == "ak-test"
    assert headers["anthropic-version"] == "2026-10-21"
    assert body["max_tokens"] == 2048
    assert not {"temperature", "top_p", "top_k"} & body.keys()
    assert body["system"] == "sys"
    assert body["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("status", [429, 500, 503])
def test_transient_statuses_are_retryable(fake: FakeLLM, status: int) -> None:
    fake.reply("/v1/chat/completions", status, {"error": {"message": "busy"}})
    provider = OpenAIProvider(api_key="k", base_url=fake.url)
    with pytest.raises(ProviderError, match=f"HTTP {status}: busy"):
        provider.complete("s", "p")


@pytest.mark.parametrize("status", [400, 401, 404])
def test_request_errors_are_not_retryable(fake: FakeLLM, status: int) -> None:
    fake.reply("/v1/messages", status, {"error": {"message": "bad key"}})
    provider = AnthropicProvider(api_key="k", base_url=fake.url)
    with pytest.raises(ProviderRequestError) as info:
        provider.complete("s", "p")
    assert info.value.status == status
    assert not isinstance(info.value, ProviderError)


def test_unreachable_host_is_retryable() -> None:
    provider = OpenAIProvider(api_key="k", base_url="http://127.0.0.1:9", timeout=2)
    with pytest.raises(ProviderError):
        provider.complete("s", "p")


def test_retry_then_success_and_no_retry_on_rejection(fake: FakeLLM) -> None:
    fake.reply("/v1/chat/completions", 429, {"error": {"message": "slow down"}})
    fake.reply("/v1/chat/completions", 200, openai_reply("ok"))
    chain = NamedResilientProvider([OpenAIProvider(api_key="k", base_url=fake.url)])
    assert chain.complete("s", "p") == "ok"
    assert (chain.stats.attempts, chain.stats.retries) == (2, 1)

    fake.reply("/v1/chat/completions", 401, {"error": {"message": "bad key"}})
    with pytest.raises(ProviderRequestError):
        chain.complete("s", "p")
    assert chain.stats.attempts == 3  # one call, no retries


def test_failover_from_anthropic_to_openai(fake: FakeLLM) -> None:
    for _ in range(2):
        fake.reply("/v1/messages", 529, {"error": {"message": "overloaded"}})
    fake.reply("/v1/chat/completions", 200, openai_reply("from openai"))
    chain = NamedResilientProvider(
        [
            AnthropicProvider(api_key="a", base_url=fake.url),
            OpenAIProvider(api_key="o", base_url=fake.url),
        ],
        max_attempts=2,
    )
    assert chain.complete("s", "p") == "from openai"
    assert chain.stats.failovers == 1
    assert chain.name == "anthropic:claude-haiku-5-5 > openai:gpt-6-luna"


def test_parse_chain_and_build_provider() -> None:
    assert parse_chain(" Anthropic, openai ,anthropic") == ("anthropic", "openai")
    with pytest.raises(ValueError, match="unknown provider"):
        parse_chain("gemini")
    with pytest.raises(ValueError, match="empty"):
        parse_chain(" , ")
    assert isinstance(build_provider(environ={}), GroundedPolicyProvider)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_provider(environ={"AGENT_PLATFORM_PROVIDER": "openai"})
    chain = build_provider(
        environ={
            "AGENT_PLATFORM_PROVIDER": "openai,offline",
            "OPENAI_API_KEY": "k",
            "AGENT_PLATFORM_OPENAI_MODEL": "m2",
        }
    )
    assert isinstance(chain, NamedResilientProvider)
    assert chain.name == "openai:m2 > grounded-policy-v1"
    settings = ProviderSettings.from_env({"AGENT_PLATFORM_PROVIDER_ATTEMPTS": "5"})
    assert settings.max_attempts == 5


def test_platform_answers_through_a_hosted_model(fake: FakeLLM, workdir: Path) -> None:
    """A scripted hosted model drives the real ReAct loop, tools, gateway, and citation check."""
    fake.reply(
        "/v1/chat/completions",
        200,
        openai_reply(
            "Thought: I should search.\nAction: search_knowledge_base\n"
            "Action Input: refund window days"
        ),
    )
    fake.reply(
        "/v1/chat/completions",
        200,
        openai_reply("Thought: Found it.\nFinal Answer: Refunds within 30 days. [refund-policy]"),
    )
    provider = OpenAIProvider(model="m1", api_key="k", base_url=fake.url)
    platform = Platform(make_settings(workdir), provider=provider)
    try:
        with ServerThread(platform) as server:
            client = Client(server.url, platform.keys.create("acme").secret)
            client.ingest(ACME_DOCUMENTS)
            answer = client.ask("How many days do customers have to request a refund?")
            trace = client.trace(answer["trace_id"])
    finally:
        platform.close()
    assert answer["answer"] == "Refunds within 30 days. [refund-policy]"
    assert answer["grounded"] is True
    assert answer["model_calls"] == 2
    second_prompt = fake.requests[1][2]["messages"][1]["content"]
    assert "Observation: [refund-policy]" in second_prompt
    assert any(span["name"] == "tool.search_knowledge_base" for span in trace["spans"])


def test_hosted_citation_of_unretrieved_document_is_not_grounded(
    fake: FakeLLM, workdir: Path
) -> None:
    fake.reply(
        "/v1/chat/completions",
        200,
        openai_reply("Thought: I know this.\nFinal Answer: 30 days. [made-up-doc]"),
    )
    platform = Platform(
        make_settings(workdir), provider=OpenAIProvider(api_key="k", base_url=fake.url)
    )
    try:
        with ServerThread(platform) as server:
            client = Client(server.url, platform.keys.create("acme").secret)
            client.ingest(ACME_DOCUMENTS)
            answer = client.ask("How many days for a refund?")
    finally:
        platform.close()
    assert answer["citations"] == ["made-up-doc"]
    assert answer["grounded"] is False


def test_provider_failures_map_to_http_errors(fake: FakeLLM, workdir: Path) -> None:
    fake.reply("/v1/chat/completions", 401, {"error": {"message": "invalid api key"}})
    fake.reply("/v1/chat/completions", 503, {"error": {"message": "down"}})
    platform = Platform(
        make_settings(workdir),
        provider=NamedResilientProvider(
            [OpenAIProvider(api_key="k", base_url=fake.url)], max_attempts=1
        ),
    )
    try:
        with ServerThread(platform) as server:
            client = Client(server.url, platform.keys.create("acme").secret)
            status, body = client.request("POST", "/v1/ask", {"question": "refund window?"})
            assert (status, body["error"]) == (502, "provider_rejected")
            status, body = client.request("POST", "/v1/ask", {"question": "refund window again?"})
            assert (status, body["error"]) == (503, "provider_unavailable")
            metrics = client.metrics()
    finally:
        platform.close()
    assert 'agent_platform_provider_errors_total{kind="rejected"} 1' in metrics
    assert 'agent_platform_provider_errors_total{kind="unavailable"} 1' in metrics

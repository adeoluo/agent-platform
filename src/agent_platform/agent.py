"""The agent runtime: one place that answers a tenant's question.

``AgentRuntime.ask`` is used by both the API (``POST /v1/ask``) and job workers, so a
question gets identical treatment on either path:

1. A ``react_agent.ReActAgent`` runs with a tenant-bound tool set: a knowledge-base search
   tool backed by the tenant's hybrid retriever, and the library's safe calculator.
2. Every model call goes through the tenant's ``llm_gateway.Gateway``.
3. Citations in the final answer are checked against the passages that were actually
   retrieved during the run; an answer is *grounded* only if it cites at least one passage
   and every citation refers to a retrieved document.
4. If a response schema is supplied, the result is re-emitted under constrained decoding.

An optional ``on_event`` callback receives each step as it happens (``step`` when the agent
chooses a tool, ``observation`` when the tool returns). The streaming endpoint forwards these
as server-sent events; an exception raised by the callback aborts the run, which is how a
disconnected client stops the agent.

Spans for the run, the model calls, and the tool calls are recorded on the caller's
``llm_observability.Tracer``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from constrained_generation_library import Property
from llm_gateway import GatewayResponse
from llm_observability import Tracer
from react_agent import AgentAction, Callbacks, FunctionTool, ReActAgent, ToolRegistry
from react_agent.tools import CalculatorTool

from agent_platform.knowledge import KnowledgeBase, Passage
from agent_platform.model import (
    NOT_FOUND,
    SEARCH_TOOL,
    GatewayModel,
    TenantGateways,
    format_passages,
)
from agent_platform.structured import generate_structured

_CITATION = re.compile(r"\[([^\[\]\n]+)\]")
MAX_QUESTION_CHARS = 2000
OBSERVATION_PREVIEW = 500

EventSink = Callable[[str, dict[str, Any]], None]


@dataclass
class AskResult:
    """The answer to a question, with its evidence and execution record."""

    question: str
    answer: str
    citations: list[str]
    grounded: bool
    finished: bool
    sources: list[dict[str, Any]]
    steps: list[dict[str, str]]
    model_calls: int = 0
    cached_calls: int = 0
    structured: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "grounded": self.grounded,
            "finished": self.finished,
            "sources": self.sources,
            "steps": self.steps,
            "model_calls": self.model_calls,
            "cached_calls": self.cached_calls,
        }
        if self.structured is not None:
            data["structured"] = self.structured
        return data


def normalize_question(question: str) -> str:
    """Collapse whitespace so a question cannot inject extra lines into the ReAct prompt."""
    cleaned = " ".join(question.split())
    if not cleaned:
        raise ValueError("'question' must be a non-empty string")
    if len(cleaned) > MAX_QUESTION_CHARS:
        raise ValueError(f"'question' exceeds {MAX_QUESTION_CHARS} characters")
    return cleaned


class AgentRuntime:
    """Runs the ReAct agent for a tenant over its knowledge base, through its gateway."""

    def __init__(
        self,
        knowledge: KnowledgeBase,
        gateways: TenantGateways | None = None,
        *,
        top_k: int = 3,
        max_steps: int = 4,
    ) -> None:
        self.knowledge = knowledge
        self.gateways = gateways or TenantGateways()
        self._top_k = top_k
        self._max_steps = max_steps

    def _tools(self, tenant: str, tracer: Tracer, retrieved: list[Passage]) -> ToolRegistry:
        def search(query: str) -> str:
            with tracer.span("tool.search_knowledge_base", tenant=tenant) as span:
                passages = self.knowledge.search(tenant, query, k=self._top_k)
                retrieved.extend(passages)
                span.set_attribute("results", len(passages))
                span.set_attribute("doc_ids", [p.doc_id for p in passages])
                return format_passages([(p.doc_id, p.text) for p in passages])

        calculator = CalculatorTool()

        def calculate(expression: str) -> str:
            with tracer.span("tool.calculator", expression=expression):
                return calculator(expression)

        return ToolRegistry(
            [
                FunctionTool(
                    name=SEARCH_TOOL,
                    description=(
                        "Search the tenant's ingested documents. Input: a natural-language "
                        "query. Output: passages prefixed with their [document id]."
                    ),
                    function=search,
                ),
                FunctionTool(
                    name=calculator.name, description=calculator.description, function=calculate
                ),
            ]
        )

    def ask(
        self,
        tenant: str,
        question: str,
        *,
        tracer: Tracer,
        schema: list[Property] | None = None,
        on_event: EventSink | None = None,
    ) -> AskResult:
        """Answer ``question`` for ``tenant``. Raises ``GatewayBlockedError`` if refused."""
        question = normalize_question(question)
        retrieved: list[Passage] = []
        responses: list[GatewayResponse] = []

        def on_response(response: GatewayResponse) -> None:
            responses.append(response)

        def on_action(index: int, action: AgentAction) -> None:
            span_name = f"agent.action.{action.tool}"
            with tracer.span(span_name, step=index, tool_input=action.tool_input[:200]):
                pass
            if on_event is not None:
                on_event(
                    "step",
                    {
                        "index": index,
                        "thought": action.thought,
                        "tool": action.tool,
                        "input": action.tool_input,
                    },
                )

        def on_observation(index: int, observation: str) -> None:
            if on_event is not None:
                on_event(
                    "observation",
                    {"index": index, "observation": observation[:OBSERVATION_PREVIEW]},
                )

        with tracer.span("agent.run", tenant=tenant) as run_span:
            model = GatewayModel(self.gateways, tenant, on_response=on_response)
            agent = ReActAgent(
                model,
                self._tools(tenant, tracer, retrieved),
                max_steps=self._max_steps,
                callbacks=Callbacks(on_action=on_action, on_observation=on_observation),
            )
            result = agent.run(question)
            answer = result.answer if result.finished else NOT_FOUND
            retrieved_ids = {p.doc_id for p in retrieved}
            cited = list(dict.fromkeys(_CITATION.findall(answer)))
            grounded = bool(cited) and all(doc in retrieved_ids for doc in cited)
            run_span.set_attribute("steps", result.step_count)
            run_span.set_attribute("grounded", grounded)
            run_span.set_attribute("model_calls", len(responses))

        sources: list[dict[str, Any]] = []
        seen: set[str] = set()
        for passage in retrieved:
            if passage.chunk_id in seen:
                continue
            seen.add(passage.chunk_id)
            sources.append(
                {"doc_id": passage.doc_id, "chunk_id": passage.chunk_id, "score": passage.score}
            )
        outcome = AskResult(
            question=question,
            answer=answer,
            citations=cited,
            grounded=grounded,
            finished=result.finished,
            sources=sources,
            steps=[
                {
                    "tool": step.action.tool,
                    "input": step.action.tool_input,
                    "observation": step.observation[:OBSERVATION_PREVIEW],
                }
                for step in result.steps
            ],
            model_calls=len(responses),
            cached_calls=sum(1 for r in responses if r.cached),
        )
        if schema is not None:
            with tracer.span("structured.decode", properties=len(schema)) as span:
                fields: dict[str, Any] = {
                    "answer": answer,
                    "source": cited[0] if cited else "",
                    "citation": cited[0] if cited else "",
                    "citations": len(cited),
                    "citation_count": len(cited),
                    "grounded": grounded,
                    "confidence": "high" if grounded else "low",
                }
                structured = generate_structured(schema, fields)
                span.set_attribute("tokens", structured.tokens)
                outcome.structured = structured.value
        return outcome

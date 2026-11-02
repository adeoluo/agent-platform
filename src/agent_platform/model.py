"""The language-model path: a deterministic offline provider behind the safety gateway.

Two adapters connect the libraries:

* :class:`GroundedPolicyProvider` implements the ``llm_gateway`` provider protocol. It reads
  a ReAct prompt and emits the next step: a calculator call for arithmetic, a knowledge-base
  search otherwise, and, once observations are in, a final answer built from the best
  supporting sentence with a ``[doc_id]`` citation. It is a transparent, deterministic
  stand-in for a hosted model so the whole platform runs offline and reproducibly. A real
  provider (for example ``llm_gateway.AnthropicProvider``) can be swapped in unchanged.
* :class:`GatewayModel` implements the ``react_agent`` language-model protocol by sending
  every completion through a tenant's ``llm_gateway.Gateway``, so injection screening, PII
  redaction, output policy, and caching apply to every model call the agent makes,
  including calls whose prompts contain retrieved document text.
"""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from hybrid_rag import tokenize
from llm_gateway import Gateway, GatewayResponse, Provider, SemanticCache

SEARCH_TOOL = "search_knowledge_base"
CALCULATOR_TOOL = "calculator"
NOT_FOUND = "I could not find an answer in the knowledge base."
NO_RESULTS = "No relevant documents found."
PASSAGE_SEPARATOR = " || "

_STEP = re.compile(
    r"^Action:[ \t]*(?P<tool>[^\n]*)\nAction Input:[ \t]*(?P<input>[^\n]*)\n"
    r"Observation:[ \t]*(?P<obs>[^\n]*)",
    re.MULTILINE,
)
_PASSAGE = re.compile(r"^\[(?P<doc>[^\]]+)\]\s*(?P<text>.*)$", re.DOTALL)
_EXPRESSION = re.compile(r"[-+]?[\d.(][\d.\s+\-*/%()]*")
_OPERATOR = re.compile(r"\d\s*(\*\*|[+\-*/%])\s*[\d(]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_ARITHMETIC_LEADS = ("what is", "what's", "calculate", "compute", "evaluate", "how much is")


class GatewayBlockedError(Exception):
    """Raised when the gateway refuses a model call (injection or output policy)."""

    def __init__(self, status: str, reasons: Sequence[str]) -> None:
        super().__init__(f"{status}: {', '.join(reasons) or 'no reason given'}")
        self.status = status
        self.reasons = tuple(reasons)


def extract_expression(question: str) -> str | None:
    """Return an arithmetic expression if ``question`` is an arithmetic request."""
    lowered = question.strip().lower()
    if not lowered.startswith(_ARITHMETIC_LEADS):
        return None
    best = ""
    for match in _EXPRESSION.finditer(question):
        candidate = match.group(0).strip().rstrip(".")
        if _OPERATOR.search(candidate) and len(candidate) > len(best):
            best = candidate
    return best or None


def format_passages(passages: Sequence[tuple[str, str]]) -> str:
    """Render ``(doc_id, text)`` passages as one observation line the policy can parse."""
    if not passages:
        return NO_RESULTS
    rendered = []
    for doc_id, text in passages:
        flat = " ".join(text.split()).replace("||", "|")
        rendered.append(f"[{doc_id}] {flat}")
    return PASSAGE_SEPARATOR.join(rendered)


def _question(prompt: str) -> str:
    question = ""
    for line in prompt.splitlines():
        if line.startswith("Question:"):
            question = line.split(":", 1)[1].strip()
    return question


def _best_sentence(question: str, observation: str) -> tuple[str, str] | None:
    query_terms = set(tokenize(question))
    if not query_terms:
        return None
    needed = max(1, math.ceil(0.34 * len(query_terms)))
    best: tuple[int, int, str, str] | None = None
    for rank, raw in enumerate(observation.split(PASSAGE_SEPARATOR)):
        match = _PASSAGE.match(raw.strip())
        if match is None:
            continue
        doc_id = match.group("doc")
        for sentence in _SENTENCE.split(match.group("text").strip()):
            overlap = len(query_terms & set(tokenize(sentence)))
            key = (overlap, -rank)
            if overlap >= needed and (best is None or key > (best[0], best[1])):
                best = (overlap, -rank, sentence.strip(), doc_id)
    if best is None:
        return None
    return best[2], best[3]


class GroundedPolicyProvider:
    """A deterministic provider that drives the ReAct loop to grounded, cited answers."""

    name = "grounded-policy-v1"

    def complete(self, system: str, prompt: str) -> str:
        question = _question(prompt)
        steps = list(_STEP.finditer(prompt))
        if not steps:
            expression = extract_expression(question)
            if expression is not None:
                return (
                    "Thought: This is arithmetic, so I will use the calculator.\n"
                    f"Action: {CALCULATOR_TOOL}\nAction Input: {expression}\n"
                )
            return (
                "Thought: I should look this up in the knowledge base.\n"
                f"Action: {SEARCH_TOOL}\nAction Input: {question}\n"
            )
        last = steps[-1]
        tool = last.group("tool").strip()
        observation = last.group("obs").strip()
        if tool == CALCULATOR_TOOL:
            if observation.startswith("Error"):
                return f"Thought: The calculation failed.\nFinal Answer: {observation}"
            return f"Thought: I have the result.\nFinal Answer: The result is {observation}."
        if observation == NO_RESULTS or observation.startswith("Error"):
            return f"Thought: Nothing relevant was found.\nFinal Answer: {NOT_FOUND}"
        found = _best_sentence(question, observation)
        if found is None:
            return f"Thought: The passages do not answer the question.\nFinal Answer: {NOT_FOUND}"
        sentence, doc_id = found
        return (
            f"Thought: The passage from {doc_id} answers it.\nFinal Answer: {sentence} [{doc_id}]"
        )


@dataclass
class GatewayCallStats:
    """Aggregate model-call counters across tenants, for the metrics endpoint."""

    requests: int = 0
    model_calls: int = 0
    cache_hits: int = 0
    blocked_injection: int = 0
    blocked_policy: int = 0
    pii_redacted: int = 0


class TenantGateways:
    """One ``llm_gateway.Gateway`` per tenant, so caches and counters never mix.

    The cache threshold is 1.0 (identical token sets only). ReAct prompts share a long
    system block, so a looser similarity threshold would let two different questions hit
    each other's cache entries.
    """

    def __init__(self, provider: Provider | None = None, *, cache_threshold: float = 1.0):
        self._provider: Provider = provider or GroundedPolicyProvider()
        self._threshold = cache_threshold
        self._gateways: dict[str, tuple[Gateway, threading.Lock]] = {}
        self._lock = threading.Lock()

    def _get(self, tenant: str) -> tuple[Gateway, threading.Lock]:
        with self._lock:
            entry = self._gateways.get(tenant)
            if entry is None:
                gateway = Gateway(
                    provider=self._provider,
                    system="You answer only from the tenant's documents and cite sources.",
                    cache=SemanticCache(threshold=self._threshold),
                )
                entry = (gateway, threading.Lock())
                self._gateways[tenant] = entry
            return entry

    def handle(self, tenant: str, prompt: str) -> GatewayResponse:
        """Run ``prompt`` through ``tenant``'s gateway."""
        gateway, lock = self._get(tenant)
        with lock:
            return gateway.handle(prompt)

    def stats(self) -> dict[str, GatewayCallStats]:
        """Return a snapshot of per-tenant gateway counters."""
        with self._lock:
            entries = list(self._gateways.items())
        snapshot: dict[str, GatewayCallStats] = {}
        for tenant, (gateway, lock) in entries:
            with lock:
                s = gateway.stats
                snapshot[tenant] = GatewayCallStats(
                    requests=s.requests,
                    model_calls=s.model_calls,
                    cache_hits=s.cache_hits,
                    blocked_injection=s.blocked_injection,
                    blocked_policy=s.blocked_policy,
                    pii_redacted=s.pii_redacted,
                )
        return snapshot


class GatewayModel:
    """A ``react_agent`` language model whose every call goes through the gateway."""

    def __init__(
        self,
        gateways: TenantGateways,
        tenant: str,
        *,
        on_response: Callable[[GatewayResponse], None] | None = None,
    ) -> None:
        self._gateways = gateways
        self._tenant = tenant
        self._on_response = on_response

    @property
    def name(self) -> str:
        return "gateway:grounded-policy-v1"

    def complete(self, prompt: str, *, stop: Sequence[str] | None = None) -> str:
        response = self._gateways.handle(self._tenant, prompt)
        if self._on_response is not None:
            self._on_response(response)
        if response.status != "ok":
            raise GatewayBlockedError(response.status, response.reasons)
        text = response.text
        for marker in stop or ():
            index = text.find(marker)
            if index != -1:
                text = text[:index]
        return text

# ADR 0005: Compose existing libraries through thin adapters

## Status

Accepted

## Context

The platform reuses eight separately developed libraries. Their interfaces were designed
independently: for example, the gateway's provider protocol is `complete(system, prompt)`
while the agent's language-model protocol is `complete(prompt, stop=...)`.

## Decision

Depend on each library as a git dependency pinned in `uv.lock` and bridge mismatched
interfaces with small adapters in this repository. The libraries are not modified.

| Adapter | Bridges |
| --- | --- |
| `GatewayModel` | `react_agent.LanguageModel` to `llm_gateway.Gateway` |
| `HnswRetriever` | `ann_vector_search_library.HNSW` to the `hybrid_rag` retriever protocol |
| `AgentWorker`, `ResearchPlanner`, `OutputCritic` | `multi_agent_orchestrator` protocols to `AgentRuntime` |
| `TraceStore` | `llm_observability` spans to SQLite, re-homed under a job trace id |
| `PlatformModel` | `evaluation_harness.Model` to the HTTP API |

## Consequences

* Each library stays independently tested and versioned.
* Known library limitations surface as platform behaviour. For example the `hybrid_rag`
  sentence chunker splits on any period, including decimal points; the eval suite reports
  the resulting miss rather than hiding it.
* Upgrading a library is a lock-file change followed by the full test suite.

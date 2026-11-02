# ADR 0003: Tenant isolation by construction

## Status

Accepted

## Context

Several tenants share one process and one database. A tenant must never retrieve, be
answered from, or observe another tenant's documents, cached responses, jobs, or traces.

## Decision

Isolation is derived from one fact: the tenant is read from the authenticated key row and
is never accepted from request input. Every component is then keyed by that tenant.

* **Retrieval.** Each tenant has its own retriever built from `SELECT ... WHERE tenant = ?`.
  There is no shared index to filter, so a ranking bug cannot leak another tenant's chunk.
  Chunks also carry their tenant in metadata and results are filtered on it as a second
  check.
* **Gateway.** Each tenant has its own `llm_gateway.Gateway`, so the semantic cache and
  counters are per tenant.
* **Rate limits.** One token bucket per tenant.
* **Jobs and traces.** Reads include the tenant in the query. A foreign id returns 404, not
  403, so ids cannot be probed for existence.
* **Workers.** A job stores its tenant, and the worker runs the agent for that tenant only.

## Consequences

* Tests ingest overlapping content (including identical document ids) for two tenants and
  assert, through the API, that answers, sources, jobs, and traces never cross.
* Memory grows with the number of active tenants because each has its own index. Indexes are
  rebuilt lazily when a tenant's `kb_version` changes.
* Isolation is logical, not physical. All tenants share one process and one database file.

# ADR 0007: Hosted providers with classified errors and ordered failover

## Status

Accepted. Extends ADR 0004.

## Context

The offline policy keeps tests deterministic, but a deployment needs hosted models, and a
single provider is a single point of failure. Retrying is only useful for some errors:
a rate limit or a 503 may succeed on retry, while a bad key or an unknown model never will.

## Decision

`agent_platform.providers` adds `OpenAIProvider` (Chat Completions) and `AnthropicProvider`
(Messages) on the standard-library HTTP client, both implementing the `llm_gateway`
provider protocol so they sit behind the same gateway as the offline policy.

* HTTP 429, 5xx, timeouts, and connection errors raise `llm_gateway.ProviderError`
  (retryable). Other 4xx raise `ProviderRequestError`, which is never retried.
* `AGENT_PLATFORM_PROVIDER` takes an ordered list, for example `anthropic,openai,offline`.
  More than one entry builds `llm_gateway.ResilientProvider` with exponential backoff
  (0.5 s doubling, capped at 8 s) and failover to the next entry.
* The API maps an exhausted chain to 503 with `Retry-After` and a rejected request to 502,
  and counts both in `agent_platform_provider_errors_total`.
* Defaults are the current small models, `gpt-6-luna` and `claude-haiku-5-5`. OpenAI
  requests set `reasoning_effort: "none"` and temperature 0, so answers repeat and no hidden
  reasoning tokens are billed. Anthropic requests use API version `2026-10-21` and send no
  sampling parameters, because that version and the Claude 5 models accept only the defaults;
  their token budget (2048) leaves room for adaptive thinking. Token usage reported by the
  provider is accumulated per provider instance.

## Consequences

* Tests run both providers against a local fake of each HTTP API: request shape and
  headers, retry classification, failover order, and full platform runs where a scripted
  hosted model drives the real tools, gateway, and citation check. No test calls a real
  hosted API, and CI needs no keys.
* The eval suite has not been run against hosted models in this repository, so no hosted
  accuracy numbers are claimed.
* Gateway calls are serialized per tenant because the gateway's cache and counters are not
  thread safe. With a hosted model, one tenant's concurrent requests wait on each other's
  model latency; different tenants do not.

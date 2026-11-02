# ADR 0004: A deterministic policy provider behind the real gateway

## Status

Accepted

## Context

The platform must run offline and deterministically in tests and CI, but the request path
should be the same one a hosted model would use, including the safety gateway.

## Decision

`GroundedPolicyProvider` implements the `llm_gateway` provider protocol. It parses the
ReAct prompt and returns the next step: a calculator call for arithmetic, a knowledge-base
search otherwise, then a final answer made of the best supporting sentence and its
`[doc_id]` citation, or an explicit "could not find" when no passage overlaps enough with
the question. It sits behind a normal `llm_gateway.Gateway`, which the `react_agent` loop
reaches through the `GatewayModel` adapter.

## Consequences

* The gateway, agent loop, tools, retrieval, tracing, and citation checks are all exercised
  for real; only text generation is rule based.
* Answers are extractive. The eval suite measures the retrieval and grounding pipeline, not
  the quality of a generative model.
* `llm_gateway.AnthropicProvider` implements the same protocol and can be passed to
  `TenantGateways` without other changes; it is not exercised by tests.
* The gateway cache threshold is set to 1.0 because ReAct prompts share a long system block
  and a looser token-overlap threshold would return another question's cached step.

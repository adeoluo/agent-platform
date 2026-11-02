# ADR 0006: Stream agent steps over server-sent events, not raw tokens

## Status

Accepted

## Context

A ReAct run makes several model calls and tool calls before it answers. Clients (the
`chat` command, web front ends) want to show progress while that happens. Hosted models can
stream tokens, but every completion in this platform passes through the gateway's output
policy, which judges the complete text.

## Decision

`POST /v1/ask/stream` returns `text/event-stream` with the events `start`, `step` (tool
chosen), `observation` (tool result, truncated), then `answer` or `error`, then `done`.
Events come from the `react_agent` callbacks through an `on_event` sink on
`AgentRuntime.ask`.

* Input is validated before the response starts, so bad requests still get a normal 4xx.
* The stream is written inside the request's root span. The persisted trace covers the
  whole run and records `streamed` and `stream_completed`.
* A failed write raises from the sink. The exception propagates out of the agent loop, so a
  disconnected client stops further model and tool calls instead of paying for an answer
  nobody receives.
* Failures after the headers are sent (gateway refusal, provider outage, internal error)
  are reported as an in-band `error` event carrying the HTTP status the non-streaming
  endpoint would have returned. The server never writes a second response.

## Consequences

* Tokens of the final answer are not streamed. Releasing text before the output policy has
  seen all of it would let a blocked completion leak in part. Time to first event is one
  model call; time to the answer is unchanged.
* The connection closes after `done` (no keep-alive for streams).
* `tests/test_streaming_mcp.py` checks event order, equality with the non-streaming answer,
  pre-stream 4xx responses, in-band refusals, and that a failing sink stops the run before
  the second model call.

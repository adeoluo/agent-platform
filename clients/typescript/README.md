# agent-platform TypeScript client

Typed client for the [agent-platform](../../README.md) HTTP API. It has no runtime
dependencies and uses the runtime's `fetch` (Node 26 and later, Deno, Bun, browsers).

```ts
import { AgentPlatformClient, APIError } from "@adeoluo/agent-platform-client";

const client = new AgentPlatformClient({
  baseUrl: "http://127.0.0.1:8080",
  apiKey: process.env.KEY,
});

await client.ingest([{ id: "refund-policy", text: "Refunds within 30 days of purchase." }]);

for await (const { event, data } of client.askStream("How long do refunds take?")) {
  if (event === "step") console.log(`[${data.index}] ${data.tool}: ${data.input}`);
  if (event === "answer") console.log(data.answer, data.citations, data.grounded);
}

try {
  await client.ask("...");
} catch (error) {
  if (error instanceof APIError && error.retryable) {
    // 429 or 503: wait error.retryAfterSeconds and retry
  }
}
```

Breaking out of an `askStream` loop closes the connection, which stops the agent run on the
server. Errors are `APIError` with `status`, `code`, `message`, `traceId`, and
`retryAfterSeconds`.

## Development

```bash
npm ci
npm run lint && npm run typecheck
npm run test:coverage   # starts the real Python server (needs uv) and tests against it
npm run build
```

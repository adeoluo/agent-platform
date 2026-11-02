/**
 * Stream a question's steps from a running server:
 *   AGENT_PLATFORM_URL=http://127.0.0.1:8080 AGENT_PLATFORM_API_KEY=ap_... \
 *     node --experimental-strip-types examples/stream.ts "How long is the warranty?"
 */

import { AgentPlatformClient } from "../src/index.js";

const client = new AgentPlatformClient({
  baseUrl: process.env.AGENT_PLATFORM_URL ?? "http://127.0.0.1:8080",
  apiKey: process.env.AGENT_PLATFORM_API_KEY,
});

for await (const { event, data } of client.askStream(
  process.argv[2] ?? "How long is the warranty?",
)) {
  if (event === "step") console.log(`[${data.index}] ${data.tool}: ${data.input}`);
  if (event === "answer") console.log(`\n${data.answer} (grounded: ${data.grounded})`);
  if (event === "error") console.log(`error ${data.status}: ${data.message}`);
}

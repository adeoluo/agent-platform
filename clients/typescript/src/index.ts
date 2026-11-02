/** TypeScript client for agent-platform: typed requests, streaming steps, typed errors. */

export { AgentPlatformClient, APIError } from "./client.js";
export type { ClientOptions } from "./client.js";
export { parseEventStream } from "./sse.js";
export type * from "./types.js";

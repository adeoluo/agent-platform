/**
 * Typed client for the agent-platform HTTP API.
 *
 * Uses the runtime's `fetch` (Node 22+, Deno, Bun, browsers) and has no dependencies.
 * Every method throws {@link APIError} for a non-2xx response, carrying the server's
 * `error` code, `message`, HTTP status, and trace id when present.
 */

import { parseEventStream } from "./sse.js";
import type {
  Answer,
  Document,
  IngestResult,
  Job,
  ResponseSchema,
  StreamEvent,
  Trace,
} from "./types.js";

export interface ClientOptions {
  baseUrl: string;
  apiKey?: string;
  /** Per-request timeout in milliseconds for non-streaming calls (default 30 000). */
  timeoutMs?: number;
  fetch?: typeof fetch;
}

export class APIError extends Error {
  override readonly name = "APIError";

  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly traceId: string | undefined,
    readonly retryAfterSeconds: number | undefined,
  ) {
    super(`HTTP ${status} ${code}: ${message}`);
  }

  /** True for statuses where retrying later can succeed (429 and 503). */
  get retryable(): boolean {
    return this.status === 429 || this.status === 503;
  }
}

export class AgentPlatformClient {
  private readonly baseUrl: string;
  private readonly apiKey: string | undefined;
  private readonly timeoutMs: number;
  private readonly fetchImpl: typeof fetch;

  constructor(options: ClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.apiKey = options.apiKey;
    this.timeoutMs = options.timeoutMs ?? 30_000;
    this.fetchImpl = options.fetch ?? globalThis.fetch;
  }

  ingest(documents: Document[]): Promise<IngestResult> {
    return this.json("POST", "/v1/ingest", { documents });
  }

  ask(question: string, options: { responseSchema?: ResponseSchema } = {}): Promise<Answer> {
    const body: Record<string, unknown> = { question };
    if (options.responseSchema) body.response_schema = options.responseSchema;
    return this.json("POST", "/v1/ask", body);
  }

  /**
   * Stream the agent's steps as they happen. Breaking out of the loop, or aborting
   * `signal`, closes the connection, which stops the run on the server.
   */
  async *askStream(
    question: string,
    options: { signal?: AbortSignal } = {},
  ): AsyncGenerator<StreamEvent> {
    const response = await this.fetchImpl(`${this.baseUrl}/v1/ask/stream`, {
      method: "POST",
      headers: { ...this.headers(), Accept: "text/event-stream" },
      body: JSON.stringify({ question }),
      signal: options.signal ?? null,
    });
    if (!response.ok) throw await toError(response);
    if (!response.body)
      throw new APIError(response.status, "empty_body", "no stream body", undefined, undefined);
    const body = response.body;
    try {
      for await (const raw of parseEventStream(body)) {
        yield { event: raw.event, data: JSON.parse(raw.data) } as StreamEvent;
      }
    } finally {
      await body.cancel().catch(() => undefined);
    }
  }

  submitJob(questions: string[], options: { stepDelaySeconds?: number } = {}): Promise<Job> {
    return this.json("POST", "/v1/jobs", {
      kind: "research",
      questions,
      step_delay_seconds: options.stepDelaySeconds ?? 0,
    });
  }

  job(jobId: string): Promise<Job> {
    return this.json("GET", `/v1/jobs/${encodeURIComponent(jobId)}`);
  }

  /** Poll a job until it succeeds or fails, or `timeoutMs` passes. */
  async waitForJob(
    jobId: string,
    options: { intervalMs?: number; timeoutMs?: number } = {},
  ): Promise<Job> {
    const deadline = Date.now() + (options.timeoutMs ?? 60_000);
    for (;;) {
      const job = await this.job(jobId);
      if (job.status === "succeeded" || job.status === "failed") return job;
      if (Date.now() > deadline) throw new Error(`job ${jobId} still ${job.status} after timeout`);
      await new Promise((resolve) => setTimeout(resolve, options.intervalMs ?? 200));
    }
  }

  trace(traceId: string): Promise<Trace> {
    return this.json("GET", `/v1/traces/${encodeURIComponent(traceId)}`);
  }

  async health(): Promise<boolean> {
    try {
      const response = await this.fetchImpl(`${this.baseUrl}/health`, {
        signal: AbortSignal.timeout(this.timeoutMs),
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  private headers(): Record<string, string> {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;
    return headers;
  }

  private async json<T>(method: string, path: string, body?: unknown): Promise<T> {
    const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
      method,
      headers: this.headers(),
      body: body === undefined ? null : JSON.stringify(body),
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    if (!response.ok) throw await toError(response);
    return (await response.json()) as T;
  }
}

async function toError(response: Response): Promise<APIError> {
  let code = "http_error";
  let message = response.statusText;
  try {
    const payload = (await response.json()) as { error?: string; message?: string };
    code = payload.error ?? code;
    message = payload.message ?? message;
  } catch {
    // non-JSON error body
  }
  const retryAfter = response.headers.get("Retry-After");
  return new APIError(
    response.status,
    code,
    message,
    response.headers.get("X-Trace-Id") ?? undefined,
    retryAfter === null ? undefined : Number(retryAfter),
  );
}

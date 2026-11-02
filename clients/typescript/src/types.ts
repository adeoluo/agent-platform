/** Wire types of the agent-platform HTTP API, mirroring the server's JSON responses. */

export interface Document {
  id: string;
  text: string;
  metadata?: Record<string, unknown>;
}

export interface IngestResult {
  ingested: number;
  kb_version: number;
}

export interface Source {
  doc_id: string;
  chunk_id: string;
  score: number;
}

export interface Step {
  tool: string;
  input: string;
  observation: string;
}

export interface Answer {
  question: string;
  answer: string;
  citations: string[];
  grounded: boolean;
  finished: boolean;
  sources: Source[];
  steps: Step[];
  model_calls: number;
  cached_calls: number;
  trace_id: string;
  structured?: Record<string, unknown>;
}

export type SchemaProperty =
  | { name: string; type: "string"; max_length?: number }
  | { name: string; type: "integer" | "number" | "boolean" }
  | { name: string; type: "enum"; values: string[] };

export interface ResponseSchema {
  properties: SchemaProperty[];
}

export interface Job {
  job_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  trace_id: string;
  [key: string]: unknown;
}

export interface Trace {
  trace_id: string;
  spans: { name: string; attributes: Record<string, unknown>; [key: string]: unknown }[];
  [key: string]: unknown;
}

/** One server-sent event from `POST /v1/ask/stream`. */
export type StreamEvent =
  | { event: "start"; data: { trace_id: string } }
  | { event: "step"; data: { index: number; thought: string; tool: string; input: string } }
  | { event: "observation"; data: { index: number; observation: string } }
  | { event: "answer"; data: Answer }
  | { event: "error"; data: { status: number; error: string; message: string; trace_id: string } }
  | { event: "done"; data: Record<string, never> };

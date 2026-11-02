import { describe, expect, it } from "vitest";
import { parseEventStream } from "../src/sse.js";

async function* chunks(...parts: string[]): AsyncGenerator<Uint8Array> {
  const encoder = new TextEncoder();
  for (const part of parts) yield encoder.encode(part);
}

async function collect(...parts: string[]) {
  const out = [];
  for await (const event of parseEventStream(chunks(...parts))) out.push(event);
  return out;
}

describe("parseEventStream", () => {
  it("handles events split across chunks, CRLF, comments and multi-line data", async () => {
    expect(
      await collect(
        "event: st",
        'ep\r\ndata: {"a":\n',
        "data: 1}\r\n\r\n: keep-alive\n\n",
        "data: x",
      ),
    ).toEqual([
      { event: "step", data: '{"a":\n1}' },
      { event: "message", data: "x" },
    ]);
  });

  it("decodes multi-byte characters split between chunks", async () => {
    const bytes = new TextEncoder().encode("data: é\n\n");
    async function* split(): AsyncGenerator<Uint8Array> {
      yield bytes.slice(0, 7);
      yield bytes.slice(7);
    }
    const out = [];
    for await (const event of parseEventStream(split())) out.push(event);
    expect(out).toEqual([{ event: "message", data: "é" }]);
  });
});

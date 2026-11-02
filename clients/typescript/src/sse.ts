/** Incremental parser for a `text/event-stream` body (UTF-8 bytes in, events out). */

export interface RawEvent {
  event: string;
  data: string;
}

export async function* parseEventStream(body: AsyncIterable<Uint8Array>): AsyncGenerator<RawEvent> {
  const decoder = new TextDecoder();
  let buffer = "";
  let event = "message";
  let data: string[] = [];
  const flushLine = function* (line: string): Generator<RawEvent> {
    if (line === "") {
      if (data.length > 0) yield { event, data: data.join("\n") };
      event = "message";
      data = [];
    } else if (line.startsWith(":")) {
      // comment or keep-alive
    } else if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      data.push(line.slice(5).replace(/^ /, ""));
    }
  };
  for await (const chunk of body) {
    buffer += decoder.decode(chunk, { stream: true });
    let newline = buffer.indexOf("\n");
    while (newline !== -1) {
      const line = buffer.slice(0, newline).replace(/\r$/, "");
      buffer = buffer.slice(newline + 1);
      yield* flushLine(line);
      newline = buffer.indexOf("\n");
    }
  }
  buffer += decoder.decode();
  if (buffer.length > 0) yield* flushLine(buffer.replace(/\r$/, ""));
  yield* flushLine("");
}

/**
 * Global setup: start the real Python agent-platform server on a free port with a fresh
 * database, issue an API key, and expose both to the tests. The SDK is tested against the
 * actual API, not a mock of it.
 */

import { execFileSync, spawn, type ChildProcess } from "node:child_process";
import { mkdirSync, rmSync } from "node:fs";
import { createServer } from "node:net";
import { join, resolve } from "node:path";
import type { TestProject } from "vitest/node";

const REPO = resolve(import.meta.dirname, "../../..");
const WORK = resolve(import.meta.dirname, "../.test-work");

declare module "vitest" {
  export interface ProvidedContext {
    baseUrl: string;
    apiKey: string;
    dbPath: string;
  }
}

function freePort(): Promise<number> {
  return new Promise((done, fail) => {
    const server = createServer();
    server.once("error", fail);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      server.close(() => done(typeof address === "object" && address ? address.port : 0));
    });
  });
}

function cli(args: string[]): string {
  return execFileSync("uv", ["run", "--project", REPO, "agent-platform", ...args], {
    cwd: REPO,
    encoding: "utf8",
  });
}

let server: ChildProcess | undefined;

export async function setup(project: TestProject): Promise<void> {
  rmSync(WORK, { recursive: true, force: true });
  mkdirSync(WORK, { recursive: true });
  const dbPath = join(WORK, "platform.db");
  const port = await freePort();
  const created = cli(["keys", "create", "--db", dbPath, "--tenant", "acme"]);
  const apiKey = /api key: (\S+)/.exec(created)?.[1];
  if (!apiKey) throw new Error(`could not parse key from: ${created}`);

  server = spawn(
    "uv",
    ["run", "--project", REPO, "agent-platform", "serve", "--db", dbPath, "--port", String(port)],
    { cwd: REPO, stdio: ["ignore", "pipe", "inherit"] },
  );
  const baseUrl = `http://127.0.0.1:${port}`;
  const deadline = Date.now() + 90_000;
  for (;;) {
    try {
      if ((await fetch(`${baseUrl}/health`)).ok) break;
    } catch {
      // not listening yet
    }
    if (Date.now() > deadline) throw new Error("agent-platform server did not start");
    await new Promise((r) => setTimeout(r, 200));
  }
  project.provide("baseUrl", baseUrl);
  project.provide("apiKey", apiKey);
  project.provide("dbPath", dbPath);
}

export async function teardown(): Promise<void> {
  server?.kill("SIGTERM");
  rmSync(WORK, { recursive: true, force: true });
}

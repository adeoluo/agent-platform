"""An MCP server (stdio transport) that exposes the platform to MCP clients.

The server is a thin adapter over the authenticated HTTP API: every tool call becomes one
API request made with the configured key. Tenant isolation, rate limits, the safety
gateway, and tracing therefore apply to MCP clients exactly as they do to HTTP clients,
and the MCP process never opens the database.

Transport: newline-delimited JSON-RPC 2.0 on stdin and stdout (MCP stdio transport).
Implemented methods: ``initialize``, ``ping``, ``tools/list``, ``tools/call``, plus the
``notifications/initialized`` notification. Tool failures are reported as tool results
with ``isError: true`` so the calling model can see and react to them; protocol errors use
JSON-RPC error codes.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import IO, Any

from agent_platform import __version__
from agent_platform.client import APIError, Client

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    run: Callable[[Client, dict[str, Any]], Any]
    read_only: bool

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "annotations": {"readOnlyHint": self.read_only, "openWorldHint": False},
        }


def _str(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' must be a non-empty string")
    return value


def _ask(client: Client, args: dict[str, Any]) -> Any:
    return client.ask(_str(args, "question"))


def _ingest(client: Client, args: dict[str, Any]) -> Any:
    documents = args.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("'documents' must be a non-empty list")
    return client.ingest(documents)


def _submit(client: Client, args: dict[str, Any]) -> Any:
    questions = args.get("questions")
    if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
        raise ValueError("'questions' must be a list of strings")
    return client.submit_job(questions)


def _job(client: Client, args: dict[str, Any]) -> Any:
    return client.job(_str(args, "job_id"))


def _trace(client: Client, args: dict[str, Any]) -> Any:
    return client.trace(_str(args, "trace_id"))


TOOLS: tuple[Tool, ...] = (
    Tool(
        "ask",
        "Answer a question from this tenant's knowledge base. Returns the answer, the cited "
        "document ids, whether every citation was retrieved (grounded), and a trace id.",
        _schema({"question": {"type": "string", "maxLength": 2000}}, ["question"]),
        _ask,
        read_only=True,
    ),
    Tool(
        "ingest_documents",
        "Add documents to this tenant's knowledge base. Each document needs an 'id' and 'text'.",
        _schema(
            {
                "documents": {
                    "type": "array",
                    "minItems": 1,
                    "items": _schema(
                        {"id": {"type": "string"}, "text": {"type": "string"}}, ["id", "text"]
                    ),
                }
            },
            ["documents"],
        ),
        _ingest,
        read_only=False,
    ),
    Tool(
        "submit_research_job",
        "Queue a durable multi-question research job. Returns a job id to poll with get_job.",
        _schema(
            {"questions": {"type": "array", "minItems": 1, "items": {"type": "string"}}},
            ["questions"],
        ),
        _submit,
        read_only=False,
    ),
    Tool(
        "get_job",
        "Get a research job's status, per-step checkpoints, and results.",
        _schema({"job_id": {"type": "string"}}, ["job_id"]),
        _job,
        read_only=True,
    ),
    Tool(
        "get_trace",
        "Get the recorded spans (model calls, tool calls, timings) for a trace id.",
        _schema({"trace_id": {"type": "string"}}, ["trace_id"]),
        _trace,
        read_only=True,
    ),
)


class MCPServer:
    """Dispatches MCP JSON-RPC messages to tools backed by the platform API client."""

    def __init__(self, client: Client) -> None:
        self._client = client
        self._tools = {tool.name: tool for tool in TOOLS}

    def handle(self, message: Any) -> dict[str, Any] | None:
        """Handle one decoded message; return the response, or None for notifications."""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _error(None, INVALID_REQUEST, "expected a JSON-RPC 2.0 object")
        method = message.get("method")
        msg_id = message.get("id")
        if "id" not in message:
            return None  # notifications (initialized, cancelled) need no reply
        if not isinstance(method, str):
            return _error(msg_id, INVALID_REQUEST, "missing method")
        params = message.get("params") or {}
        if not isinstance(params, dict):
            return _error(msg_id, INVALID_PARAMS, "params must be an object")
        if method == "initialize":
            return _result(msg_id, self._initialize(params))
        if method == "ping":
            return _result(msg_id, {})
        if method == "tools/list":
            return _result(msg_id, {"tools": [t.describe() for t in self._tools.values()]})
        if method == "tools/call":
            return self._call(msg_id, params)
        return _error(msg_id, METHOD_NOT_FOUND, f"method not found: {method}")

    @staticmethod
    def _initialize(params: dict[str, Any]) -> dict[str, Any]:
        requested = params.get("protocolVersion")
        version = requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "agent-platform", "version": __version__},
            "instructions": (
                "Tools act on one tenant's knowledge base through the agent platform API. "
                "Answers cite document ids; 'grounded' is true only if every citation was "
                "retrieved during the run."
            ),
        }

    def _call(self, msg_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        tool = self._tools.get(name) if isinstance(name, str) else None
        if tool is None:
            return _error(msg_id, INVALID_PARAMS, f"unknown tool: {name}")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return _error(msg_id, INVALID_PARAMS, "arguments must be an object")
        try:
            output = tool.run(self._client, arguments)
        except ValueError as error:
            return _result(msg_id, _tool_error(str(error)))
        except APIError as error:
            return _result(msg_id, _tool_error(f"API error {error.status}: {error.payload}"))
        except OSError as error:
            return _result(msg_id, _tool_error(f"platform unreachable: {error}"))
        return _result(
            msg_id,
            {
                "content": [{"type": "text", "text": json.dumps(output, indent=2)}],
                "structuredContent": output if isinstance(output, dict) else {"result": output},
                "isError": False,
            },
        )


def _tool_error(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": True}


def _result(msg_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def serve(client: Client, stdin: IO[str] | None = None, stdout: IO[str] | None = None) -> None:
    """Run the stdio loop until stdin closes. Logs never go to stdout (it carries protocol)."""
    reader = stdin or sys.stdin
    writer = stdout or sys.stdout
    server = MCPServer(client)
    for line in reader:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            response: dict[str, Any] | None = _error(None, PARSE_ERROR, "invalid JSON")
        else:
            response = server.handle(message)
        if response is not None:
            writer.write(json.dumps(response) + "\n")
            writer.flush()

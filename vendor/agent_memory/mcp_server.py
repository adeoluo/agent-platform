"""Model Context Protocol server exposing memory tools to MCP clients.

The server is bound to one database and one namespace at start-up. Tools cannot switch
namespaces, so a model connected to this server can only read and write its own memories.

Tools: ``remember``, ``recall``, ``forget``, ``list_facts``.
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from agent_memory.memory import AgentMemory

INSTRUCTIONS = (
    "Long-term memory for this user. Call `recall` before answering questions that may depend "
    "on what the user said in earlier conversations, and `remember` when the user states a "
    "durable fact or preference. Facts are versioned: newer statements supersede older ones."
)


def build_server(memory: AgentMemory) -> MCPServer:
    from agent_memory import __version__

    server = MCPServer(name="agent-memory", instructions=INSTRUCTIONS, version=__version__)

    @server.tool(description="Store a statement in long-term memory and extract facts from it.")
    def remember(text: str) -> dict[str, Any]:
        result = memory.remember(text, session_id="mcp")
        return {
            "episode_id": result.episode.id,
            "facts": [fact.text for fact in result.facts],
            "preferences": [item.text for item in result.preferences],
            "actions": list(result.actions),
        }

    @server.tool(
        description=(
            "Retrieve the most relevant memories for a query. Each result includes a score "
            "breakdown and an explanation of why it was retrieved."
        )
    )
    def recall(query: str, k: int = 5, include_history: bool = False) -> list[dict[str, Any]]:
        k = min(max(k, 1), 50)
        return [hit.to_dict() for hit in memory.recall(query, k, include_history=include_history)]

    @server.tool(description="Delete one memory by its identifier.")
    def forget(memory_id: int) -> dict[str, Any]:
        return {"memory_id": memory_id, "forgotten": memory.forget(memory_id)}

    @server.tool(description="List semantic facts, optionally including superseded history.")
    def list_facts(
        subject: str | None = None, include_history: bool = False
    ) -> list[dict[str, Any]]:
        return [
            {
                "id": fact.id,
                "subject": fact.subject,
                "predicate": fact.predicate,
                "object": fact.object,
                "confidence": round(fact.confidence, 3),
                "status": fact.status.value,
                "superseded_by": fact.superseded_by,
            }
            for fact in memory.facts(subject, include_history=include_history)
        ]

    return server


def serve(database: str, namespace: str) -> None:  # pragma: no cover - blocks on stdio
    with AgentMemory(database, namespace=namespace) as memory:
        build_server(memory).run("stdio")

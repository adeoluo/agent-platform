"""Constructing the ReAct prompt from the task, tools, and running scratchpad.

The prompt is the agent's policy specification: it defines the exact output grammar, lists
the available tools with their descriptions, and replays the transcript so far so the
model can decide the next action. Keeping prompt construction in one place makes the
agent's behavior auditable and easy to tune.
"""

from __future__ import annotations

from react_agent.core.tool import ToolRegistry

_SYSTEM_TEMPLATE = """\
You are a careful reasoning agent. Answer the question by thinking step by step and using
tools when helpful. Always respond using this exact format, starting every step with
"Thought:":

Thought: your reasoning about what to do next
Action: the tool to use, exactly one of [{tool_names}]
Action Input: the input to pass to the tool

After each action you will be shown an Observation. Repeat the Thought/Action/Action Input
block as needed. When you can answer, respond with:

Thought: your final reasoning
Final Answer: the answer to the question

Available tools:
{tool_descriptions}"""


def build_system_prompt(tools: ToolRegistry) -> str:
    """Render the system prompt describing the format and the available tools."""
    descriptions = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)
    return _SYSTEM_TEMPLATE.format(
        tool_names=", ".join(tools.names) or "(none)",
        tool_descriptions=descriptions or "- (no tools available)",
    )


def build_prompt(question: str, tools: ToolRegistry, scratchpad: str) -> str:
    """Assemble the full prompt: system block, question, and the running scratchpad."""
    parts = [build_system_prompt(tools), "", f"Question: {question}"]
    if scratchpad:
        parts.append(scratchpad)
    # Trailing newline cues the model to begin its next "Thought:" line.
    return "\n".join(parts) + "\n"

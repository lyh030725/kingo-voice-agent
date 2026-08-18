"""Provider-neutral voice-agent persona and tool adapter."""

from __future__ import annotations

import json

from brain import (
    MODE_PROMPTS,
    SYSTEM_PROMPT,
    TOOLS,
    StageTimer,
    run_tool as run_brain_tool,
)

VOICE_TOOL_PROMPT = """
# Realtime context and filler
In this realtime session, context is not preloaded. For lecture, PDF, or course
concept questions, call search_course_materials before answering. Call
recall_weak_concepts only when learner history would improve the teaching step.
Use search_trusted_web only when course search is insufficient.

Only immediately before calling search_course_materials, recall_weak_concepts,
or search_trusted_web, say exactly one short topic-specific Korean filler. Do
not say a filler before show_visualization, after a tool result, or on a turn
with no slow tool call. Answer greetings, thanks, casual conversation, and
simple questions directly without a filler.
""".strip()

VOICE_CONTEXT_TOOLS = [{
    "type": "function",
    "name": "recall_weak_concepts",
    "description": (
        "Recall learner weaknesses relevant to the current topic. Call only when "
        "learner history would improve the teaching step."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Current question or concept.",
            }
        },
        "required": ["topic"],
        "additionalProperties": False,
    },
}, {
    "type": "function",
    "name": "search_course_materials",
    "description": "Search uploaded course PDFs before answering course-content questions.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Focused terms from the student's question.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}]


def persona(mode: str) -> str:
    """Return the shared teaching policy with conditional realtime fillers."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{MODE_PROMPTS.get(mode, MODE_PROMPTS['socratic'])}\n\n"
        f"{VOICE_TOOL_PROMPT}"
    )


def json_schemas() -> list[dict]:
    """Expose course retrieval plus the existing optional realtime tools."""
    return [*VOICE_CONTEXT_TOOLS, *(
        {"type": "function", **tool["function"]} for tool in TOOLS
    )]


async def run_tool(name: str, args: dict) -> object:
    """Run an optional course-agent tool and return JSON-compatible data."""
    return json.loads(await run_brain_tool(name, args, StageTimer()))

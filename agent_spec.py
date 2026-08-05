"""Provider-neutral voice-agent persona and tool adapter."""

from __future__ import annotations

import json

from brain import MODE_PROMPTS, SYSTEM_PROMPT, TOOLS, StageTimer, run_tool as run_brain_tool


def persona(mode: str) -> str:
    """Return the shared teaching persona for one voice session."""
    return f"{SYSTEM_PROMPT} {MODE_PROMPTS.get(mode, MODE_PROMPTS['socratic'])}"


def json_schemas() -> list[dict]:
    """Convert existing OpenAI chat tools to realtime function schemas."""
    return [{"type": "function", **tool["function"]} for tool in TOOLS]


async def run_tool(name: str, args: dict) -> object:
    """Run an existing course-agent tool and return JSON-compatible data."""
    return json.loads(await run_brain_tool(name, args, StageTimer()))

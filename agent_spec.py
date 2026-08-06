"""Provider-neutral voice-agent persona and tool adapter."""

from __future__ import annotations

import json

from brain import MODE_PROMPTS, SYSTEM_PROMPT, TOOLS, StageTimer, run_tool as run_brain_tool

VOICE_FILLER_PROMPT = """
# Voice-only filler
Immediately before the first tool call, say exactly one short Korean filler.
Either briefly confirm the student's topic, such as '소프트맥스 연산에 대해
물으신 거 맞죠?', or say '잠시만요. 강의 자료를 찾아볼게요.' Do not include
equations, source details, or the answer. Never repeat the filler after the tools
finish.
""".strip()


def persona(mode: str) -> str:
    """Return the shared teaching persona for one voice session."""
    return f"{SYSTEM_PROMPT}\n\n{VOICE_FILLER_PROMPT}\n\n{MODE_PROMPTS.get(mode, MODE_PROMPTS['socratic'])}"


def json_schemas() -> list[dict]:
    """Convert existing OpenAI chat tools to realtime function schemas."""
    return [{"type": "function", **tool["function"]} for tool in TOOLS]


async def run_tool(name: str, args: dict) -> object:
    """Run an existing course-agent tool and return JSON-compatible data."""
    return json.loads(await run_brain_tool(name, args, StageTimer()))

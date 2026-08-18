"""Provider-neutral voice-agent persona and tool adapter."""

from __future__ import annotations

import json

from brain import (
    SYSTEM_PROMPT,
    TOOLS,
    StageTimer,
    answer_instructions,
    prefetch_context,
    run_tool as run_brain_tool,
)

VOICE_FILLER_PROMPT = """
# Voice-only filler
Immediately before the first tool call, if one is later needed, the first
response after each student turn must already have said exactly one short Korean
filler while the server recalls student memory and searches course PDFs.
Make it fit the student's topic when possible, for example '소프트맥스 자료를
찾아볼게요.' or '그 부분을 강의 자료에서 확인해볼게요.' Do not answer the
question, explain the concept, mention sources, read equations, or call tools.
The server will request a separate final response after the context is ready.
""".strip()


def persona(mode: str) -> str:
    """Return the lightweight session persona used for dynamic filler turns."""
    return (
        "You are KINGO VOICE TA for a Sungkyunkwan University student. "
        "Speak natural polite Korean.\n\n"
        f"{VOICE_FILLER_PROMPT}"
    )


def answer_persona(mode: str, context: dict) -> str:
    """Return the same final-answer policy used by the text path."""
    return answer_instructions(mode, context)


def json_schemas() -> list[dict]:
    """Expose only optional realtime tools; mandatory retrieval is server-side."""
    return [{"type": "function", **tool["function"]} for tool in TOOLS]


async def run_tool(name: str, args: dict) -> object:
    """Run an optional course-agent tool and return JSON-compatible data."""
    return json.loads(await run_brain_tool(name, args, StageTimer()))

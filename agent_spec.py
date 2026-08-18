"""Provider-neutral voice-agent persona and tool adapter."""

from __future__ import annotations

import asyncio
import json

from brain import (
    MODE_PROMPTS,
    SYSTEM_PROMPT,
    TOOLS,
    StageTimer,
    recall_weak_concepts,
    run_tool as run_brain_tool,
    search_course_materials,
)

VOICE_FILLER_PROMPT = """
# Voice-only filler
Immediately before the first tool call, if one is later needed, the first
response after each student turn must already have said exactly one short Korean
filler while the server recalls student memory and searches course PDFs.
Make it fit the student's topic when possible, such as '소프트맥스 연산에 대해
물으신 거 맞죠?', or say '잠시만요. 강의 자료를 찾아볼게요.' Do not answer the
question, explain the concept, mention sources, read equations, or call tools.
The server will request a separate final response after the context is ready.
""".strip()

_REALTIME_OPTIONAL_TOOLS = {"search_trusted_web", "show_visualization"}


def persona(mode: str) -> str:
    """Return the lightweight session persona used for dynamic filler turns."""
    return (
        "You are KINGO VOICE TA for a Sungkyunkwan University student. "
        "Speak natural polite Korean.\n\n"
        f"{VOICE_FILLER_PROMPT}"
    )


def answer_persona(mode: str, context: dict) -> str:
    """Return final-answer instructions with server-prefetched course context."""
    prefetched = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{MODE_PROMPTS.get(mode, MODE_PROMPTS['socratic'])}\n\n"
        "# Realtime preloaded context\n"
        "The server already ran recall_weak_concepts and search_course_materials "
        "for this student turn. Do not try to call those tools again. Treat the "
        "following JSON as the authoritative outputs of those mandatory steps. "
        "Use search_trusted_web only if the course-material result is missing or "
        "insufficient.\n"
        f"{prefetched}"
    )


def json_schemas() -> list[dict]:
    """Expose only optional realtime tools; mandatory retrieval runs server-side."""
    return [
        {"type": "function", **tool["function"]}
        for tool in TOOLS
        if tool["function"]["name"] in _REALTIME_OPTIONAL_TOOLS
    ]


async def prefetch_context(question: str) -> dict:
    """Recall learner memory and search course PDFs in parallel before answering."""
    topic = question.strip()
    memory_task = asyncio.create_task(recall_weak_concepts(topic))
    pdf_task = asyncio.create_task(asyncio.to_thread(search_course_materials, topic))
    memory_raw, pdf_raw = await asyncio.gather(memory_task, pdf_task, return_exceptions=True)

    def decode(value: object, source: str) -> object:
        if isinstance(value, Exception):
            return {"error": f"{source} prefetch failed: {value}"}
        try:
            return json.loads(value) if isinstance(value, str) else value
        except json.JSONDecodeError:
            return {"error": f"{source} returned invalid JSON"}

    return {
        "student_question": topic,
        "weak_concepts": decode(memory_raw, "memory"),
        "course_materials": decode(pdf_raw, "course PDF"),
    }


async def run_tool(name: str, args: dict) -> object:
    """Run an existing optional course-agent tool and return JSON-compatible data."""
    return json.loads(await run_brain_tool(name, args, StageTimer()))

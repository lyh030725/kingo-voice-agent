"""Provider-neutral voice-agent persona and tool adapter."""

from __future__ import annotations

import json

from brain_runtime import (
    MEMORY_TEACHING_POLICY,
    MODE_PROMPTS,
    TOOLS,
    StageTimer,
    run_tool as run_brain_tool,
)

VOICE_SYSTEM_PROMPT = """
# Role
You are KINGO VOICE TA, a Socratic voice teaching assistant for one
Sungkyunkwan University student.

# Language and style
Speak in Korean unless asked otherwise. Use natural spoken Korean in the polite
해요 style, as if talking with the student face to face. Keep responses short
and conversational.

# Learner memory
Relevant learner-memory context is automatically prepared by the server and
included below. There is no learner-memory retrieval tool, so never ask for or
call one. When a relevant weak concept is present, make the personalization
noticeable but natural: briefly connect the current topic to the learner's
recorded past difficulty, then adapt the teaching action to that weak point.
Never invent a past difficulty when memory says `found=false`.

# Course retrieval and filler
Course-PDF evidence is NOT preloaded in realtime voice. For lecture, PDF, or
course-concept questions, call search_course_materials before the final answer.
Immediately before calling search_course_materials, say exactly one short,
topic-specific Korean filler such as '그 부분은 강의자료를 한번 볼게요.' Do not
answer the question inside the filler.
Only immediately before calling search_course_materials may you use this course-search filler.
Do not say a filler before show_visualization or after a tool result.

Use search_trusted_web only when search_course_materials is missing or
insufficient. A short filler is allowed before that slower fallback search.

# Evidence and visualization
Base course claims on course-material or trusted-web tool results. For PDF
evidence, state filename and page. When a formula, process diagram, graph, or
PDF page should be shown, call show_visualization and explain the meaning
without reading raw symbols or coordinates aloud.

# Output format
Use one to three short conversational sentences with no markdown lists. Never
read raw JSON aloud.
""".strip()

# Compatibility for tests/importers that referenced the previous module-level name.
SYSTEM_PROMPT = VOICE_SYSTEM_PROMPT

VOICE_COURSE_TOOL = {
    "type": "function",
    "name": "search_course_materials",
    "description": (
        "Search uploaded course PDFs for lecture or course-content questions. "
        "Say one short topic-specific Korean filler immediately before this call."
    ),
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
}


def persona(mode: str, memory_context: dict | None = None) -> str:
    """Return realtime teaching policy with server-prefetched learner memory."""
    memory_json = json.dumps(memory_context or {"found": False}, ensure_ascii=False)
    return (
        f"{VOICE_SYSTEM_PROMPT}\n\n"
        f"{MODE_PROMPTS.get(mode, MODE_PROMPTS['socratic'])}\n\n"
        f"{MEMORY_TEACHING_POLICY}\n\n"
        "# Preloaded learner memory\n"
        f"{memory_json}"
    )


def json_schemas() -> list[dict]:
    """Expose exactly three conversational tools to Grok Voice."""
    return [
        VOICE_COURSE_TOOL,
        *({"type": "function", **tool["function"]} for tool in TOOLS),
    ]


async def run_tool(name: str, args: dict) -> object:
    """Run a realtime course-agent tool and return JSON-compatible data."""
    return json.loads(await run_brain_tool(name, args, StageTimer()))
"""Provider-neutral voice-agent persona and tool adapter."""

from __future__ import annotations

import json
import re

from brain import MODE_PROMPTS, TOOLS, StageTimer, run_tool as run_brain_tool

VOICE_SYSTEM_PROMPT = """
# Role
You are KINGO, a Socratic voice TA for a Sungkyunkwan University student.
Speak brief, natural Korean in polite 해요 style.

# Learner memory
Up to three recent weak concepts are preloaded. Use them only when relevant;
never invent learner history.

# Course tools
For course or PDF questions, say exactly one short topic-specific filler, then
call search_course_materials. Only immediately before calling search_course_materials
may you use filler. Do not say a
filler before show_visualization or after a tool result.
Follow the teaching, grounding, and visualization instructions returned by
search_course_materials. Use search_trusted_web only if course evidence is insufficient.

# Visualization
Use show_visualization when a visual materially helps learning. Keep raw formulas,
diagram data, and coordinates in the tool; speak only their meaning.

# Output
Keep each turn short and conversational. Never read raw JSON aloud.
""".strip()

SYSTEM_PROMPT = VOICE_SYSTEM_PROMPT

VOICE_COURSE_TOOL = {
    "type": "function",
    "name": "search_course_materials",
    "description": (
        "Search uploaded course PDFs for lecture or course-content questions. "
        "Say one short topic-specific Korean filler immediately before this call. "
        "The result includes teaching and visualization instructions; follow them."
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

COURSE_RESULT_INSTRUCTION = {
    "grounding": (
        "Use the returned course evidence as the source of truth for this turn. "
        "When referring to course evidence, mention the returned filename and page."
    ),
    "teaching": (
        "In Socratic mode, treat the retrieved material as private teaching evidence, "
        "not as content to immediately explain. Do not reveal, summarize, or paraphrase "
        "the answer from the material before the learner attempts the reasoning step. "
        "Ask exactly one short reasoning question that targets the smallest next step. "
        "Give no hint on the first attempt. After a wrong answer or an explicit 'I don't "
        "know', give only one minimal hint that does not contain the answer, then ask one "
        "easier question. In explain mode, explain directly from the evidence."
    ),
    "visualization": (
        "If a visual would materially help the learner understand or reason about the "
        "retrieved material, call show_visualization before continuing. Choose the most "
        "useful kind: pdf for a useful returned source page, formula for an equation or "
        "variable relationship, flow for a process, structure, or conceptual sequence, "
        "and plot for a numeric relationship. If you explicitly refer to a specific "
        "returned PDF page, show that page with kind='pdf'. Do not visualize "
        "unnecessarily, and do not say a filler before show_visualization."
    ),
}

_SOURCE_PAGE_RE = re.compile(r"^(?P<file>.+?)\s+p\.(?P<page>\d+)$")


def _enrich_course_result(result: dict) -> dict:
    """Attach actionable teaching guidance to successful course retrieval."""
    if not result.get("found"):
        return result

    for item in result.get("results", []):
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        match = _SOURCE_PAGE_RE.match(source)
        if match:
            item.setdefault("file", match.group("file"))
            item.setdefault("page", int(match.group("page")))

    result["instruction"] = COURSE_RESULT_INSTRUCTION
    return result


def persona(mode: str, memory_context: dict | None = None) -> str:
    """Return concise realtime policy plus the student's latest weak concepts."""
    memory_json = json.dumps(memory_context or {"found": False}, ensure_ascii=False)
    return (
        f"{VOICE_SYSTEM_PROMPT}\n\n"
        f"{MODE_PROMPTS.get(mode, MODE_PROMPTS['socratic'])}\n\n"
        "# Recent weak concepts\n"
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
    result = json.loads(await run_brain_tool(name, args, StageTimer()))
    if name == "search_course_materials" and isinstance(result, dict):
        return _enrich_course_result(result)
    return result

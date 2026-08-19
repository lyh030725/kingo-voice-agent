"""Provider-neutral voice-agent persona and tool adapter."""

from __future__ import annotations

import json
import re

from brain import MODE_PROMPTS, TOOLS, StageTimer, run_tool as run_brain_tool

VOICE_SYSTEM_PROMPT = """
# Role
You are KINGO, a Socratic voice TA for a student.
Speak brief, natural Korean in polite 해요 style.

# Memory
Use up to three preloaded weak concepts only when relevant; never invent.

# Tools
Only immediately before calling search_course_materials, search_trusted_web, or
show_visualization, say one short Korean line: search filler for searches, transition
for visualization. The filler before show_visualization must not reveal formula/visual
data; never guess search results. Follow instructions returned by search tools.
Use search_trusted_web only when course evidence is insufficient.

# Visualization
Visualization is part of teaching, not decoration. Use show_visualization proactively
for any helpful formula, process, structure, or relevant PDF page. Default to a visual
before explaining or asking about such content; skip only if it adds no value.
Prefer show_visualization over speech alone. In Socratic mode, use it as a clue, not the
final answer. Keep raw visual data in the tool; speak only its meaning.
All user-visible visualization text MUST be Korean; formulas/standard terms may remain.

# Output
Keep each turn short and conversational. Never read raw JSON aloud.
""".strip()

VOICE_SOCRATIC_VISUAL_RULE = """
# Socratic visual priority
A visual clue is not a spoken explanation. When the current reasoning step has a useful
formula, process, or structure, MUST call show_visualization before the one Socratic
question. The no-answer/worked-example rule applies to speech; keep the visual a clue
and never use it to reveal the final answer.
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

VOICE_VISUALIZATION_TOOL = {
    "type": "function",
    "name": "show_visualization",
    "description": (
        "Proactively show a visual whenever a formula, process, structure, or relevant "
        "course PDF page can help the student reason. In Socratic mode, treat the visual "
        "as the preferred clue before the question, not as a spoken explanation. Prefer "
        "calling this tool over describing such content only with speech. Use formula for "
        "an equation or variable relationship, flow for a process or structure, and pdf "
        "for an exact returned course page."
    ),
    "parameters": {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "const": "formula"},
                    "title": {
                        "type": "string",
                        "description": "Short Korean title.",
                    },
                    "caption": {
                        "type": "string",
                        "description": "One concise Korean takeaway.",
                    },
                    "latex": {
                        "type": "string",
                        "description": "Raw LaTeX for the formula.",
                    },
                },
                "required": ["kind", "title", "caption", "latex"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "const": "flow"},
                    "title": {
                        "type": "string",
                        "description": "Short Korean title.",
                    },
                    "caption": {
                        "type": "string",
                        "description": "One concise Korean takeaway.",
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 8,
                        "description": "Ordered process or structure labels.",
                    },
                },
                "required": ["kind", "title", "caption", "labels"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "const": "pdf"},
                    "title": {
                        "type": "string",
                        "description": "Short Korean title.",
                    },
                    "caption": {
                        "type": "string",
                        "description": "One concise Korean takeaway.",
                    },
                    "file": {
                        "type": "string",
                        "description": "Exact returned course PDF filename.",
                    },
                    "page": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Exact returned course PDF page.",
                    },
                },
                "required": ["kind", "title", "caption", "file", "page"],
                "additionalProperties": False,
            },
        ]
    },
}

COURSE_RESULT_INSTRUCTION = {
    "grounding": (
        "Teach from this course evidence and cite filename/page when relevant."
    ),
    "teaching": (
        "Keep the current teaching mode. In Socratic mode, guide with one next-step "
        "question instead of giving the answer."
    ),
    "visualization": (
        "Prefer show_visualization. If this evidence contains a useful formula, "
        "structure, process, or PDF page, use it before continuing the teaching step. "
        "Use it as a clue, not the final answer."
    ),
}

WEB_RESULT_INSTRUCTION = {
    "grounding": (
        "Use this trusted-web evidence only for claims not supported by course material."
    ),
    "teaching": (
        "Keep the current teaching mode. In Socratic mode, guide with one next-step "
        "question instead of giving the answer."
    ),
    "visualization": (
        "Prefer show_visualization. If this evidence contains a useful formula, "
        "structure, or process, use it before continuing the teaching step."
    ),
    "sources": "Do not read raw URLs aloud.",
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


def _enrich_web_result(result: dict) -> dict:
    """Attach actionable teaching guidance to successful trusted-web retrieval."""
    if not result.get("found"):
        return result
    result["instruction"] = WEB_RESULT_INSTRUCTION
    return result


def persona(mode: str, memory_context: dict | None = None) -> str:
    """Return concise realtime policy plus the student's latest weak concepts."""
    memory_json = json.dumps(memory_context or {"found": False}, ensure_ascii=False)
    mode_prompt = MODE_PROMPTS.get(mode, MODE_PROMPTS["socratic"])
    if mode == "socratic":
        mode_prompt = f"{mode_prompt}\n\n{VOICE_SOCRATIC_VISUAL_RULE}"
    return (
        f"{VOICE_SYSTEM_PROMPT}\n\n"
        f"{mode_prompt}\n\n"
        "# Recent weak concepts\n"
        f"{memory_json}"
    )


def json_schemas() -> list[dict]:
    """Expose exactly three conversational tools to Grok Voice."""
    trusted_web = next(
        {"type": "function", **tool["function"]}
        for tool in TOOLS
        if tool["function"]["name"] == "search_trusted_web"
    )
    return [
        VOICE_COURSE_TOOL,
        trusted_web,
        VOICE_VISUALIZATION_TOOL,
    ]


async def run_tool(name: str, args: dict) -> object:
    """Run a realtime course-agent tool and return JSON-compatible data."""
    result = json.loads(await run_brain_tool(name, args, StageTimer()))
    if name == "search_course_materials" and isinstance(result, dict):
        return _enrich_course_result(result)
    if name == "search_trusted_web" and isinstance(result, dict):
        return _enrich_web_result(result)
    return result

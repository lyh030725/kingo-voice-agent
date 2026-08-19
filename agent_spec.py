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
may you use filler. Do not say a filler before show_visualization or after a tool result.
Follow instructions returned by search tools. Use search_trusted_web only if
course evidence is insufficient.

# Visualization
Visualization is part of teaching, not decoration.
When the current reasoning step is best understood through a formula, process,
structure, or PDF page, MUST call show_visualization before responding.
In Socratic mode, use the visual to support the next reasoning step without
revealing the final answer. Keep raw formulas and visual data in the tool;
speak only their meaning.
All user-visible visualization text, including title, caption, and flow labels,
MUST be Korean. Keep only formulas and standard technical terms in their original form.

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

VOICE_VISUALIZATION_TOOL = {
    "type": "function",
    "name": "show_visualization",
    "description": (
        "Show the visual needed for the current teaching step. Use formula for an "
        "equation or variable relationship, flow for a process or structure, and pdf "
        "for an exact returned course page. Prefer a useful visual clue over verbalizing "
        "raw visual data."
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
        "If the current reasoning step is best understood visually, call "
        "show_visualization before continuing. Use pdf for a useful returned source page, "
        "formula for an equation or variable relationship, and flow for a process, "
        "structure, or conceptual sequence. If you explicitly refer to a specific returned "
        "PDF page, show that page with kind='pdf'. In Socratic mode, visualize a clue for "
        "the next reasoning step rather than the final answer. Do not say a filler before "
        "show_visualization."
    ),
}

WEB_RESULT_INSTRUCTION = {
    "grounding": (
        "Use only the returned trusted-web evidence for factual claims that were not "
        "supported by course material. Do not add unsupported facts from memory."
    ),
    "teaching": (
        "In Socratic mode, treat the web answer as private teaching evidence rather than "
        "a learner-facing final answer. Do not reveal the conclusion before the learner "
        "attempts the reasoning step. Ask exactly one short reasoning question that "
        "targets the smallest next step. After a wrong answer or an explicit 'I don't "
        "know', give only one minimal hint that does not contain the answer, then ask one "
        "easier question. In explain mode, explain directly from the evidence."
    ),
    "visualization": (
        "If the current reasoning step is best understood visually, call "
        "show_visualization before continuing. Use formula for equations or variable "
        "relationships and flow for processes or structures. In Socratic mode, visualize "
        "a clue for the next reasoning step rather than the final answer. Do not say a "
        "filler before show_visualization."
    ),
    "sources": (
        "The UI displays trusted source URLs separately. Do not read or repeat raw URLs "
        "in the spoken answer."
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


def _enrich_web_result(result: dict) -> dict:
    """Attach actionable teaching guidance to successful trusted-web retrieval."""
    if not result.get("found"):
        return result
    result["instruction"] = WEB_RESULT_INSTRUCTION
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

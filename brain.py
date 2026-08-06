"""KINGO VOICE TA brain shared by Week 3 streaming transports.

The single-student agent automatically stores weak concepts, recalls them on
later turns, searches local course PDFs first, and uses trusted web search only
when the course material is insufficient. External answers are accepted only
when source URLs are present and are also written to an audit log.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Awaitable, Callable, Literal
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageFunctionToolCall
from pydantic import BaseModel, Field
from pypdf import PdfReader

from moss_memory import MossMemoryStore

if os.environ.get("VOICE_AI_SKIP_DOTENV") != "1":
    load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dispatcher")

MOSS_MEMORY = MossMemoryStore()



BASE_DIR = Path(__file__).resolve().parent
WEB_SEARCH_LOG = BASE_DIR / "sources" / "web-search.jsonl"
COURSE_SRCS_DIR = BASE_DIR / "srcs"
DEFAULT_TRUSTED_WEB_DOMAINS = (
    "skku.edu",
    "arxiv.org",
    "aclanthology.org",
    "proceedings.neurips.cc",
    "jmlr.org",
)
TRUSTED_SITES_FILE = BASE_DIR / "trusted-sites.json"
MAX_MATERIAL_BYTES = 25 * 1024 * 1024
PDF_MAX_RESULTS = 3
PDF_PAGE_CACHE: list[dict] | None = None
MAX_HISTORY_MESSAGES = 12



def list_course_materials() -> list[dict]:
    """List indexed PDF files.

    Returns:
        Material records containing filename and byte size.
    """
    COURSE_SRCS_DIR.mkdir(parents=True, exist_ok=True)
    return [
        {"name": path.name, "size": path.stat().st_size}
        for path in sorted(COURSE_SRCS_DIR.glob("*.pdf"))
    ]


def add_course_material(filename: str, content: bytes) -> dict:
    """Save one professor-uploaded course PDF and invalidate search cache.

    Args:
        filename: Plain PDF filename without directory components.
        content: Complete PDF file bytes.

    Returns:
        Saved material record containing filename and byte size.
    """
    global PDF_PAGE_CACHE
    if not filename or filename != Path(filename).name or any(char in filename for char in ("\\", "\0")):
        raise ValueError("invalid filename")
    if Path(filename).suffix.casefold() != ".pdf":
        raise ValueError("only PDF course materials are supported")
    if not content.startswith(b"%PDF-"):
        raise ValueError("file is not a valid PDF")
    if len(content) > MAX_MATERIAL_BYTES:
        raise ValueError("course material exceeds 25 MB")
    COURSE_SRCS_DIR.mkdir(parents=True, exist_ok=True)
    path = COURSE_SRCS_DIR / filename
    path.write_bytes(content)
    PDF_PAGE_CACHE = None
    return {"name": path.name, "size": len(content)}


def remove_course_material(filename: str) -> None:
    """Delete one uploaded PDF and invalidate the search cache."""
    global PDF_PAGE_CACHE
    if not filename or filename != Path(filename).name or any(char in filename for char in ("\\", "\0")):
        raise ValueError("invalid filename")
    if Path(filename).suffix.casefold() != ".pdf":
        raise ValueError("only PDF course materials are supported")
    path = COURSE_SRCS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(filename)
    path.unlink()
    PDF_PAGE_CACHE = None


def _trusted_domain(value: str) -> str:
    """Normalize a URL or hostname to a trusted domain.

    Args:
        value: HTTPS URL or bare hostname supplied by professor.

    Returns:
        Lowercase hostname accepted by xAI web-search filters.
    """
    raw = value.strip()
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    domain = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme not in {"http", "https"} or "." not in domain:
        raise ValueError("valid HTTP(S) site is required")
    return domain


def get_trusted_domains() -> list[str]:
    """Load current professor-managed trusted domains.

    Returns:
        Sorted unique domain allowlist.
    """
    if not TRUSTED_SITES_FILE.exists():
        return sorted(DEFAULT_TRUSTED_WEB_DOMAINS)
    try:
        values = json.loads(TRUSTED_SITES_FILE.read_text(encoding="utf-8"))
        return sorted({_trusted_domain(value) for value in values})
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        log.exception("failed to load trusted sites; using defaults")
        return sorted(DEFAULT_TRUSTED_WEB_DOMAINS)


def _save_trusted_domains(domains: list[str]) -> None:
    """Persist trusted domains.

    Args:
        domains: Normalized domain allowlist.

    Returns:
        None.
    """
    TRUSTED_SITES_FILE.write_text(
        json.dumps(sorted(set(domains)), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def add_trusted_domain(value: str) -> list[str]:
    """Add one professor-approved trusted domain.

    Args:
        value: HTTPS URL or bare hostname.

    Returns:
        Updated sorted domain allowlist.
    """
    domains = set(get_trusted_domains())
    domains.add(_trusted_domain(value))
    result = sorted(domains)
    _save_trusted_domains(result)
    return result


def remove_trusted_domain(value: str) -> list[str]:
    """Remove one trusted domain.

    Args:
        value: HTTPS URL or bare hostname.

    Returns:
        Updated sorted domain allowlist.
    """
    domain = _trusted_domain(value)
    result = [item for item in get_trusted_domains() if item != domain]
    _save_trusted_domains(result)
    return result


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in; "
            f"model names and voices live at https://docs.x.ai."
        )
    return value


def xai_client() -> OpenAI:
    return OpenAI(
        base_url=os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1"),
        api_key=require_env("XAI_API_KEY"),
    )


class StageTimer:
    def __init__(self) -> None:
        self.timings_ms: dict[str, int] = {}

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            ms = round((time.perf_counter() - t0) * 1000)
            self.timings_ms[name] = ms
            log.info("stage %-5s %5d ms", name, ms)

    def record(self, name: str, started_at: float) -> None:
        ms = round((time.perf_counter() - started_at) * 1000)
        self.timings_ms[name] = ms
        log.info("stage %-5s %5d ms", name, ms)



class TextQuestion(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    mode: Literal["explain", "socratic"] = "socratic"


class WeakConceptCapture(BaseModel):
    course: str
    concept: str
    original_question: str
    difficulty_note: str


class PlotPoint(BaseModel):
    x: float = Field(ge=-1e12, le=1e12)
    y: float = Field(ge=-1e12, le=1e12)


class Visualization(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    kind: Literal["formula", "flow", "plot"]
    caption: str = Field(min_length=1, max_length=300)
    latex: str = Field(max_length=1000)
    labels: list[str] = Field(max_length=8)
    points: list[PlotPoint] = Field(max_length=40)
    x_label: str = Field(max_length=40)
    y_label: str = Field(max_length=40)




HISTORY: list[dict] = []
# ponytail: one in-process student session; split by authenticated user when auth lands.

SYSTEM_PROMPT = """
# Role and objective
You are KINGO VOICE TA, a Socratic voice teaching assistant for one
Sungkyunkwan University student.

# Language and speaking style
Speak in Korean unless asked otherwise. Use natural spoken Korean in the
polite 해요 style, as if talking with the student face to face. Prefer endings
such as 해요, 예요, 볼까요, and 해볼게요. Avoid written declarative endings such
as 한다, 이다, and 하였다, and avoid textbook or report-like prose unless you are
quoting a source.

# Tool workflow
At the start of every student turn, call recall_weak_concepts and
search_course_materials together.

# Evidence and sources
Base factual claims only on tool results. For PDF evidence, state filename and
page, but paraphrase any equation instead of quoting or reading it. Put the
exact equation in show_visualization. If PDF evidence is missing or
insufficient, call search_trusted_web.
Return source URLs through the separate sources field; do not repeat raw URLs
in the conversational answer.

# Learning memory
Use recalled weaknesses to personalize hints and check prerequisites. Call
save_weak_concept only for explicit confusion, an incorrect answer, or an
incomplete explanation; ordinary questions are not weaknesses. When the
student answers a review prompt containing a memory id, call
review_weak_concept with your correctness judgment.

# Visual references
Never put raw equations, symbolic notation, diagrams, or coordinate data in
the conversational answer and never read them symbol by symbol. When the
answer would otherwise contain a formula, process diagram, or graph, you MUST
call show_visualization first and put the exact visual data only in that tool.
Treat any mathematical expression—including a single equation, variable
relationship, Greek letter, fraction, exponent, subscript, or LaTeX—as a
formula. When unsure whether visual support is useful, prefer calling
show_visualization. Do not send the final conversational answer until that tool
has succeeded. In the spoken answer, explain only what the visual means; never
repeat its LaTeX, symbols, equation, or coordinates, even when quoting a PDF.
Then explain it naturally with a reference such as '제가 보여드린 그림처럼'.

# Response format
Use one to three short conversational sentences with no markdown lists. Never
read raw JSON aloud.
""".strip()

MODE_PROMPTS = {
    "explain": (
        "Explanation mode: explain the concept directly in plain Korean, give "
        "one concrete example, then ask one short understanding-check question."
    ),
    "socratic": (
        "Socratic mode: do not reveal the final answer first. Ask one focused "
        "question or give one progressive hint that makes the student reason."
    ),
}

MAX_TOOL_ROUNDS = 6
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall_weak_concepts",
            "description": (
                "Recall semantically relevant weak concepts for this student. "
                "Call on every student turn before answering."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Current question or concept used for memory retrieval.",
                    },
                },
                "required": ["topic"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_course_materials",
            "description": (
                "Search uploaded course PDFs for grounded evidence. "
                "Call on every student turn before answering."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Focused terms from the student question.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_trusted_web",
            "description": (
                "Search trusted academic or SKKU domains only after course "
                "material search is missing or insufficient."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Focused external search query.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Specific gap in course PDF evidence.",
                    },
                },
                "required": ["query", "reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_weak_concept",
            "description": (
                "Save a weak concept only after confusion, an incorrect answer, "
                "or an incomplete explanation. Do not save ordinary questions."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "course": {
                        "type": "string",
                        "description": "Course containing the weak concept.",
                    },
                    "concept": {
                        "type": "string",
                        "description": "Concise weak-concept label.",
                    },
                    "original_question": {
                        "type": "string",
                        "description": "Student question exposing the weakness.",
                    },
                    "difficulty_note": {
                        "type": "string",
                        "description": "Observed misunderstanding or missing prerequisite.",
                    },
                },
                "required": [
                    "course",
                    "concept",
                    "original_question",
                    "difficulty_note",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "review_weak_concept",
            "description": (
                "Update spaced-repetition state when the student answers a "
                "review question containing a memory id."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Weak-concept memory id from the review prompt.",
                    },
                    "correct": {
                        "type": "boolean",
                        "description": "Whether the student explanation is correct.",
                    },
                },
                "required": ["memory_id", "correct"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_visualization",
            "description": (
                "Show a safe visual reference instead of putting formulas, diagrams, "
                "or graph coordinates in the spoken answer. Prefer calling this even "
                "for one equation or variable relationship, and whenever visual support "
                "might help. Use formula for LaTeX, "
                "flow for ordered labeled steps, or plot for numeric x/y points."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short Korean title."},
                    "kind": {"type": "string", "enum": ["formula", "flow", "plot"]},
                    "caption": {"type": "string", "description": "One concise Korean takeaway."},
                    "latex": {"type": "string", "description": "Raw LaTeX without dollar delimiters; empty unless kind is formula."},
                    "labels": {"type": "array", "items": {"type": "string"}, "description": "Ordered node labels; empty unless kind is flow."},
                    "points": {
                        "type": "array",
                        "description": "Numeric points; empty unless kind is plot.",
                        "items": {
                            "type": "object",
                            "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                            "required": ["x", "y"],
                            "additionalProperties": False,
                        },
                    },
                    "x_label": {"type": "string", "description": "Plot x-axis label, otherwise empty."},
                    "y_label": {"type": "string", "description": "Plot y-axis label, otherwise empty."},
                },
                "required": ["title", "kind", "caption", "latex", "labels", "points", "x_label", "y_label"],
                "additionalProperties": False,
            },
        },
    },
]
# Retrieval and persistence helpers.

# --------------------------------------------------------------------------
# HOMEWORK 2 — the tool implementations. Fast, terse, validated.
# --------------------------------------------------------------------------

def _json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


def _tool_log_value(value: object, limit: int = 600) -> str:
    """Return compact, bounded JSON for tool debugging logs."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    text = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    return text if len(text) <= limit else f"{text[:limit]}…"


def show_visualization(**args) -> str:
    """Validate one formula, flow diagram, or numeric plot for the chat UI.

    Args:
        **args: Structured visualization fields from the model tool call.

    Returns:
        JSON string containing only validated, render-safe data.
    """
    visualization = Visualization(**args)
    if visualization.kind == "formula" and not visualization.latex.strip():
        raise ValueError("formula visualization requires latex")
    if visualization.kind == "flow" and len(visualization.labels) < 2:
        raise ValueError("flow visualization requires at least two labels")
    if visualization.kind == "plot" and len(visualization.points) < 2:
        raise ValueError("plot visualization requires at least two points")
    return _json(visualization.model_dump())


async def save_weak_concept(
    course: str,
    concept: str,
    original_question: str,
    difficulty_note: str,
) -> str:
    """Store one weak concept and queue cloud sync.

    Args:
        course: Course name containing the concept.
        concept: Concise weak-concept label.
        original_question: Question that exposed the weakness.
        difficulty_note: Observed misunderstanding.

    Returns:
        JSON string with memory id and status.
    """
    return _json(
        await MOSS_MEMORY.save(
            course=course,
            concept=concept,
            original_question=original_question,
            difficulty_note=difficulty_note,
        )
    )


def _terms(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[0-9A-Za-z가-힣_]{2,}", text)
        if token.casefold() not in {"대해서", "설명", "질문", "무엇", "어떻게"}
    }


async def recall_weak_concepts(topic: str) -> str:
    """Recall weak concepts relevant to a topic.

    Args:
        topic: Current student question or concept.

    Returns:
        JSON string containing matched memories.
    """
    return _json(await MOSS_MEMORY.recall(topic))


def _pdf_pages() -> list[dict]:
    global PDF_PAGE_CACHE
    if PDF_PAGE_CACHE is not None:
        return PDF_PAGE_CACHE

    pages = []
    for pdf_path in sorted(COURSE_SRCS_DIR.glob("*.pdf")):
        try:
            reader = PdfReader(str(pdf_path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append(
                        {
                            "file": pdf_path.name,
                            "page": page_number,
                            "text": re.sub(r"\s+", " ", text),
                        }
                    )
        except Exception:
            log.exception("failed to index PDF: %s", pdf_path)
    PDF_PAGE_CACHE = pages
    log.info("indexed %d PDF page(s) from %s", len(pages), COURSE_SRCS_DIR)
    return pages


def _excerpt(text: str, terms: set[str], limit: int = 1000) -> str:
    lowered = text.casefold()
    positions = [lowered.find(term) for term in terms if lowered.find(term) >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - 250)
    return text[start : start + limit].strip()


def search_course_materials(query: str) -> str:
    """Search indexed course PDFs by page.

    Args:
        query: Focused terms from student question.

    Returns:
        JSON string with ranked filename, page, and excerpt results.
    """
    query = query.strip()
    terms = _terms(query)
    if not terms:
        return _json({"error": "a focused PDF search query is required"})

    ranked = []
    for page in _pdf_pages():
        lowered = page["text"].casefold()
        score = sum(lowered.count(term) for term in terms)
        if score:
            ranked.append((score, page))
    ranked.sort(key=lambda item: item[0], reverse=True)

    results = [
        {
            "source": f"{page['file']} p.{page['page']}",
            "excerpt": _excerpt(page["text"], terms),
        }
        for _, page in ranked[:PDF_MAX_RESULTS]
    ]
    return _json(
        {
            "found": bool(results),
            "query": query,
            "results": results,
            "instruction": (
                "Use filename and page in the answer."
                if results
                else "No PDF evidence found; trusted web search is now allowed."
            ),
        }
    )


def _trusted_urls(response) -> list[str]:
    payload = response.model_dump() if hasattr(response, "model_dump") else {}
    candidates = set(re.findall(r"https?://[^\s\]\)\"']+", json.dumps(payload)))
    candidates.update(re.findall(r"https?://[^\s\]\)\"']+", response.output_text or ""))

    trusted = []
    for url in sorted(candidates):
        host = urlparse(url.rstrip(".,;")).hostname or ""
        if any(host == domain or host.endswith("." + domain) for domain in get_trusted_domains()):
            trusted.append(url.rstrip(".,;"))
    return trusted


def search_trusted_web(
    query: str,
    pdf_evidence_insufficient: bool = False,
    reason: str = "",
) -> str:
    """Search trusted web domains and retain citation audit data.

    Args:
        query: Focused external search query.
        pdf_evidence_insufficient: Whether course PDF evidence is insufficient.
        reason: Evidence gap permitting web fallback.

    Returns:
        JSON string with cited answer and trusted source URLs.
    """
    query = query.strip()
    reason = reason.strip()
    if not query:
        return _json({"error": "web search query is required"})
    if not reason:
        return _json({"error": "PDF fallback reason is required"})

    response = xai_client().responses.create(
        model=os.environ.get("WEB_SEARCH_MODEL", "grok-4.3"),
        input=[
            {
                "role": "system",
                "content": (
                    "Answer in Korean using only the web-search evidence. "
                    "Keep source URLs in the tool result, not in the prose answer. If evidence is insufficient, say so."
                ),
            },
            {"role": "user", "content": query},
        ],
        tools=[
            {
                "type": "web_search",
                "filters": {"allowed_domains": list(get_trusted_domains())},
            }
        ],
    )
    answer = (response.output_text or "").strip()
    sources = _trusted_urls(response)
    if not answer or not sources:
        return _json(
            {
                "error": "trusted web search returned no citable sources",
                "query": query,
            }
        )

    record = {
        "query": query,
        "answer": answer,
        "sources": sources,
        "pdf_evidence_insufficient": pdf_evidence_insufficient,
        "fallback_reason": reason,
        "searched_at": time.time(),
    }
    WEB_SEARCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with WEB_SEARCH_LOG.open("a", encoding="utf-8") as file:
        file.write(_json(record) + "\n")
    return _json(
        {
            "found": True,
            "answer": answer,
            "sources": sources,
            "instruction": "The UI displays source URLs separately; do not repeat them in the answer.",
        }
    )


# --------------------------------------------------------------------------
async def review_weak_concept(memory_id: str, correct: bool) -> str:
    """Update one weak concept after a review answer.

    Args:
        memory_id: Stored Moss memory identifier.
        correct: Whether the student explanation is correct.

    Returns:
        JSON string containing updated review status.
    """
    return _json(await MOSS_MEMORY.review(memory_id, correct))


async def run_tool(name: str, args: dict, timer: StageTimer) -> str:
    """Dispatch one validated function tool call.

    Args:
        name: Tool name emitted by Grok.
        args: Parsed JSON arguments matching the tool schema.
        timer: Collector receiving tool latency.

    Returns:
        JSON string containing tool output or a recoverable error.
    """
    started_at = time.perf_counter()
    stage = {
        "recall_weak_concepts": "recall",
        "search_course_materials": "pdf",
        "search_trusted_web": "web",
        "save_weak_concept": "save",
        "review_weak_concept": "review",
        "show_visualization": "visual",
    }.get(name, "tool")
    log.info("tool call name=%s args=%s", name, _tool_log_value(args))
    try:
        if name == "recall_weak_concepts":
            result = await recall_weak_concepts(**args)
        elif name == "search_course_materials":
            result = await asyncio.to_thread(search_course_materials, **args)
        elif name == "search_trusted_web":
            result = await asyncio.to_thread(
                search_trusted_web,
                pdf_evidence_insufficient=True,
                **args,
            )
        elif name == "save_weak_concept":
            result = await save_weak_concept(**args)
        elif name == "review_weak_concept":
            result = await review_weak_concept(**args)
        elif name == "show_visualization":
            result = show_visualization(**args)
        else:
            result = _json({"error": f"unknown tool: {name}"})
    except (TypeError, ValueError) as exc:
        result = _json({"error": f"invalid {name} arguments: {exc}"})
    except Exception as exc:
        log.exception("tool %s failed", name)
        result = _json({"error": str(exc)})
    finally:
        timer.record(stage, started_at)
    payload = json.loads(result)
    status = "error" if isinstance(payload, dict) and "error" in payload else "ok"
    elapsed_ms = round((time.perf_counter() - started_at) * 1000)
    log.info(
        "tool result name=%s status=%s elapsed_ms=%d result=%s",
        name,
        status,
        elapsed_ms,
        _tool_log_value(payload),
    )
    return result


# Function-tool response pipeline.
# --------------------------------------------------------------------------

CONFUSION_MARKERS = (
    "모르겠", "모르겠어", "잘 모르", "어려워", "어렵", "헷갈",
    "이해가 안", "이해 안", "이해되지", "감이 안", "막혀", "틀린 것 같",
    "don't know", "do not know", "confused", "difficult", "hard to understand",
)


def _explicit_confusion(transcript: str) -> bool:
    normalized = transcript.casefold()
    return any(marker in normalized for marker in CONFUSION_MARKERS)


def _fallback_weak_concept(transcript: str) -> WeakConceptCapture:
    concise_question = re.sub(r"\s+", " ", transcript).strip()
    return WeakConceptCapture(
        course="미지정 과목",
        concept=concise_question[:160],
        original_question=concise_question,
        difficulty_note="학생이 명시적으로 이해 부족, 혼란 또는 어려움을 표현함",
    )


def _append_history(message: dict) -> None:
    HISTORY.append(message)
    if len(HISTORY) > MAX_HISTORY_MESSAGES:
        del HISTORY[:-MAX_HISTORY_MESSAGES]


async def _stream_completion(client, request: dict, on_token: Callable[[str], Awaitable[None]]):
    """Stream text deltas while reconstructing any function calls."""
    stream = await asyncio.to_thread(client.chat.completions.create, **request, stream=True)
    content: list[str] = []
    calls: dict[int, dict] = {}
    sentinel = object()
    try:
        while (chunk := await asyncio.to_thread(next, stream, sentinel)) is not sentinel:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content.append(delta.content)
                await on_token(delta.content)
            for part in delta.tool_calls or []:
                call = calls.setdefault(part.index, {"id": "", "name": "", "arguments": ""})
                if part.id:
                    call["id"] = part.id
                if part.function:
                    call["name"] += part.function.name or ""
                    call["arguments"] += part.function.arguments or ""
    finally:
        close = getattr(stream, "close", None)
        if close:
            await asyncio.to_thread(close)

    tool_calls = [
        ChatCompletionMessageFunctionToolCall(
            id=call["id"],
            type="function",
            function={"name": call["name"], "arguments": call["arguments"]},
        )
        for _, call in sorted(calls.items())
    ]
    return ChatCompletionMessage(
        role="assistant",
        content="".join(content) or None,
        tool_calls=tool_calls or None,
    )



async def think(
    transcript: str,
    timer: StageTimer,
    mode: str = "socratic",
    on_token: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[str, list[str], list[str], list[dict]]:
    """Generate one grounded Socratic response through function tools.

    Args:
        transcript: Student utterance transcribed to text.
        timer: Collector for tool and model latency.
        mode: Learner-selected explanation or Socratic mode.

    Returns:
        Reply text, tool names, trusted source URLs, and visual references.
    """
    _append_history({"role": "user", "content": transcript})
    client = xai_client()
    tool_messages: list[dict] = []
    tools_used: list[str] = []
    external_sources: list[str] = []
    visualizations: list[dict] = []
    required_context = {"recall_weak_concepts", "search_course_materials"}

    def completion_request(tool_choice: str) -> dict:
        return {
            "model": os.environ.get("CHAT_MODEL", "grok-4.3"),
            "reasoning_effort": os.environ.get("CHAT_REASONING_EFFORT", "none"),
            "max_completion_tokens": int(os.environ.get("CHAT_MAX_TOKENS", "1200")),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": MODE_PROMPTS.get(mode, MODE_PROMPTS["socratic"])},
                *HISTORY,
                *tool_messages,
            ],
            "tools": TOOLS,
            "tool_choice": tool_choice,
            "parallel_tool_calls": True,
        }

    def complete(tool_choice: str):
        """Call Grok with tool schemas and accumulated tool results.

        Args:
            tool_choice: OpenAI tool selection mode for this round.

        Returns:
            OpenAI-compatible assistant message.
        """
        response = client.chat.completions.create(**completion_request(tool_choice))
        return response.choices[0].message

    async def execute(call) -> tuple[str, str]:
        """Validate and dispatch one model tool call.

        Args:
            call: OpenAI-compatible function tool call.

        Returns:
            Tuple containing tool name and serialized result.
        """
        name = call.function.name
        try:
            args = json.loads(call.function.arguments)
            if not isinstance(args, dict):
                raise TypeError("tool arguments must be an object")
            result = await run_tool(name, args, timer)
        except (TypeError, json.JSONDecodeError) as exc:
            result = _json({"error": f"invalid tool arguments: {exc}"})
        return name, result

    for _ in range(MAX_TOOL_ROUNDS):
        tool_choice = "auto" if required_context.issubset(tools_used) else "required"
        started_at = time.perf_counter()
        if on_token:
            msg = await _stream_completion(client, completion_request(tool_choice), on_token)
        else:
            msg = await asyncio.to_thread(complete, tool_choice)
        elapsed_ms = round((time.perf_counter() - started_at) * 1000)
        timer.timings_ms["grok"] = timer.timings_ms.get("grok", 0) + elapsed_ms
        log.info("stage grok  %5d ms", elapsed_ms)

        if not msg.tool_calls:
            reply_text = (msg.content or "").strip() or "답변을 생성하지 못했어요. 다시 질문해 주세요."
            if external_sources:
                reply_text = _for_speech(reply_text)

            if _explicit_confusion(transcript) and "save_weak_concept" not in tools_used:
                memory = _fallback_weak_concept(transcript)
                await run_tool("save_weak_concept", memory.model_dump(), timer)
                tools_used.append("save_weak_concept")

            _append_history({"role": "assistant", "content": reply_text})
            return reply_text, tools_used, external_sources[:3], visualizations

        tool_messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [call.model_dump() for call in msg.tool_calls],
        })
        results = await asyncio.gather(*(execute(call) for call in msg.tool_calls))
        for call, (name, result) in zip(msg.tool_calls, results):
            if name not in tools_used:
                tools_used.append(name)
            if name == "search_trusted_web":
                try:
                    external_sources = json.loads(result).get("sources", [])
                except json.JSONDecodeError:
                    external_sources = []
            elif name == "show_visualization":
                try:
                    visual = json.loads(result)
                    if "error" not in visual:
                        visualizations.append(visual)
                except json.JSONDecodeError:
                    pass
            tool_messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

    raise RuntimeError("Grok exceeded the tool-call round limit")
# Speech and lifecycle helpers used by server.py.
# --------------------------------------------------------------------------

def _for_speech(text: str) -> str:
    """Remove URLs from speech while keeping them in screen text.

    Args:
        text: Answer containing optional source URLs.

    Returns:
        Speech-safe answer text.
    """
    text = re.sub(
        r"\s*외부 출처\s*:?\s*(?:https?://\S+\s*,?\s*)+$", "", text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"https?://\S+", "", text).strip()



def reset_conversation() -> None:
    HISTORY.clear()
    log.info("conversation reset")


async def next_review_prompt() -> dict:
    if not MOSS_MEMORY.is_configured:
        return {"due": False}
    memory = await MOSS_MEMORY.next_review()
    if memory is None:
        return {"due": False}
    question = f"{memory['concept']}을 자신의 말로 설명해 볼까요?"
    _append_history({
        "role": "assistant",
        "content": f"복습 질문 (memory_id={memory['id']}): {question}",
    })
    return {
        "due": True,
        "memory_id": memory["id"],
        "concept": memory["concept"],
        "question": question,
    }


async def startup() -> None:
    tasks = [asyncio.to_thread(_pdf_pages)]
    if MOSS_MEMORY.is_configured:
        tasks.append(MOSS_MEMORY.initialize())
    else:
        log.warning(
            "Moss memory is not configured; set MOSS_PROJECT_ID and MOSS_PROJECT_KEY"
        )
    await asyncio.gather(*tasks)


async def shutdown() -> None:
    await MOSS_MEMORY.close()

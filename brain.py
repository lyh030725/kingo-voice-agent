"""KINGO brain shared by text and realtime transports."""

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
from pdf_retrieval import clear_embedding_cache, hybrid_rank
from session_state import (
    clear_history,
    close_all,
    external_brain_for,
    history_for,
    memory_for,
)

if os.environ.get("VOICE_AI_SKIP_DOTENV") != "1":
    load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dispatcher")

BASE_DIR = Path(__file__).resolve().parent
WEB_SEARCH_LOG = BASE_DIR / "sources" / "web-search.jsonl"
COURSE_SRCS_DIR = BASE_DIR / "srcs"
DEFAULT_TRUSTED_WEB_DOMAINS = (
    "skku.edu",
    "arxiv.org",
    "aclanthology.org",
    "proceedings.neurips.cc",
    "wikipedia.org",
)
TRUSTED_SITES_FILE = BASE_DIR / "trusted-sites.json"
MAX_TRUSTED_WEB_DOMAINS = 5
MAX_MATERIAL_BYTES = 25 * 1024 * 1024
PDF_MAX_RESULTS = 3
PDF_PAGE_CACHE: list[dict] | None = None
MAX_HISTORY_MESSAGES = 12
MAX_TOOL_ROUNDS = 6
MEMORY_TOP_K = 3

MEMORY_GUIDANCE = """
# Learner memory
The server supplies up to three of this student's most recently observed weak
concepts, newest first. Use a memory only when it is relevant to the current
conversation. When relevant, naturally acknowledge the prior difficulty once
and adapt the next question or explanation to it. If the learner asks what they
recently struggled with or studied, answer from these records while making clear
they are weak-concept history, not a complete study log. Never invent learner
history or force an unrelated memory into the conversation.
""".strip()


class StageTimer:
    def __init__(self) -> None:
        self.timings_ms: dict[str, int] = {}

    @contextmanager
    def stage(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, started)

    def record(self, name: str, started_at: float) -> None:
        ms = round((time.perf_counter() - started_at) * 1000)
        self.timings_ms[name] = ms
        log.info("stage %-5s %5d ms", name, ms)


class TextQuestion(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    mode: Literal["explain", "socratic"] = "socratic"


class PlotPoint(BaseModel):
    x: float = Field(ge=-1e12, le=1e12)
    y: float = Field(ge=-1e12, le=1e12)


class Visualization(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    kind: Literal["formula", "flow", "plot", "pdf"]
    caption: str = Field(min_length=1, max_length=300)
    latex: str = Field(default="", max_length=1000)
    labels: list[str] = Field(default_factory=list, max_length=8)
    points: list[PlotPoint] = Field(default_factory=list, max_length=40)
    x_label: str = Field(default="", max_length=40)
    y_label: str = Field(default="", max_length=40)
    file: str = Field(default="", max_length=255)
    page: int = Field(default=0, ge=0)


SYSTEM_PROMPT = """
# Role
You are KINGO VOICE TA, a Socratic voice teaching assistant for one
Sungkyunkwan University student.

# Language and style
Speak in Korean unless asked otherwise. Use natural spoken Korean in the
polite 해요 style, as if talking with the student face to face. Prefer endings
such as 해요, 예요, 볼까요, and 해볼게요. Avoid written declarative endings such
as 한다, 이다, and 하였다, and avoid textbook or report-like prose unless you are
quoting a source.

# Context
Before every answer, the server provides relevant learner-memory context and
course-PDF evidence. Use that preloaded context to personalize hints, check
prerequisites, and ground factual claims. Do not ask for or invoke separate
memory/PDF retrieval tools. If the course evidence is missing or insufficient,
trusted web search is allowed.

# Tool usage
search_trusted_web: Call only when the preloaded course-PDF evidence is missing
or insufficient.

show_visualization: Call before the final answer when it would otherwise
contain a formula, process diagram, graph, or other visual data. Also call it
with kind pdf when the student asks to see a referenced course PDF page. Follow
all visualization rules below.

# Evidence and source rules
Base factual claims only on the preloaded context and tool results. For PDF
evidence, state filename and page, but paraphrase any equation instead of
quoting or reading it. Put the exact equation in show_visualization.
Return source URLs through the separate sources field; do not repeat raw URLs
in the conversational answer.

# Visualization rules
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
For a PDF visualization, use the exact filename and page from the preloaded
course evidence, the student's explicit request, or a prior assistant
reference; never invent a file or page number.

# Output format
Use one to three short conversational sentences with no markdown lists. Never
read raw JSON aloud.
""".strip()

MODE_PROMPTS = {
    "explain": (
        "Explanation mode: explain the concept directly in plain Korean, give "
        "one concrete example, then ask one short understanding-check question."
    ),
    "socratic": """
Socratic mode:
Your goal is to make the student perform the next reasoning step.

- On the first turn about a concept, never explain or summarize the answer.
- Start by identifying the first prerequisite or decision needed.
- Ask exactly one question that the student can answer in one short sentence.
- Do not include the answer, a worked example, or a disguised explanation
  before or after the question.
- If partly correct, acknowledge only the correct part and ask for the next step.
- After the first wrong or "I don't know" response, give one minimal hint,
  then ask an easier question.
- After two unsuccessful attempts at the same step, ask whether the student
  wants another hint or a direct explanation.
- Give a direct explanation only after the student explicitly chooses it.
- End every response with exactly one question.
- Maximum two short spoken sentences.
""",
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_trusted_web",
            "description": (
                "Search professor-approved trusted domains only after the preloaded "
                "course material is missing or insufficient."
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
                        "description": "Specific gap in the course-PDF evidence.",
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
            "name": "show_visualization",
            "description": (
                "Show a safe visual reference instead of putting formulas, diagrams, "
                "or graph coordinates in the spoken answer. Use formula for LaTeX, flow "
                "for ordered labels, plot for numeric points, or pdf for a course page."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short Korean title."},
                    "kind": {
                        "type": "string",
                        "enum": ["formula", "flow", "plot", "pdf"],
                    },
                    "caption": {
                        "type": "string",
                        "description": "One concise Korean takeaway.",
                    },
                    "latex": {
                        "type": "string",
                        "description": "Raw LaTeX; empty unless formula.",
                    },
                    "labels": {"type": "array", "items": {"type": "string"}},
                    "points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                            },
                            "required": ["x", "y"],
                            "additionalProperties": False,
                        },
                    },
                    "x_label": {"type": "string"},
                    "y_label": {"type": "string"},
                    "file": {"type": "string"},
                    "page": {"type": "integer", "minimum": 0},
                },
                "required": [
                    "title",
                    "kind",
                    "caption",
                    "latex",
                    "labels",
                    "points",
                    "x_label",
                    "y_label",
                    "file",
                    "page",
                ],
                "additionalProperties": False,
            },
        },
    },
]


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


# Compatibility aliases for older single-student tests/callers.
MOSS_MEMORY = memory_for("default-student")
HISTORY = history_for("default-student", "default-session")
EXTERNAL_BRAIN = external_brain_for("default-student", xai_client)


def _memory_store(student_id: str) -> MossMemoryStore:
    return MOSS_MEMORY if student_id == "default-student" else memory_for(student_id)


def get_external_brain(student_id: str):
    return (
        EXTERNAL_BRAIN
        if student_id == "default-student"
        else external_brain_for(student_id, xai_client)
    )


def _conversation_history(student_id: str, session_id: str) -> list[dict]:
    if student_id == "default-student" and session_id == "default-session":
        return HISTORY
    return history_for(student_id, session_id)


def list_course_materials() -> list[dict]:
    COURSE_SRCS_DIR.mkdir(parents=True, exist_ok=True)
    return [
        {"name": path.name, "size": path.stat().st_size}
        for path in sorted(COURSE_SRCS_DIR.glob("*.pdf"))
    ]


def get_course_material_path(filename: str) -> Path:
    if not filename or filename != Path(filename).name or any(
        char in filename for char in ("\\", "\0")
    ):
        raise ValueError("invalid filename")
    if Path(filename).suffix.casefold() != ".pdf":
        raise ValueError("only PDF course materials are supported")
    path = COURSE_SRCS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(filename)
    return path


def add_course_material(filename: str, content: bytes) -> dict:
    global PDF_PAGE_CACHE
    if not filename or filename != Path(filename).name or any(
        char in filename for char in ("\\", "\0")
    ):
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
    clear_embedding_cache()
    return {"name": path.name, "size": len(content)}


def remove_course_material(filename: str) -> None:
    global PDF_PAGE_CACHE
    get_course_material_path(filename).unlink()
    PDF_PAGE_CACHE = None
    clear_embedding_cache()


def _trusted_domain(value: str) -> str:
    raw = value.strip()
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    domain = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme not in {"http", "https"} or "." not in domain:
        raise ValueError("valid HTTP(S) site is required")
    return domain


def _trusted_domains(values: list[str] | tuple[str, ...]) -> list[str]:
    domains = sorted({_trusted_domain(value) for value in values})
    if len(domains) > MAX_TRUSTED_WEB_DOMAINS:
        raise ValueError(f"at most {MAX_TRUSTED_WEB_DOMAINS} trusted sites are allowed")
    return domains


def get_trusted_domains() -> list[str]:
    if not TRUSTED_SITES_FILE.exists():
        return _trusted_domains(DEFAULT_TRUSTED_WEB_DOMAINS)
    try:
        values = json.loads(TRUSTED_SITES_FILE.read_text(encoding="utf-8"))
        return _trusted_domains(values)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        log.exception("failed to load trusted sites; using defaults")
        return _trusted_domains(DEFAULT_TRUSTED_WEB_DOMAINS)


def _save_trusted_domains(domains: list[str]) -> None:
    TRUSTED_SITES_FILE.write_text(
        json.dumps(_trusted_domains(domains), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def add_trusted_domain(value: str) -> list[str]:
    domains = set(get_trusted_domains())
    domain = _trusted_domain(value)
    if domain not in domains and len(domains) >= MAX_TRUSTED_WEB_DOMAINS:
        raise ValueError(f"at most {MAX_TRUSTED_WEB_DOMAINS} trusted sites are allowed")
    domains.add(domain)
    result = sorted(domains)
    _save_trusted_domains(result)
    return result


def remove_trusted_domain(value: str) -> list[str]:
    domain = _trusted_domain(value)
    result = [item for item in get_trusted_domains() if item != domain]
    _save_trusted_domains(result)
    return result


def _json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


def _tool_log_value(value: object, limit: int = 600) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    text = json.dumps(
        value, ensure_ascii=False, default=str, separators=(",", ":")
    )
    return text if len(text) <= limit else f"{text[:limit]}…"


def show_visualization(**args) -> str:
    """Validate a visual reference.

    Args:
        **args: Structured visualization fields from the model tool call.

    Returns:
        JSON containing validated render-safe visual data.
    """
    visualization = Visualization(**args)
    if visualization.kind == "formula" and not visualization.latex.strip():
        raise ValueError("formula visualization requires latex")
    if visualization.kind == "flow" and len(visualization.labels) < 2:
        raise ValueError("flow visualization requires at least two labels")
    if visualization.kind == "plot" and len(visualization.points) < 2:
        raise ValueError("plot visualization requires at least two points")
    if visualization.kind == "pdf":
        try:
            pdf_path = get_course_material_path(visualization.file)
            page_count = len(PdfReader(str(pdf_path)).pages)
        except FileNotFoundError as exc:
            raise ValueError("PDF course material not found") from exc
        except Exception as exc:
            raise ValueError("PDF course material could not be read") from exc
        if visualization.page < 1 or visualization.page > page_count:
            raise ValueError(f"PDF page must be between 1 and {page_count}")
    return _json(visualization.model_dump())


def _terms(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[0-9A-Za-z가-힣_]{2,}", text)
        if token.casefold() not in {"대해서", "설명", "질문", "무엇", "어떻게"}
    }


def _compact_memory(memory: dict) -> dict:
    record = {
        "concept": memory.get("concept", ""),
        "difficulty": memory.get("difficulty_note") or memory.get("difficulty", ""),
        "status": memory.get("status", "new"),
    }
    course = memory.get("course", "")
    if course:
        record["course"] = course
    return record


async def recent_weak_concepts(
    student_id: str = "default-student",
    *,
    top_k: int = MEMORY_TOP_K,
) -> dict:
    """Return the learner's most recently observed weak concepts."""
    memories = await _memory_store(student_id).all_memories()
    memories.sort(
        key=lambda item: (
            float(item.get("last_seen_at", 0) or 0),
            float(item.get("saved_at", 0) or 0),
        ),
        reverse=True,
    )
    compact = [_compact_memory(memory) for memory in memories[:top_k]]
    compact = [memory for memory in compact if memory["concept"]]
    return {"found": bool(compact), "memories": compact}


async def bootstrap_memory_context(student_id: str = "default-student") -> dict:
    """Return the three most recent compact weak concepts for realtime startup."""
    return await recent_weak_concepts(student_id, top_k=MEMORY_TOP_K)


async def recall_weak_concepts(
    topic: str = "",
    student_id: str = "default-student",
) -> str:
    """Return recent learner weak concepts for server-side prefetch.

    Args:
        topic: Kept for compatibility; recent-memory selection does not depend on it.
        student_id: Learner identity used to isolate memory.

    Returns:
        JSON with the three most recently observed compact weak-concept records.
    """
    del topic
    return _json(await recent_weak_concepts(student_id, top_k=MEMORY_TOP_K))


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
    """Hybrid-search indexed course PDFs.

    Args:
        query: Focused terms from the student question.

    Returns:
        JSON with ranked filename, page, excerpt, and retrieval mode.
    """
    query = query.strip()
    if not query:
        return _json({"error": "a focused PDF search query is required"})
    terms = _terms(query)
    pages = _pdf_pages()
    lexical_scores = []
    for page in pages:
        lowered = page["text"].casefold()
        lexical_scores.append(float(sum(lowered.count(term) for term in terms)))
    ranked, retrieval_mode = hybrid_rank(query, pages, lexical_scores, xai_client)
    results = [
        {
            "source": f"{page['file']} p.{page['page']}",
            "excerpt": _excerpt(page["text"], terms),
            "score": round(float(score), 4),
        }
        for score, page in ranked[:PDF_MAX_RESULTS]
    ]
    return _json(
        {
            "found": bool(results),
            "query": query,
            "retrieval_mode": retrieval_mode,
            "results": results,
            "instruction": (
                "Use filename and page in the answer."
                if results
                else "No PDF evidence found; trusted web search is now allowed."
            ),
        }
    )


async def prefetch_memory_context(question: str, student_id: str) -> dict:
    """Prepare only the three most recent compact learner memories for voice."""
    topic = question.strip()
    try:
        raw = await recall_weak_concepts(topic, student_id)
        memory = json.loads(raw)
    except Exception as exc:
        memory = {"error": f"memory prefetch failed: {exc}"}
    return {"student_question": topic, "weak_concepts": memory}


async def prefetch_context(
    question: str,
    timer: StageTimer | None = None,
    student_id: str = "default-student",
) -> dict:
    """Prefetch learner memory and course-PDF evidence in parallel.

    Args:
        question: Current student utterance or typed question.
        timer: Optional latency collector.
        student_id: Learner identity used to isolate memory.

    Returns:
        JSON-compatible learner-memory and course-material context.
    """
    topic = question.strip()

    async def recall() -> object:
        started = time.perf_counter()
        try:
            if student_id == "default-student":
                return await recall_weak_concepts(topic)
            return await recall_weak_concepts(topic, student_id)
        finally:
            if timer is not None:
                timer.record("recall", started)

    async def search_pdf() -> object:
        started = time.perf_counter()
        try:
            return await asyncio.to_thread(search_course_materials, topic)
        finally:
            if timer is not None:
                timer.record("pdf", started)

    memory_raw, pdf_raw = await asyncio.gather(
        recall(), search_pdf(), return_exceptions=True
    )

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


def answer_instructions(mode: str, context: dict) -> str:
    """Build text-answer instructions with compact learner memory.

    Args:
        mode: Learner-selected explanation or Socratic mode.
        context: Server-prefetched memory and course-PDF evidence.

    Returns:
        System instructions containing shared policy and preloaded context.
    """
    prefetched = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{MODE_PROMPTS.get(mode, MODE_PROMPTS['socratic'])}\n\n"
        f"{MEMORY_GUIDANCE}\n\n"
        "# Preloaded context\n"
        "Use search_trusted_web only when course_materials is missing, reports an error, "
        "or is insufficient for the requested factual claim.\n"
        f"{prefetched}"
    )


def _trusted_urls(response) -> list[str]:
    payload = response.model_dump() if hasattr(response, "model_dump") else {}
    candidates = set(re.findall(r"https?://[^\s\]\)\"']+", json.dumps(payload)))
    candidates.update(
        re.findall(r"https?://[^\s\]\)\"']+", response.output_text or "")
    )
    trusted = []
    for url in sorted(candidates):
        host = urlparse(url.rstrip(".,;")).hostname or ""
        if any(
            host == domain or host.endswith("." + domain)
            for domain in get_trusted_domains()
        ):
            trusted.append(url.rstrip(".,;"))
    return trusted


def search_trusted_web(
    query: str,
    pdf_evidence_insufficient: bool = False,
    reason: str = "",
) -> str:
    """Search professor-approved web domains.

    Args:
        query: Focused external search query.
        pdf_evidence_insufficient: Whether course evidence was insufficient.
        reason: Evidence gap permitting web fallback.

    Returns:
        JSON with answer and trusted source URLs.
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
                    "Keep source URLs in the tool result, not in the prose answer. "
                    "If evidence is insufficient, say so."
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
            {"error": "trusted web search returned no citable sources", "query": query}
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
            "instruction": (
                "The UI displays source URLs separately; do not repeat them in the answer."
            ),
        }
    )


async def run_tool(name: str, args: dict, timer: StageTimer) -> str:
    """Dispatch one model-selected conversational tool.

    Args:
        name: Tool name emitted by Grok.
        args: Parsed JSON arguments.
        timer: Latency collector.

    Returns:
        JSON tool output or a recoverable error.
    """
    started_at = time.perf_counter()
    stage = {
        "search_course_materials": "pdf",
        "search_trusted_web": "web",
        "show_visualization": "visual",
    }.get(name, "tool")
    log.info("tool call name=%s args=%s", name, _tool_log_value(args))
    try:
        if name == "search_course_materials":
            result = await asyncio.to_thread(search_course_materials, **args)
        elif name == "search_trusted_web":
            result = await asyncio.to_thread(
                search_trusted_web,
                pdf_evidence_insufficient=True,
                **args,
            )
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


def _append_history(history: list[dict], message: dict) -> None:
    history.append(message)
    if len(history) > MAX_HISTORY_MESSAGES:
        del history[:-MAX_HISTORY_MESSAGES]


async def _stream_completion(
    client,
    request: dict,
    on_token: Callable[[str], Awaitable[None]],
):
    stream = await asyncio.to_thread(
        client.chat.completions.create, **request, stream=True
    )
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
                call = calls.setdefault(
                    part.index, {"id": "", "name": "", "arguments": ""}
                )
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
    *,
    student_id: str = "default-student",
    session_id: str = "default-session",
) -> tuple[str, list[str], list[str], list[dict]]:
    """Generate one grounded response with isolated learner state."""
    history = _conversation_history(student_id, session_id)
    _append_history(history, {"role": "user", "content": transcript})
    context = await prefetch_context(transcript, timer, student_id)
    client = xai_client()
    tool_messages: list[dict] = []
    tools_used: list[str] = []
    external_sources: list[str] = []
    visualizations: list[dict] = []

    def completion_request() -> dict:
        return {
            "model": os.environ.get("CHAT_MODEL", "grok-4.3"),
            "reasoning_effort": os.environ.get("CHAT_REASONING_EFFORT", "none"),
            "max_completion_tokens": int(
                os.environ.get("CHAT_MAX_TOKENS", "1200")
            ),
            "messages": [
                {"role": "system", "content": answer_instructions(mode, context)},
                *history,
                *tool_messages,
            ],
            "tools": TOOLS,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }

    def complete():
        response = client.chat.completions.create(**completion_request())
        return response.choices[0].message

    async def execute(call) -> tuple[str, str]:
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
        started = time.perf_counter()
        if on_token:
            msg: ChatCompletionMessage = await _stream_completion(
                client, completion_request(), on_token
            )
        else:
            msg = await asyncio.to_thread(complete)
        elapsed = round((time.perf_counter() - started) * 1000)
        timer.timings_ms["grok"] = timer.timings_ms.get("grok", 0) + elapsed
        log.info("stage grok  %5d ms", elapsed)

        if not msg.tool_calls:
            reply = (
                (msg.content or "").strip()
                or "답변을 생성하지 못했어요. 다시 질문해 주세요."
            )
            if external_sources:
                reply = _for_speech(reply)
            _append_history(history, {"role": "assistant", "content": reply})
            get_external_brain(student_id).schedule(history, source="text")
            return reply, tools_used, external_sources[:3], visualizations

        tool_messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [call.model_dump() for call in msg.tool_calls],
            }
        )
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
            tool_messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )

    raise RuntimeError("Grok exceeded the tool-call round limit")


def _for_speech(text: str) -> str:
    text = re.sub(
        r"\s*외부 출처\s*:?\s*(?:https?://\S+\s*,?\s*)+$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"https?://\S+", "", text).strip()


def reset_conversation(
    student_id: str = "default-student",
    session_id: str = "default-session",
) -> None:
    if student_id == "default-student" and session_id == "default-session":
        HISTORY.clear()
    else:
        clear_history(student_id, session_id)
    log.info("conversation reset student=%s session=%s", student_id, session_id)


async def next_review_prompt(
    student_id: str = "default-student",
    session_id: str = "default-session",
) -> dict:
    memory = await _memory_store(student_id).next_review()
    if memory is None:
        return {"due": False}
    question = f"{memory['concept']}을 자신의 말로 설명해 볼까요?"
    _append_history(
        _conversation_history(student_id, session_id),
        {
            "role": "assistant",
            "content": f"복습 질문 (memory_id={memory['id']}): {question}",
        },
    )
    return {
        "due": True,
        "memory_id": memory["id"],
        "concept": memory["concept"],
        "question": question,
    }


async def list_weak_concepts(student_id: str = "default-student") -> list[dict]:
    memories = await _memory_store(student_id).all_memories()
    return [
        {
            "memory_id": memory.get("id", ""),
            "course": memory.get("course", ""),
            "concept": memory.get("concept", ""),
            "difficulty_note": memory.get("difficulty_note", ""),
            "status": memory.get("status", "new"),
            "mastery_percent": round(
                min(max(float(memory.get("confidence", 0)), 0), 1) * 100
            ),
            "success_count": int(memory.get("success_count", 0)),
            "failure_count": int(memory.get("failure_count", 0)),
            "last_seen_at": float(memory.get("last_seen_at", 0)),
        }
        for memory in sorted(
            memories,
            key=lambda item: float(item.get("last_seen_at", 0)),
            reverse=True,
        )
    ]


async def startup() -> None:
    tasks = [asyncio.to_thread(_pdf_pages)]
    if MOSS_MEMORY.is_configured:
        tasks.append(MOSS_MEMORY.initialize())
    else:
        log.warning("Moss memory is not configured; local learner memory will be used")
    await asyncio.gather(*tasks)


async def shutdown() -> None:
    await close_all()

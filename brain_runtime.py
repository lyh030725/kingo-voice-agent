"""Session-aware KINGO brain built on the existing brain primitives."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Awaitable, Callable

import brain as legacy
from openai.types.chat import ChatCompletionMessage

from pdf_retrieval import clear_embedding_cache, hybrid_rank
from session_state import clear_history, close_all, external_brain_for, history_for, memory_for

log = logging.getLogger("brain-runtime")

StageTimer = legacy.StageTimer
TextQuestion = legacy.TextQuestion
SYSTEM_PROMPT = legacy.SYSTEM_PROMPT
MODE_PROMPTS = legacy.MODE_PROMPTS
TOOLS = legacy.TOOLS
MAX_HISTORY_MESSAGES = legacy.MAX_HISTORY_MESSAGES
MEMORY_BOOTSTRAP_LIMIT = int(os.environ.get("MEMORY_BOOTSTRAP_LIMIT", "5"))

MEMORY_TEACHING_POLICY = """
# Learner-memory teaching policy
Learner memory is a real record of this student's prior difficulty, not generic
background. Use it only when relevant, but make relevant personalization visible.

- `recall_type=recent` means the learner is asking about prior/recent learning.
  Answer from the memory records directly. Briefly say what concept(s) were
  recently difficult and what the recorded difficulty was. Phrase this as
  "기록을 보면" or "지난번에는" so you do not pretend the weak-concept log is a
  complete study-history log. Do not search the course PDF just to answer what
  the learner previously struggled with.
- `recall_type=semantic` means the records were retrieved for the current topic.
  If an unresolved memory is the same concept as the current question or a
  necessary prerequisite, explicitly connect it to the current turn once, for
  example "전에 이 부분에서 조금 막혔었어요." Do not expose memory IDs or raw
  metadata.
- In Socratic mode, when such an unresolved memory is directly relevant, use
  retrieval practice before a new explanation: ask exactly one short question
  targeted at the recorded `difficulty_note`. Do not reveal the answer in that
  question. If the student answers correctly, continue to the current topic; if
  they are wrong or say they do not know, give the normal minimal hint.
- In explanation mode, do not force a quiz first. Briefly acknowledge the prior
  difficulty and tailor the explanation/example to that exact weak point, then
  ask the normal understanding-check question.
- `recall_type=bootstrap` is recent weak-concept context loaded when the realtime
  session starts. Treat it as background until the current turn clearly matches
  one of those concepts or asks about recent learning; then it may be used the
  same way as semantic/recent memory.
- Do not force review for memories that are merely related but not prerequisites,
  or for memories with status `mastered`. Do not repeat the same memory
  acknowledgement over and over in one conversation.
- If no memory is found, never invent a past difficulty.
""".strip()

_for_speech = legacy._for_speech
add_trusted_domain = legacy.add_trusted_domain
get_course_material_path = legacy.get_course_material_path
get_trusted_domains = legacy.get_trusted_domains
list_course_materials = legacy.list_course_materials
remove_trusted_domain = legacy.remove_trusted_domain
require_env = legacy.require_env
show_visualization = legacy.show_visualization
search_trusted_web = legacy.search_trusted_web
xai_client = legacy.xai_client


def add_course_material(filename: str, content: bytes) -> dict:
    result = legacy.add_course_material(filename, content)
    clear_embedding_cache()
    return result


def remove_course_material(filename: str) -> None:
    legacy.remove_course_material(filename)
    clear_embedding_cache()


def _is_recent_memory_query(topic: str) -> bool:
    """Detect generic questions about the learner's recent/past learning."""
    normalized = " ".join(topic.casefold().split())
    markers = (
        "저번에 공부", "지난번에 공부", "전에 공부", "최근에 공부", "뭐 공부했",
        "뭘 공부했", "무엇을 공부", "어떤 걸 공부", "어떤 거 공부", "마지막으로 공부",
        "저번에 뭐", "지난번에 뭐", "최근에 뭐", "전에 뭐 어려", "저번에 어려",
        "지난번에 어려", "최근에 어려", "내 취약 개념", "내가 취약", "내가 어려워",
        "기억하고 있는 취약", "last time", "recently studied", "previously studied",
        "what did i study", "what was i weak at",
    )
    return any(marker in normalized for marker in markers)


def _memory_record(memory: dict) -> dict:
    """Return only learner-memory fields useful to the teaching model."""
    return {
        "memory_id": memory.get("id") or memory.get("memory_id", ""),
        "course": memory.get("course", ""),
        "concept": memory.get("concept", ""),
        "original_question": memory.get("original_question", ""),
        "difficulty_note": memory.get("difficulty_note", ""),
        "status": memory.get("status", "new"),
        "confidence": float(memory.get("confidence", 0) or 0),
        "failure_count": int(memory.get("failure_count", 0) or 0),
        "success_count": int(memory.get("success_count", 0) or 0),
        "saved_at": float(memory.get("saved_at", 0) or 0),
        "last_seen_at": float(memory.get("last_seen_at", 0) or 0),
        "next_review_at": float(memory.get("next_review_at", 0) or 0),
    }


def _memory_response(topic: str, memories: list[dict], recall_type: str, storage: str) -> dict:
    records = [_memory_record(memory) for memory in memories]
    return {
        "found": bool(records),
        "topic": topic,
        "storage": storage,
        "recall_type": recall_type,
        "memories": records,
    }


async def recent_weak_concepts(
    student_id: str = "default-student",
    *,
    top_k: int = MEMORY_BOOTSTRAP_LIMIT,
    topic: str = "recent learner memory",
    recall_type: str = "recent",
) -> dict:
    """Return the learner's most recently seen weak concepts."""
    store = memory_for(student_id)
    memories = await store.all_memories()
    memories.sort(
        key=lambda item: (
            float(item.get("last_seen_at", 0) or 0),
            float(item.get("saved_at", 0) or 0),
        ),
        reverse=True,
    )
    return _memory_response(topic, memories[:top_k], recall_type, "recent")


async def bootstrap_memory_context(student_id: str = "default-student") -> dict:
    """Load a small recent-memory snapshot before a realtime session starts."""
    return await recent_weak_concepts(
        student_id,
        top_k=MEMORY_BOOTSTRAP_LIMIT,
        topic="realtime session bootstrap",
        recall_type="bootstrap",
    )


async def recall_weak_concepts(topic: str, student_id: str = "default-student") -> str:
    """Recall relevant or recent learner memory without exposing a model tool."""
    topic = topic.strip()
    if _is_recent_memory_query(topic):
        result = await recent_weak_concepts(student_id, topic=topic, recall_type="recent")
        return json.dumps(result, ensure_ascii=False)

    result = await memory_for(student_id).recall(topic)
    if isinstance(result, dict):
        result = dict(result)
        result["recall_type"] = "semantic"
    return json.dumps(result, ensure_ascii=False)


def search_course_materials(query: str) -> str:
    """Hybrid lexical + semantic retrieval over professor-uploaded PDF pages."""
    query = query.strip()
    terms = legacy._terms(query)
    if not query:
        return json.dumps({"error": "a focused PDF search query is required"}, ensure_ascii=False)

    pages = legacy._pdf_pages()
    lexical_scores = []
    for page in pages:
        lowered = page["text"].casefold()
        lexical_scores.append(float(sum(lowered.count(term) for term in terms)))

    ranked, retrieval_mode = hybrid_rank(query, pages, lexical_scores, xai_client)
    results = [
        {
            "source": f"{page['file']} p.{page['page']}",
            "excerpt": legacy._excerpt(page["text"], terms),
            "score": round(float(score), 4),
        }
        for score, page in ranked[: legacy.PDF_MAX_RESULTS]
    ]
    return json.dumps(
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
        },
        ensure_ascii=False,
    )


async def prefetch_memory_context(question: str, student_id: str) -> dict:
    """Prepare only learner memory for realtime voice; PDF remains a filler-backed tool."""
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
    """Text path: prefetch learner memory and course PDF evidence in parallel."""
    topic = question.strip()

    async def recall() -> object:
        started = time.perf_counter()
        try:
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

    memory_raw, pdf_raw = await asyncio.gather(recall(), search_pdf(), return_exceptions=True)

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


async def run_tool(name: str, args: dict, timer: StageTimer) -> str:
    """Dispatch only conversational tools; learner memory is infrastructure."""
    started = time.perf_counter()
    stage = {
        "search_course_materials": "pdf",
        "search_trusted_web": "web",
        "show_visualization": "visual",
    }.get(name, "tool")
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
            result = json.dumps({"error": f"unknown tool: {name}"}, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        result = json.dumps({"error": f"invalid {name} arguments: {exc}"}, ensure_ascii=False)
    except Exception as exc:
        log.exception("tool %s failed", name)
        result = json.dumps({"error": str(exc)}, ensure_ascii=False)
    finally:
        timer.record(stage, started)
    return result


def _append_history(history: list[dict], message: dict) -> None:
    history.append(message)
    if len(history) > MAX_HISTORY_MESSAGES:
        del history[:-MAX_HISTORY_MESSAGES]


async def think(
    transcript: str,
    timer: StageTimer,
    mode: str = "socratic",
    on_token: Callable[[str], Awaitable[None]] | None = None,
    *,
    student_id: str = "default-student",
    session_id: str = "default-session",
) -> tuple[str, list[str], list[str], list[dict]]:
    """Generate one text response with isolated history and learner memory."""
    history = history_for(student_id, session_id)
    _append_history(history, {"role": "user", "content": transcript})
    context = await prefetch_context(transcript, timer, student_id)
    client = xai_client()
    tool_messages: list[dict] = []
    tools_used: list[str] = []
    external_sources: list[str] = []
    visualizations: list[dict] = []

    def completion_request() -> dict:
        instructions = f"{legacy.answer_instructions(mode, context)}\n\n{MEMORY_TEACHING_POLICY}"
        return {
            "model": os.environ.get("CHAT_MODEL", "grok-4.3"),
            "reasoning_effort": os.environ.get("CHAT_REASONING_EFFORT", "none"),
            "max_completion_tokens": int(os.environ.get("CHAT_MAX_TOKENS", "1200")),
            "messages": [
                {"role": "system", "content": instructions},
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
            result = json.dumps({"error": f"invalid tool arguments: {exc}"}, ensure_ascii=False)
        return name, result

    for _ in range(legacy.MAX_TOOL_ROUNDS):
        started = time.perf_counter()
        if on_token:
            msg: ChatCompletionMessage = await legacy._stream_completion(
                client, completion_request(), on_token
            )
        else:
            msg = await asyncio.to_thread(complete)
        elapsed = round((time.perf_counter() - started) * 1000)
        timer.timings_ms["grok"] = timer.timings_ms.get("grok", 0) + elapsed

        if not msg.tool_calls:
            reply = (msg.content or "").strip() or "답변을 생성하지 못했어요. 다시 질문해 주세요."
            if external_sources:
                reply = _for_speech(reply)
            _append_history(history, {"role": "assistant", "content": reply})
            external_brain_for(student_id, xai_client).schedule(history, source="text")
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


def reset_conversation(student_id: str = "default-student", session_id: str = "default-session") -> None:
    clear_history(student_id, session_id)
    log.info("conversation reset student=%s session=%s", student_id, session_id)


async def next_review_prompt(
    student_id: str = "default-student",
    session_id: str = "default-session",
) -> dict:
    memory_store = memory_for(student_id)
    memory = await memory_store.next_review()
    if memory is None:
        return {"due": False}
    question = f"{memory['concept']}을 자신의 말로 설명해 볼까요?"
    _append_history(
        history_for(student_id, session_id),
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
    memories = await memory_for(student_id).all_memories()
    return [
        {
            "memory_id": memory.get("id", ""),
            "course": memory.get("course", ""),
            "concept": memory.get("concept", ""),
            "difficulty_note": memory.get("difficulty_note", ""),
            "status": memory.get("status", "new"),
            "mastery_percent": round(min(max(float(memory.get("confidence", 0)), 0), 1) * 100),
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


def get_external_brain(student_id: str):
    return external_brain_for(student_id, xai_client)


async def startup() -> None:
    await asyncio.to_thread(legacy._pdf_pages)


async def shutdown() -> None:
    await close_all()
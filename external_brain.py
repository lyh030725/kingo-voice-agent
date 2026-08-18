"""Background weak-concept assessment by a text-based Grok model."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from itertools import count
from collections.abc import Callable
from typing import Any

from moss_memory import MossMemoryStore

log = logging.getLogger("external-brain")
_ASSESSMENT_IDS = count(1)
LOG_PREVIEW_LIMIT = 180

SYSTEM_PROMPT = """
당신은 음성 튜터와 분리된 학습 진단 모델입니다. 매 턴이 끝난 뒤 최근 전체
대화 맥락과 저장된 취약 개념을 보고 아래 두 작업만 수행합니다.

1. 학생이 명백히 틀리거나, 혼동하거나, 불완전하게 이해한 경우에만 새 취약
   개념을 제안합니다. 단순 질문이나 처음 배우는 주제는 취약점이 아닙니다.
2. 저장된 개념과 의미상 같은 취약점은 표현이 달라도 다시 저장하지 않습니다.
3. concept와 difficulty_note는 짧고 구체적인 한국어로 작성합니다.
4. 현재 대화가 저장된 취약 개념의 주제를 실제로 다루고, 학생의 답변에서
   이해 또는 오해의 증거가 드러난 경우에만 reviews에 평가를 넣습니다.
   단순히 튜터 설명을 들었거나 관련 없는 주제라면 업데이트하지 않습니다.
5. 추측하지 말고 JSON만 반환합니다.

반환 형식:
{
  "save": null 또는 {
    "concept": "한국어 주제",
    "original_question": "취약점이 드러난 학생 발화",
    "difficulty_note": "한국어 설명"
  },
  "reviews": [
    {"memory_id": "기존 ID", "correct": true 또는 false}
  ]
}
""".strip()


class ExternalBrain:
    """Schedule and apply one weak-concept assessment per completed turn."""

    def __init__(
        self,
        memory: MossMemoryStore,
        client_factory: Callable[[], Any],
    ) -> None:
        self.memory = memory
        self.client_factory = client_factory
        self._tasks: set[asyncio.Task[None]] = set()
        self._lock = asyncio.Lock()

    def schedule(
        self,
        conversation: list[dict[str, str]],
        *,
        source: str = "unknown",
    ) -> asyncio.Task[None] | None:
        """Start assessment without delaying the user-facing response."""
        if not conversation:
            log.warning("external brain skipped source=%s reason=empty_conversation", source)
            return None
        snapshot = [dict(message) for message in conversation]
        assessment_id = next(_ASSESSMENT_IDS)
        log.info(
            "external brain scheduled id=%s source=%s messages=%s user=%s assistant=%s",
            assessment_id,
            source,
            len(snapshot),
            _last_preview(snapshot, "user"),
            _last_preview(snapshot, "assistant"),
        )
        task = asyncio.create_task(self.assess(snapshot, assessment_id=assessment_id, source=source))
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            log.exception("background weak-concept assessment failed")

    async def assess(
        self,
        conversation: list[dict[str, str]],
        *,
        assessment_id: int = 0,
        source: str = "test",
    ) -> None:
        """Ask text Grok to assess context, then persist validated decisions."""
        started_at = time.perf_counter()
        log.info(
            "external brain started id=%s source=%s messages=%s",
            assessment_id,
            source,
            len(conversation),
        )
        async with self._lock:
            memories = await self.memory.all_memories()
            log.info(
                "external brain context ready id=%s stored_concepts=%s",
                assessment_id,
                len(memories),
            )
            decision = await asyncio.to_thread(self._decide, conversation, memories)
            save = decision.get("save")
            reviews = decision.get("reviews", [])
            if not isinstance(reviews, list):
                reviews = []
                decision["reviews"] = reviews
            log.info(
                "external brain decided id=%s save=%s concept=%s reviews=%s",
                assessment_id,
                isinstance(save, dict),
                _preview(str(save.get("concept", ""))) if isinstance(save, dict) else "-",
                len(reviews),
            )
            saved, reviewed = await self._apply(decision, memories)
        elapsed_ms = round((time.perf_counter() - started_at) * 1000)
        log.info(
            "external brain completed id=%s source=%s saved=%s reviewed=%s elapsed_ms=%s",
            assessment_id,
            source,
            saved,
            reviewed,
            elapsed_ms,
        )

    def _decide(self, conversation: list[dict[str, str]], memories: list[dict]) -> dict:
        payload = {
            "conversation": conversation,
            "stored_weak_concepts": [
                {
                    "memory_id": item.get("id"),
                    "concept": item.get("concept"),
                    "difficulty_note": item.get("difficulty_note"),
                    "status": item.get("status"),
                    "confidence": item.get("confidence"),
                }
                for item in memories
            ],
        }
        response = self.client_factory().chat.completions.create(
            model=os.environ.get("EXTERNAL_BRAIN_MODEL", "grok-4.3"),
            reasoning_effort=os.environ.get("EXTERNAL_BRAIN_REASONING_EFFORT", "high"),
            max_completion_tokens=int(os.environ.get("EXTERNAL_BRAIN_MAX_TOKENS", "900")),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        content = response.choices[0].message.content or "{}"
        decision = json.loads(content)
        if not isinstance(decision, dict):
            raise ValueError("external brain response must be a JSON object")
        return decision

    async def _apply(self, decision: dict, memories: list[dict]) -> tuple[int, int]:
        known_ids = {item.get("id") for item in memories if item.get("id")}
        saved = 0
        reviewed = 0
        save = decision.get("save")
        if isinstance(save, dict):
            concept = str(save.get("concept", "")).strip()
            question = str(save.get("original_question", "")).strip()
            note = str(save.get("difficulty_note", "")).strip()
            if concept and question and note and _contains_korean(concept) and _contains_korean(note):
                result = await self.memory.save(
                    os.environ.get("EXTERNAL_BRAIN_COURSE", "시계열데이터처리개론"),
                    concept,
                    question,
                    note,
                )
                saved = 1
                log.info("weak-concept save decision applied: %s", result)
            else:
                log.warning("ignored invalid or non-Korean weak-concept save decision")

        seen: set[str] = set()
        for review in decision.get("reviews", []):
            if not isinstance(review, dict):
                continue
            memory_id = str(review.get("memory_id", "")).strip()
            correct = review.get("correct")
            if memory_id not in known_ids or memory_id in seen or type(correct) is not bool:
                continue
            seen.add(memory_id)
            result = await self.memory.review(memory_id, correct)
            reviewed += 1
            log.info("weak-concept review decision applied: %s", result)
        return saved, reviewed

    async def flush(self) -> None:
        """Wait for all currently scheduled assessments."""
        while self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)


def _contains_korean(value: str) -> bool:
    return any("가" <= character <= "힣" for character in value)


def _preview(value: str) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= LOG_PREVIEW_LIMIT else compact[:LOG_PREVIEW_LIMIT] + "…"


def _last_preview(conversation: list[dict[str, str]], role: str) -> str:
    for message in reversed(conversation):
        if message.get("role") == role:
            return _preview(str(message.get("content", "")))
    return "-"

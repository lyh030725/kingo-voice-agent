"""Low-latency weak-concept memory backed by a Moss local session."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any


log = logging.getLogger("moss-memory")


def _load_sdk() -> SimpleNamespace:
    """Import Moss lazily so non-memory tools can still be tested in isolation."""
    try:
        from moss import DocumentInfo, GetDocumentsOptions, MossClient, QueryOptions
    except ImportError as exc:  # pragma: no cover - exercised only in misconfigured installs
        raise RuntimeError(
            "Moss SDK is not installed. Run: uv sync"
        ) from exc
    return SimpleNamespace(
        DocumentInfo=DocumentInfo,
        GetDocumentsOptions=GetDocumentsOptions,
        MossClient=MossClient,
        QueryOptions=QueryOptions,
    )


class MossMemoryStore:
    """Create-or-resume one Moss session and keep its hot index in process."""

    def __init__(
        self,
        project_id: str | None = None,
        project_key: str | None = None,
        *,
        index_name: str | None = None,
        model_id: str | None = None,
        student_id: str | None = None,
        sync_debounce_seconds: float | None = None,
        local_path: str | Path | None = None,
        sdk_loader: Callable[[], Any] = _load_sdk,
    ) -> None:
        self.project_id = (project_id or os.environ.get("MOSS_PROJECT_ID", "")).strip()
        self.project_key = (project_key or os.environ.get("MOSS_PROJECT_KEY", "")).strip()
        self.index_name = (
            index_name
            or os.environ.get("MOSS_MEMORY_INDEX", "kingo-week3-weak-concepts")
        ).strip()
        self.model_id = (
            model_id or os.environ.get("MOSS_MEMORY_MODEL", "moss-minilm")
        ).strip()
        self.student_id = (
            student_id or os.environ.get("MOSS_STUDENT_ID", "default-student")
        ).strip()
        configured_debounce = os.environ.get("MOSS_SYNC_DEBOUNCE_SECONDS", "0.75")
        self.sync_debounce_seconds = (
            float(configured_debounce)
            if sync_debounce_seconds is None
            else sync_debounce_seconds
        )
        self.local_path = Path(
            local_path
            or os.environ.get(
                "MOSS_LOCAL_FALLBACK_FILE",
                Path(__file__).resolve().parent / "memory" / "weak-concepts.json",
            )
        )
        self._sdk_loader = sdk_loader
        self._sdk: Any = None
        self._client: Any = None
        self._session: Any = None
        self._initialize_lock = asyncio.Lock()
        self._operation_lock = asyncio.Lock()
        self._local_lock = asyncio.Lock()
        self._push_task: asyncio.Task[None] | None = None
        self._dirty = False
        self._local_mode = False

    @property
    def is_configured(self) -> bool:
        return bool(self.project_id and self.project_key)

    @property
    def is_local_mode(self) -> bool:
        return self._local_mode

    @staticmethod
    def _is_quota_error(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if status is None:
            status = getattr(getattr(exc, "response", None), "status_code", None)
        message = str(exc).casefold().replace("_", " ")
        return status in {402, 429} or any(
            marker in message
            for marker in (
                "quota",
                "usage limit",
                "usage exceeded",
                "limit exceeded",
                "resource exhausted",
                "insufficient credit",
                "credit balance",
                "payment required",
                "free tier limit",
                "사용 한도",
            )
        )

    def _activate_local_mode(self, exc: Exception) -> bool:
        if not self._is_quota_error(exc):
            return False
        if not self._local_mode:
            log.warning("Moss usage limit reached; using local weak-concept memory: %s", exc)
        self._local_mode = True
        push_task = self._push_task
        if (
            push_task is not None
            and not push_task.done()
            and push_task is not asyncio.current_task()
        ):
            push_task.cancel()
        return True

    def _require_credentials(self) -> None:
        missing = [
            name
            for name, value in (
                ("MOSS_PROJECT_ID", self.project_id),
                ("MOSS_PROJECT_KEY", self.project_key),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                f"{', '.join(missing)} is not set. Add it to your environment."
            )

    async def initialize(self) -> None:
        """Hydrate the cloud index once; later reads and writes stay in-process."""
        if self._session is not None or self._local_mode:
            return
        async with self._initialize_lock:
            if self._session is not None or self._local_mode:
                return
            self._require_credentials()
            try:
                self._sdk = self._sdk_loader()
                self._client = self._sdk.MossClient(self.project_id, self.project_key)
                self._session = await self._client.session(
                    index_name=self.index_name,
                    model_id=self.model_id,
                )
            except Exception as exc:
                if self._activate_local_mode(exc):
                    return
                raise
            log.info(
                "Moss memory ready: index=%s docs=%s model=%s",
                self.index_name,
                getattr(self._session, "doc_count", "?"),
                self.model_id,
            )

    def _memory_id(self, course: str, concept: str) -> str:
        canonical = "\0".join(
            (
                self.student_id.casefold(),
                course.strip().casefold(),
                concept.strip().casefold(),
            )
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return f"M-{digest}"

    def _read_local(self) -> list[dict[str, Any]]:
        if not self.local_path.exists():
            return []
        try:
            data = json.loads(self.local_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.exception("failed to read local weak-concept memory")
            return []
        return [memory for memory in data if isinstance(memory, dict)] if isinstance(data, list) else []

    def _write_local(self, memories: list[dict[str, Any]]) -> None:
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.local_path.with_suffix(self.local_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(memories, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.local_path)

    async def _local_memories(self) -> list[dict[str, Any]]:
        async with self._local_lock:
            return self._read_local()

    async def _upsert_local(self, memory: dict[str, Any]) -> None:
        async with self._local_lock:
            memories = self._read_local()
            by_id = {item.get("id"): item for item in memories if item.get("id")}
            by_id[memory["id"]] = memory
            self._write_local(list(by_id.values()))

    def _saved_memory(
        self,
        fields: dict[str, str],
        existing: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        memory_id = self._memory_id(fields["course"], fields["concept"])
        now = time.time()
        if existing:
            existing.update(
                {
                    "id": memory_id,
                    "student_id": self.student_id,
                    **fields,
                    "status": "practicing",
                    "confidence": max(float(existing.get("confidence", 0)) - 0.2, 0),
                    "failure_count": int(existing.get("failure_count", 0)) + 1,
                    "success_count": 0,
                    "last_seen_at": now,
                    "next_review_at": now + 86400,
                }
            )
            return existing
        return {
            "id": memory_id,
            "student_id": self.student_id,
            **fields,
            "status": "new",
            "confidence": 0.0,
            "failure_count": 1,
            "success_count": 0,
            "saved_at": now,
            "last_seen_at": now,
            "next_review_at": now + 86400,
        }

    async def _save_local(self, fields: dict[str, str]) -> dict[str, Any]:
        async with self._local_lock:
            memories = self._read_local()
            memory_id = self._memory_id(fields["course"], fields["concept"])
            existing = next((item for item in memories if item.get("id") == memory_id), None)
            memory = self._saved_memory(fields, existing)
            by_id = {item.get("id"): item for item in memories if item.get("id")}
            by_id[memory_id] = memory
            self._write_local(list(by_id.values()))
        return {
            "memory_id": memory_id,
            "status": "updated" if existing else "saved",
            "concept": fields["concept"],
            "storage": "local",
        }

    @staticmethod
    def _searchable_text(memory: dict[str, Any]) -> str:
        return (
            f"과목: {memory['course']}\n"
            f"취약 개념: {memory['concept']}\n"
            f"학생 질문: {memory['original_question']}\n"
            f"관찰된 어려움: {memory['difficulty_note']}"
        )

    @staticmethod
    def _memory_from_doc(doc: Any) -> dict[str, Any] | None:
        payload = getattr(doc, "payload", None)
        if not payload:
            return None
        try:
            memory = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            log.warning("ignoring Moss document with malformed payload: %s", doc.id)
            return None
        if not isinstance(memory, dict):
            return None
        return memory

    async def save(
        self,
        course: str,
        concept: str,
        original_question: str,
        difficulty_note: str,
    ) -> dict[str, Any]:
        fields = {
            "course": course.strip(),
            "concept": concept.strip(),
            "original_question": original_question.strip(),
            "difficulty_note": difficulty_note.strip(),
        }
        if not all(fields.values()):
            return {"error": "all weak-concept fields are required"}

        if self._local_mode:
            return await self._save_local(fields)

        await self.initialize()
        if self._local_mode:
            return await self._save_local(fields)
        memory_id = self._memory_id(fields["course"], fields["concept"])
        try:
            async with self._operation_lock:
                docs = await self._session.get_docs(
                    self._sdk.GetDocumentsOptions(doc_ids=[memory_id])
                )
                existing = self._memory_from_doc(docs[0]) if docs else None
                memory = self._saved_memory(fields, existing)
                await self._session.add_docs([self._document(memory)])
                self._dirty = True
        except Exception as exc:
            if self._activate_local_mode(exc):
                return await self._save_local(fields)
            raise

        await self._upsert_local(memory)
        self._schedule_push()
        log.info("weak concept embedded in Moss: %s", memory_id)
        return {
            "memory_id": memory_id,
            "status": "updated" if existing else "saved",
            "concept": fields["concept"],
            "storage": "moss",
        }

    def _document(self, memory: dict[str, Any]) -> Any:
        return self._sdk.DocumentInfo(
            id=memory["id"],
            text=self._searchable_text(memory),
            metadata={
                "student_id": self.student_id,
                "course": memory["course"],
                "concept": memory["concept"],
            },
            payload=json.dumps(memory, ensure_ascii=False),
        )

    async def review(self, memory_id: str, correct: bool) -> dict[str, Any]:
        """Update spaced-repetition state after one recalled concept is assessed."""
        if self._local_mode:
            return await self._review_local(memory_id, correct)
        await self.initialize()
        if self._local_mode:
            return await self._review_local(memory_id, correct)
        now = time.time()
        try:
            async with self._operation_lock:
                docs = await self._session.get_docs(
                    self._sdk.GetDocumentsOptions(doc_ids=[memory_id])
                )
                if not docs or not (memory := self._memory_from_doc(docs[0])):
                    return {"error": "weak concept not found", "memory_id": memory_id}
                self._apply_review(memory, correct, now)
                await self._session.add_docs([self._document(memory)])
                self._dirty = True
        except Exception as exc:
            if self._activate_local_mode(exc):
                return await self._review_local(memory_id, correct)
            raise

        await self._upsert_local(memory)
        self._schedule_push()
        return {
            "memory_id": memory_id,
            "status": memory["status"],
            "correct": correct,
            "storage": "moss",
        }

    @staticmethod
    def _apply_review(memory: dict[str, Any], correct: bool, now: float) -> None:
        successes = int(memory.get("success_count", 0)) + 1 if correct else 0
        failures = int(memory.get("failure_count", 0)) + (not correct)
        delays = (1, 3, 7, 30)
        memory.update(
            {
                "status": "mastered" if successes >= 3 else "practicing",
                "confidence": min(successes / 3, 1.0),
                "success_count": successes,
                "failure_count": failures,
                "last_seen_at": now,
                "next_review_at": now + delays[min(successes, 3)] * 86400,
            }
        )

    async def _review_local(self, memory_id: str, correct: bool) -> dict[str, Any]:
        async with self._local_lock:
            memories = self._read_local()
            memory = next((item for item in memories if item.get("id") == memory_id), None)
            if memory is None:
                return {"error": "weak concept not found", "memory_id": memory_id}
            self._apply_review(memory, correct, time.time())
            self._write_local(memories)
        return {
            "memory_id": memory_id,
            "status": memory["status"],
            "correct": correct,
            "storage": "local",
        }

    async def next_review(self) -> dict[str, Any] | None:
        """Return oldest due, unmastered concept."""
        now = time.time()
        due = [
            memory
            for memory in await self.all_memories()
            if memory.get("status") != "mastered"
            and float(memory.get("next_review_at", 0)) <= now
        ]
        return min(due, key=lambda memory: memory.get("next_review_at", 0), default=None)

    async def recall(self, topic: str, *, top_k: int = 5) -> dict[str, Any]:
        topic = topic.strip()
        if not topic:
            return {"error": "topic is required"}
        if self._local_mode:
            return await self._recall_local(topic, top_k)
        await self.initialize()
        if self._local_mode:
            return await self._recall_local(topic, top_k)
        options = self._sdk.QueryOptions(
            top_k=top_k,
            alpha=0.85,
            filter={
                "field": "student_id",
                "condition": {"$eq": self.student_id},
            },
        )
        try:
            async with self._operation_lock:
                result = await self._session.query(topic, options)
        except Exception as exc:
            if self._activate_local_mode(exc):
                return await self._recall_local(topic, top_k)
            raise
        memories = [
            memory
            for doc in result.docs
            if (memory := self._memory_from_doc(doc)) is not None
        ]
        return self._recall_response(topic, memories, "moss")

    def _recall_response(
        self,
        topic: str,
        memories: list[dict[str, Any]],
        storage: str,
    ) -> dict[str, Any]:
        return {
            "found": bool(memories),
            "topic": topic,
            "storage": storage,
            "memories": [
                {
                    "memory_id": memory["id"],
                    "course": memory["course"],
                    "concept": memory["concept"],
                    "original_question": memory["original_question"],
                    "difficulty_note": memory["difficulty_note"],
                    "status": memory.get("status", "new"),
                    "confidence": memory.get("confidence", 0),
                    "failure_count": memory.get("failure_count", 1),
                    "success_count": memory.get("success_count", 0),
                    "next_review_at": memory.get("next_review_at", 0),
                }
                for memory in memories
            ],
        }

    async def _recall_local(self, topic: str, top_k: int) -> dict[str, Any]:
        query_terms = self._terms(topic)
        # ponytail: lexical fallback only; add local embeddings if recall quality becomes insufficient.
        scored = [
            (len(query_terms & self._terms(self._searchable_text(memory))), memory)
            for memory in await self._local_memories()
            if memory.get("student_id") == self.student_id
        ]
        memories = [
            memory
            for score, memory in sorted(
                scored,
                key=lambda item: (item[0], float(item[1].get("last_seen_at", 0))),
                reverse=True,
            )
            if score > 0
        ][:top_k]
        return self._recall_response(topic, memories, "local")

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {
            token.casefold()
            for token in re.findall(r"[0-9A-Za-z가-힣_]{2,}", text)
        }

    async def all_memories(self) -> list[dict[str, Any]]:
        """Return structured records for the optional follow-up worker."""
        if self._local_mode:
            return [
                memory
                for memory in await self._local_memories()
                if memory.get("student_id") == self.student_id
            ]
        await self.initialize()
        if self._local_mode:
            return await self.all_memories()
        try:
            async with self._operation_lock:
                docs = await self._session.get_docs()
        except Exception as exc:
            if self._activate_local_mode(exc):
                return await self.all_memories()
            raise
        return [
            memory
            for doc in docs
            if (memory := self._memory_from_doc(doc)) is not None
            and memory.get("student_id") == self.student_id
        ]

    def _schedule_push(self) -> None:
        if self._push_task is None or self._push_task.done():
            self._push_task = asyncio.create_task(self._debounced_push())

    async def _debounced_push(self) -> None:
        await asyncio.sleep(max(self.sync_debounce_seconds, 0))
        try:
            async with self._operation_lock:
                if not self._dirty:
                    return
                await self._session.push_index()
                self._dirty = False
                log.info("Moss memory synced to cloud: %s", self.index_name)
        except Exception as exc:
            if not self._activate_local_mode(exc):
                log.exception("Moss memory cloud sync failed; next save will retry")

    async def flush(self) -> None:
        """Wait for a scheduled cloud push, or persist dirty data immediately."""
        if self._local_mode:
            return
        task = self._push_task
        if task is not None and not task.done():
            await task
        if self._dirty and self._session is not None:
            async with self._operation_lock:
                await self._session.push_index()
                self._dirty = False

    async def close(self) -> None:
        await self.flush()

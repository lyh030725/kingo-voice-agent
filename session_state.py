"""Per-student memory and per-chat history isolation for KINGO."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from external_brain import ExternalBrain
from moss_memory import MossMemoryStore

BASE_DIR = Path(__file__).resolve().parent
MAX_ID_LENGTH = 128

_HISTORIES: dict[tuple[str, str], list[dict[str, str]]] = {}
_MEMORIES: dict[str, MossMemoryStore] = {}
_EXTERNAL_BRAINS: dict[str, ExternalBrain] = {}
_LOCK = asyncio.Lock()


def clean_id(value: str | None, fallback: str) -> str:
    """Return a bounded identifier safe for use as an in-process key."""
    value = (value or "").strip()
    if not value or len(value) > MAX_ID_LENGTH or not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        return fallback
    return value


def history_for(student_id: str, session_id: str) -> list[dict[str, str]]:
    """Return the mutable history owned by one student's chat session."""
    key = (clean_id(student_id, "default-student"), clean_id(session_id, "default-session"))
    return _HISTORIES.setdefault(key, [])


def clear_history(student_id: str, session_id: str) -> None:
    """Delete only one chat history; persistent learner memory is retained."""
    key = (clean_id(student_id, "default-student"), clean_id(session_id, "default-session"))
    _HISTORIES.pop(key, None)


def _local_memory_path(student_id: str) -> Path:
    digest = hashlib.sha256(student_id.encode("utf-8")).hexdigest()[:16]
    return BASE_DIR / "memory" / f"weak-concepts-{digest}.json"


def memory_for(student_id: str) -> MossMemoryStore:
    """Return one shared memory store per learner."""
    student_id = clean_id(student_id, "default-student")
    store = _MEMORIES.get(student_id)
    if store is None:
        store = MossMemoryStore(
            student_id=student_id,
            local_path=_local_memory_path(student_id),
        )
        _MEMORIES[student_id] = store
    return store


def external_brain_for(student_id: str, client_factory: Callable[[], Any]) -> ExternalBrain:
    """Return one background assessor bound to the learner's memory store."""
    student_id = clean_id(student_id, "default-student")
    worker = _EXTERNAL_BRAINS.get(student_id)
    if worker is None:
        worker = ExternalBrain(memory_for(student_id), client_factory)
        _EXTERNAL_BRAINS[student_id] = worker
    return worker


async def close_all() -> None:
    """Flush all learner workers and memory stores at process shutdown."""
    async with _LOCK:
        workers = list(_EXTERNAL_BRAINS.values())
        stores = list(_MEMORIES.values())
    if workers:
        await asyncio.gather(*(worker.flush() for worker in workers), return_exceptions=True)
    if stores:
        await asyncio.gather(*(store.close() for store in stores), return_exceptions=True)

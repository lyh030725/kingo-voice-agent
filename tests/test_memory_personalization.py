from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ["VOICE_AI_SKIP_DOTENV"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent_spec
import brain_runtime
import grok_live


class FakeSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed = False

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))

    async def close(self) -> None:
        self.closed = True


class MemoryPersonalizationTests(unittest.TestCase):
    def test_recent_history_question_routes_to_recent_memories(self) -> None:
        async def scenario() -> None:
            recent = {
                "found": True,
                "recall_type": "recent",
                "memories": [{"concept": "일반 닷 프로덕트 어텐션"}],
            }
            with patch.object(
                brain_runtime,
                "recent_weak_concepts",
                new=AsyncMock(return_value=recent),
            ) as recall_recent:
                result = json.loads(await brain_runtime.recall_weak_concepts(
                    "저번에 공부했던 거 뭐였지?", "student-a"
                ))

            recall_recent.assert_awaited_once_with(
                "student-a",
                topic="저번에 공부했던 거 뭐였지?",
                recall_type="recent",
            )
            self.assertEqual(result["recall_type"], "recent")
            self.assertEqual(result["memories"][0]["concept"], "일반 닷 프로덕트 어텐션")

        asyncio.run(scenario())

    def test_recent_memories_are_sorted_by_last_seen(self) -> None:
        async def scenario() -> None:
            store = SimpleNamespace(all_memories=AsyncMock(return_value=[
                {
                    "id": "old",
                    "concept": "RNN",
                    "difficulty_note": "hidden state를 혼동함",
                    "last_seen_at": 10,
                    "saved_at": 5,
                },
                {
                    "id": "new",
                    "concept": "Self-Attention",
                    "difficulty_note": "Q와 K의 역할을 혼동함",
                    "last_seen_at": 30,
                    "saved_at": 20,
                },
            ]))
            with patch.object(brain_runtime, "memory_for", return_value=store):
                result = await brain_runtime.recent_weak_concepts(
                    "student-a", top_k=2, topic="최근 공부", recall_type="recent"
                )

            self.assertEqual([item["memory_id"] for item in result["memories"]], ["new", "old"])
            self.assertEqual(result["memories"][0]["last_seen_at"], 30)

        asyncio.run(scenario())

    def test_topic_question_keeps_semantic_recall(self) -> None:
        async def scenario() -> None:
            store = SimpleNamespace(recall=AsyncMock(return_value={
                "found": True,
                "memories": [{
                    "memory_id": "M-1",
                    "concept": "일반 닷 프로덕트 어텐션",
                    "difficulty_note": "개념 자체를 모름",
                    "status": "practicing",
                }],
            }))
            with patch.object(brain_runtime, "memory_for", return_value=store):
                result = json.loads(await brain_runtime.recall_weak_concepts(
                    "scaled dot product attention에서 왜 scaling해?", "student-a"
                ))

            store.recall.assert_awaited_once()
            self.assertEqual(result["recall_type"], "semantic")

        asyncio.run(scenario())

    def test_voice_prompt_makes_weak_concept_usage_visible(self) -> None:
        prompt = agent_spec.persona("socratic", {
            "found": True,
            "recall_type": "semantic",
            "memories": [{
                "concept": "일반 닷 프로덕트 어텐션",
                "difficulty_note": "개념 자체를 모름",
                "status": "practicing",
                "confidence": 0,
            }],
        })
        self.assertIn("make relevant personalization visible", prompt)
        self.assertIn("retrieval practice", prompt)
        self.assertIn("전에 이 부분에서 조금 막혔었어요", prompt)
        self.assertIn("일반 닷 프로덕트 어텐션", prompt)

    def test_realtime_session_bootstraps_recent_memory_before_configuration(self) -> None:
        async def scenario() -> None:
            transport = grok_live.GrokTransport(student_id="student-a")
            transport._ready.set()
            socket = FakeSocket()
            bootstrap = {
                "found": True,
                "recall_type": "bootstrap",
                "memories": [{
                    "concept": "일반 닷 프로덕트 어텐션",
                    "difficulty_note": "개념 자체를 모름",
                    "status": "practicing",
                }],
            }
            with (
                patch.dict(os.environ, {"XAI_API_KEY": "test-key"}),
                patch("grok_live.websockets.connect", new=AsyncMock(return_value=socket)),
                patch("grok_live.bootstrap_memory_context", new=AsyncMock(return_value=bootstrap)) as load,
            ):
                await transport.start()

            load.assert_awaited_once_with("student-a")
            self.assertEqual(transport._memory_context["recall_type"], "bootstrap")
            update = socket.sent[0]
            self.assertEqual(update["type"], "session.update")
            self.assertIn("일반 닷 프로덕트 어텐션", update["session"]["instructions"])

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()

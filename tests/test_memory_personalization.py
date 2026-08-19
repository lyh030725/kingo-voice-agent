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
import brain
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
    def test_recent_memories_are_sorted_and_limited_to_three(self) -> None:
        async def scenario() -> None:
            store = SimpleNamespace(
                all_memories=AsyncMock(
                    return_value=[
                        {
                            "id": "old",
                            "concept": "RNN",
                            "difficulty_note": "hidden state를 혼동함",
                            "last_seen_at": 10,
                            "saved_at": 5,
                        },
                        {
                            "id": "newest",
                            "concept": "Self-Attention",
                            "difficulty_note": "Q와 K의 역할을 혼동함",
                            "last_seen_at": 40,
                            "saved_at": 20,
                        },
                        {
                            "id": "third",
                            "concept": "Softmax",
                            "difficulty_note": "확률 변환 이유가 불명확함",
                            "last_seen_at": 20,
                            "saved_at": 15,
                        },
                        {
                            "id": "second",
                            "concept": "Dot Product Attention",
                            "difficulty_note": "유사도 의미를 혼동함",
                            "last_seen_at": 30,
                            "saved_at": 18,
                        },
                    ]
                )
            )
            with patch.object(brain, "_memory_store", return_value=store):
                result = await brain.recent_weak_concepts("student-a")

            self.assertEqual(
                [item["concept"] for item in result["memories"]],
                ["Self-Attention", "Dot Product Attention", "Softmax"],
            )
            self.assertNotIn("RNN", [item["concept"] for item in result["memories"]])
            self.assertNotIn("last_seen_at", result["memories"][0])
            self.assertNotIn("memory_id", result["memories"][0])

        asyncio.run(scenario())

    def test_recall_is_topic_agnostic_and_returns_recent_three(self) -> None:
        async def scenario() -> None:
            recent = {
                "found": True,
                "memories": [
                    {
                        "concept": "일반 닷 프로덕트 어텐션",
                        "difficulty": "개념 자체를 모름",
                        "status": "practicing",
                    }
                ],
            }
            with patch.object(
                brain,
                "recent_weak_concepts",
                new=AsyncMock(return_value=recent),
            ) as load_recent:
                first = json.loads(
                    await brain.recall_weak_concepts(
                        "scaled dot product attention이 뭐야?", "student-a"
                    )
                )
                second = json.loads(
                    await brain.recall_weak_concepts(
                        "저번에 뭐 공부했지?", "student-a"
                    )
                )

            self.assertEqual(first, second)
            self.assertEqual(load_recent.await_count, 2)
            for call in load_recent.await_args_list:
                self.assertEqual(call.args[0], "student-a")
                self.assertEqual(call.kwargs["top_k"], 3)

        asyncio.run(scenario())

    def test_voice_prompt_is_short_and_exposes_recent_memories(self) -> None:
        prompt = agent_spec.persona(
            "socratic",
            {
                "found": True,
                "memories": [
                    {
                        "concept": "일반 닷 프로덕트 어텐션",
                        "difficulty": "개념 자체를 모름",
                        "status": "practicing",
                    }
                ],
            },
        )
        self.assertIn("Up to three", prompt)
        self.assertIn("most recent weak concepts", prompt)
        self.assertIn("일반 닷 프로덕트 어텐션", prompt)
        self.assertNotIn("recall_type", prompt)
        self.assertNotIn("retrieval practice", prompt)

    def test_realtime_session_bootstraps_recent_memory_before_configuration(self) -> None:
        async def scenario() -> None:
            transport = grok_live.GrokTransport(student_id="student-a")
            transport._ready.set()
            socket = FakeSocket()
            bootstrap = {
                "found": True,
                "memories": [
                    {
                        "concept": "일반 닷 프로덕트 어텐션",
                        "difficulty": "개념 자체를 모름",
                        "status": "practicing",
                    }
                ],
            }
            with (
                patch.dict(os.environ, {"XAI_API_KEY": "test-key"}),
                patch(
                    "grok_live.websockets.connect",
                    new=AsyncMock(return_value=socket),
                ),
                patch(
                    "grok_live.bootstrap_memory_context",
                    new=AsyncMock(return_value=bootstrap),
                ) as load,
            ):
                await transport.start()

            load.assert_awaited_once_with("student-a")
            self.assertEqual(transport._memory_context, bootstrap)
            update = socket.sent[0]
            self.assertEqual(update["type"], "session.update")
            self.assertIn(
                "일반 닷 프로덕트 어텐션",
                update["session"]["instructions"],
            )

        asyncio.run(scenario())

    def test_voice_turn_refreshes_recent_memory_without_transcript_search(self) -> None:
        async def scenario() -> None:
            transport = grok_live.GrokTransport(student_id="student-a")
            transport._ws = FakeSocket()
            transport._memory_context = {"found": False}
            latest = {
                "found": True,
                "memories": [
                    {
                        "concept": "Softmax",
                        "difficulty": "정규화 이유를 혼동함",
                        "status": "practicing",
                    }
                ],
            }
            with patch(
                "grok_live.bootstrap_memory_context",
                new=AsyncMock(return_value=latest),
            ) as refresh:
                await transport._refresh_memory_context()

            refresh.assert_awaited_once_with("student-a")
            self.assertEqual(transport._memory_context, latest)
            self.assertIn(
                "Softmax",
                transport._ws.sent[0]["session"]["instructions"],
            )

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()

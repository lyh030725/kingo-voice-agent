import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from external_brain import ExternalBrain


class ExternalBrainTests(unittest.TestCase):
    def test_context_drives_korean_save_and_only_known_review(self) -> None:
        memory = SimpleNamespace(
            all_memories=AsyncMock(return_value=[{
                "id": "M-known", "concept": "차분",
                "difficulty_note": "차분 목적을 혼동함",
                "status": "practicing", "confidence": 0,
            }]),
            save=AsyncMock(return_value={"status": "saved"}),
            review=AsyncMock(return_value={"status": "practicing"}),
        )
        completion = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps({
                "save": {
                    "concept": "정상성 판단",
                    "original_question": "추세가 있어도 정상인가요?",
                    "difficulty_note": "추세와 정상성의 관계를 반대로 이해함",
                },
                "reviews": [
                    {"memory_id": "M-known", "correct": True},
                    {"memory_id": "M-invented", "correct": False},
                ],
            }, ensure_ascii=False))
        )]))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
        external = ExternalBrain(memory, lambda: client)
        context = [
            {"role": "user", "content": "차분하면 추세가 없어지는 거죠?"},
            {"role": "assistant", "content": "그 이유를 말해볼까요?"},
        ]

        with patch.dict("os.environ", {"EXTERNAL_BRAIN_COURSE": "시계열"}):
            asyncio.run(external.assess(context))

        sent = json.loads(completion.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(sent["conversation"], context)
        self.assertEqual(sent["stored_weak_concepts"][0]["memory_id"], "M-known")
        memory.save.assert_awaited_once_with(
            "시계열", "정상성 판단", "추세가 있어도 정상인가요?", "추세와 정상성의 관계를 반대로 이해함"
        )
        memory.review.assert_awaited_once_with("M-known", True)

    def test_non_korean_save_is_ignored(self) -> None:
        memory = SimpleNamespace(save=AsyncMock(), review=AsyncMock())
        external = ExternalBrain(memory, Mock())
        asyncio.run(external._apply({
            "save": {"concept": "stationarity", "original_question": "what?", "difficulty_note": "trend"},
            "reviews": [],
        }, []))
        memory.save.assert_not_awaited()

    def test_schedule_returns_before_assessment_finishes_and_flush_waits(self) -> None:
        external = ExternalBrain(SimpleNamespace(), Mock())

        async def scenario() -> None:
            gate = asyncio.Event()

            async def wait_for_gate(_context, **_kwargs) -> None:
                await gate.wait()

            external.assess = wait_for_gate
            task = external.schedule([{"role": "user", "content": "질문"}])
            self.assertIsNotNone(task)
            self.assertFalse(task.done())
            gate.set()
            await external.flush()
            self.assertTrue(task.done())

        asyncio.run(scenario())

    def test_logs_show_end_to_end_assessment_lifecycle(self) -> None:
        memory = SimpleNamespace(
            all_memories=AsyncMock(return_value=[]),
            save=AsyncMock(),
            review=AsyncMock(),
        )
        completion = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"save": null, "reviews": []}')
        )]))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
        external = ExternalBrain(memory, lambda: client)

        async def scenario() -> None:
            with self.assertLogs("external-brain", level="INFO") as logs:
                external.schedule([
                    {"role": "user", "content": "정상성을 이해했어요"},
                    {"role": "assistant", "content": "이유를 설명해 볼까요?"},
                ], source="realtime")
                await external.flush()
            output = "\n".join(logs.output)
            self.assertIn("external brain scheduled id=", output)
            self.assertIn("source=realtime", output)
            self.assertIn("external brain context ready id=", output)
            self.assertIn("external brain decided id=", output)
            self.assertIn("external brain completed id=", output)
            self.assertIn("saved=0 reviewed=0", output)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()

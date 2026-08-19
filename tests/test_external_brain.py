import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from external_brain import ExternalBrain, SYSTEM_PROMPT, _explicit_uncertainty_evidence


class ExternalBrainTests(unittest.TestCase):
    def test_prompt_treats_repeated_struggle_as_weakness_evidence(self) -> None:
        self.assertIn("답을 반복해 못하거나", SYSTEM_PROMPT)
        self.assertIn("힌트 후에도", SYSTEM_PROMPT)

    def test_prompt_treats_explicit_uncertainty_as_strong_evidence(self) -> None:
        for phrase in (
            '"모르겠어요"',
            '"잘 모르겠어요"',
            '"기억이 안 나요"',
            '"감이 안 와요"',
            "강한 이해 부족의 증거",
            "문맥을 사용하는 것은 추측이 아닙니다",
        ):
            self.assertIn(phrase, SYSTEM_PROMPT)
        self.assertIn("한 번의 명시적 모름 응답도", SYSTEM_PROMPT)
        self.assertIn("직전 튜터 질문과 학생 응답을 함께", SYSTEM_PROMPT)

    def test_explicit_uncertainty_pairs_response_with_preceding_tutor_prompt(self) -> None:
        conversation = [
            {"role": "assistant", "content": "차분을 하면 추세에는 어떤 일이 생길까요?"},
            {"role": "user", "content": "잘 모르겠어요."},
            {"role": "assistant", "content": "추세 제거 관점에서 다시 생각해볼까요?"},
            {"role": "user", "content": "감이 안 와요."},
        ]

        evidence = _explicit_uncertainty_evidence(conversation)

        self.assertEqual(evidence, [
            {
                "preceding_tutor_prompt": "차분을 하면 추세에는 어떤 일이 생길까요?",
                "student_response": "잘 모르겠어요.",
            },
            {
                "preceding_tutor_prompt": "추세 제거 관점에서 다시 생각해볼까요?",
                "student_response": "감이 안 와요.",
            },
        ])

    def test_non_uncertainty_answer_is_not_marked_as_explicit_failure(self) -> None:
        conversation = [
            {"role": "assistant", "content": "정상성의 조건을 말해볼까요?"},
            {"role": "user", "content": "평균과 분산이 시간에 따라 일정한 거예요."},
        ]
        self.assertEqual(_explicit_uncertainty_evidence(conversation), [])

    def test_decide_sends_explicit_uncertainty_evidence_to_model(self) -> None:
        completion = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"save": null, "reviews": []}')
        )]))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
        external = ExternalBrain(SimpleNamespace(), lambda: client)
        conversation = [
            {"role": "assistant", "content": "AR 모델에서 현재 값은 무엇에 의존할까요?"},
            {"role": "user", "content": "기억이 안 나요."},
        ]

        external._decide(conversation, [])

        sent = json.loads(completion.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(sent["explicit_uncertainty_evidence"], [{
            "preceding_tutor_prompt": "AR 모델에서 현재 값은 무엇에 의존할까요?",
            "student_response": "기억이 안 나요.",
        }])

    def test_legacy_weakness_settings_configure_external_brain(self) -> None:
        memory = SimpleNamespace(all_memories=AsyncMock(return_value=[]))
        completion = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"save": null, "reviews": []}')
        )]))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
        external = ExternalBrain(memory, lambda: client)

        with patch.dict("os.environ", {
            "EXTERNAL_BRAIN_MODEL": "",
            "EXTERNAL_BRAIN_REASONING_EFFORT": "",
            "EXTERNAL_BRAIN_MAX_TOKENS": "",
            "WEAKNESS_MODEL": "legacy-model",
            "WEAKNESS_REASONING_EFFORT": "low",
            "WEAKNESS_MAX_TOKENS": "321",
        }):
            external._decide([], [])

        self.assertEqual(completion.call_args.kwargs["model"], "legacy-model")
        self.assertEqual(completion.call_args.kwargs["reasoning_effort"], "low")
        self.assertEqual(completion.call_args.kwargs["max_completion_tokens"], 321)

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
        self.assertEqual(sent["explicit_uncertainty_evidence"], [])
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
            self.assertIn("explicit_uncertainty=0", output)
            self.assertIn("external brain decided id=", output)
            self.assertIn("external brain completed id=", output)
            self.assertIn("saved=0 reviewed=0", output)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()

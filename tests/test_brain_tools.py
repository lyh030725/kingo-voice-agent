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

import brain


class ToolCall:
    def __init__(self, call_id: str, name: str, args: dict) -> None:
        self.id = call_id
        self.function = SimpleNamespace(
            name=name,
            arguments=json.dumps(args, ensure_ascii=False),
        )

    def model_dump(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


class FakeClient:
    def __init__(self) -> None:
        self.messages = iter([
            SimpleNamespace(content=None, tool_calls=[
                ToolCall("recall", "recall_weak_concepts", {"topic": "attention"}),
                ToolCall("pdf", "search_course_materials", {"query": "attention"}),
            ]),
            SimpleNamespace(content=None, tool_calls=[
                ToolCall("web", "search_trusted_web", {
                    "query": "attention paper",
                    "reason": "PDF 설명이 불충분함",
                }),
                ToolCall("save", "save_weak_concept", {
                    "course": "AI 개론",
                    "concept": "Self-Attention",
                    "original_question": "왜 곱해?",
                    "difficulty_note": "유사도 의미가 불명확함",
                }),
                ToolCall("review", "review_weak_concept", {
                    "memory_id": "M-test",
                    "correct": True,
                }),
            ]),
            SimpleNamespace(
                content=json.dumps({"answer": "근거를 바탕으로 설명할게요."}, ensure_ascii=False),
                tool_calls=[],
            ),
        ])
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create),
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=next(self.messages))],
        )


class BrainToolTests(unittest.TestCase):
    def setUp(self) -> None:
        brain.HISTORY.clear()

    def tearDown(self) -> None:
        brain.HISTORY.clear()

    def test_all_schemas_execute_through_dispatcher(self) -> None:
        fake_client = FakeClient()
        with (
            patch.object(brain, "xai_client", return_value=fake_client),
            patch.object(
                brain,
                "recall_weak_concepts",
                new=AsyncMock(return_value=json.dumps({"found": False, "memories": []})),
            ) as recall,
            patch.object(
                brain,
                "search_course_materials",
                return_value=json.dumps({"found": True, "results": []}),
            ) as course_search,
            patch.object(
                brain,
                "search_trusted_web",
                return_value=json.dumps({
                    "found": True,
                    "sources": ["https://arxiv.org/abs/1706.03762"],
                }),
            ) as web_search,
            patch.object(
                brain,
                "save_weak_concept",
                new=AsyncMock(return_value=json.dumps({"status": "saved"})),
            ) as save,
            patch.object(
                brain,
                "review_weak_concept",
                new=AsyncMock(return_value=json.dumps({"status": "practicing"})),
            ) as review,
        ):
            reply, tools = asyncio.run(
                brain.think("Query와 Key를 왜 곱해?", brain.StageTimer())
            )

        self.assertEqual(
            {tool["function"]["name"] for tool in brain.TOOLS},
            {
                "recall_weak_concepts",
                "search_course_materials",
                "search_trusted_web",
                "save_weak_concept",
                "review_weak_concept",
            },
        )
        self.assertEqual(set(tools), {
            "recall_weak_concepts",
            "search_course_materials",
            "search_trusted_web",
            "save_weak_concept",
            "review_weak_concept",
        })
        self.assertEqual(fake_client.calls[0]["tool_choice"], "required")
        self.assertEqual(fake_client.calls[1]["tool_choice"], "auto")
        self.assertTrue(fake_client.calls[0]["parallel_tool_calls"])
        self.assertEqual(
            sum(message["role"] == "tool" for message in fake_client.calls[2]["messages"]),
            5,
        )
        recall.assert_awaited_once_with(topic="attention")
        course_search.assert_called_once_with(query="attention")
        web_search.assert_called_once_with(
            pdf_evidence_insufficient=True,
            query="attention paper",
            reason="PDF 설명이 불충분함",
        )
        save.assert_awaited_once()
        review.assert_awaited_once_with(memory_id="M-test", correct=True)
        self.assertIn("https://arxiv.org/abs/1706.03762", reply)

    def test_tool_docstrings_describe_contracts(self) -> None:
        for target in (
            brain.save_weak_concept,
            brain.recall_weak_concepts,
            brain.search_course_materials,
            brain.search_trusted_web,
            brain.review_weak_concept,
            brain.run_tool,
        ):
            doc = target.__doc__ or ""
            self.assertIn("Args:", doc)
            self.assertIn("Returns:", doc)


if __name__ == "__main__":
    unittest.main()

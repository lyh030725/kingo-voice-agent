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
                ToolCall("visual", "show_visualization", {
                    "title": "Attention 가중치",
                    "kind": "formula",
                    "caption": "유사도 점수를 비율로 바꿔요.",
                    "latex": r"\\alpha_i=\\frac{e^{s_i}}{\\sum_j e^{s_j}}",
                    "labels": [],
                    "points": [],
                    "x_label": "",
                    "y_label": "",
                }),
            ]),
            SimpleNamespace(
                content="근거를 바탕으로 설명할게요. https://arxiv.org/abs/1706.03762",
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
            reply, tools, sources, visualizations = asyncio.run(
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
                "show_visualization",
            },
        )
        self.assertEqual(set(tools), {
            "recall_weak_concepts",
            "search_course_materials",
            "search_trusted_web",
            "save_weak_concept",
            "review_weak_concept",
            "show_visualization",
        })
        self.assertEqual(fake_client.calls[0]["tool_choice"], "required")
        self.assertEqual(fake_client.calls[0]["max_completion_tokens"], 1200)
        self.assertNotIn("response_format", fake_client.calls[0])
        self.assertEqual(fake_client.calls[1]["tool_choice"], "auto")
        self.assertTrue(fake_client.calls[0]["parallel_tool_calls"])
        self.assertEqual(
            sum(message["role"] == "tool" for message in fake_client.calls[2]["messages"]),
            6,
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
        self.assertNotIn("https://arxiv.org/abs/1706.03762", reply)
        self.assertEqual(sources, ["https://arxiv.org/abs/1706.03762"])
        self.assertEqual(visualizations[0]["kind"], "formula")
        self.assertEqual(visualizations[0]["title"], "Attention 가중치")

    def test_confusion_does_not_force_save_without_model_tool_call(self) -> None:
        messages = iter([
            SimpleNamespace(content=None, tool_calls=[
                ToolCall("recall", "recall_weak_concepts", {"topic": "attention"}),
                ToolCall("pdf", "search_course_materials", {"query": "attention"}),
            ]),
            SimpleNamespace(content="어느 부분부터 막히는지 같이 찾아볼까요?", tool_calls=[]),
        ])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **_kwargs: SimpleNamespace(
                choices=[SimpleNamespace(message=next(messages))],
            ),
        )))

        with (
            patch.object(brain, "xai_client", return_value=client),
            patch.object(
                brain,
                "recall_weak_concepts",
                new=AsyncMock(return_value=json.dumps({"found": False, "memories": []})),
            ),
            patch.object(
                brain,
                "search_course_materials",
                return_value=json.dumps({"found": False, "results": []}),
            ),
            patch.object(brain, "save_weak_concept", new=AsyncMock()) as save,
        ):
            _reply, tools, _sources, _visualizations = asyncio.run(
                brain.think("Self-Attention을 잘 모르겠어", brain.StageTimer())
            )

        save.assert_not_awaited()
        self.assertNotIn("save_weak_concept", tools)

    def test_stream_completion_emits_each_text_delta(self) -> None:
        chunks = iter([
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="첫 ", tool_calls=[]))]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="토큰", tool_calls=[]))]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[
                SimpleNamespace(
                    index=0,
                    id="call-1",
                    function=SimpleNamespace(name="show_", arguments='{"x":'),
                ),
            ]))]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[
                SimpleNamespace(
                    index=0,
                    id=None,
                    function=SimpleNamespace(name="visualization", arguments="1}"),
                ),
            ]))]),
        ])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **_kwargs: chunks,
        )))
        received = []

        async def scenario():
            async def on_token(token: str) -> None:
                received.append(token)

            return await brain._stream_completion(client, {}, on_token)

        message = asyncio.run(scenario())

        self.assertEqual(received, ["첫 ", "토큰"])
        self.assertEqual(message.content, "첫 토큰")
        self.assertEqual(message.tool_calls[0].function.name, "show_visualization")
        self.assertEqual(message.tool_calls[0].function.arguments, '{"x":1}')

    def test_visualization_rejects_incomplete_shapes(self) -> None:
        with self.assertRaises(ValueError):
            brain.show_visualization(
                title="빈 그래프",
                kind="plot",
                caption="점이 부족해요.",
                latex="",
                labels=[],
                points=[{"x": 0, "y": 0}],
                x_label="x",
                y_label="y",
            )

    def test_tool_logs_include_args_status_timing_and_result(self) -> None:
        args = {
            "title": "소프트맥스",
            "kind": "formula",
            "caption": "확률로 변환해요.",
            "latex": r"\frac{e^{x_i}}{\sum_j e^{x_j}}",
            "labels": [],
            "points": [],
            "x_label": "",
            "y_label": "",
        }

        with self.assertLogs(brain.log, level="INFO") as logs:
            asyncio.run(brain.run_tool("show_visualization", args, brain.StageTimer()))

        output = "\n".join(logs.output)
        self.assertIn('tool call name=show_visualization args={"title":"소프트맥스"', output)
        self.assertIn("tool result name=show_visualization status=ok elapsed_ms=", output)
        self.assertIn('result={"title":"소프트맥스"', output)

    def test_tool_docstrings_describe_contracts(self) -> None:
        for target in (
            brain.save_weak_concept,
            brain.recall_weak_concepts,
            brain.search_course_materials,
            brain.search_trusted_web,
            brain.review_weak_concept,
            brain.show_visualization,
            brain.run_tool,
        ):
            doc = target.__doc__ or ""
            self.assertIn("Args:", doc)
            self.assertIn("Returns:", doc)


if __name__ == "__main__":
    unittest.main()

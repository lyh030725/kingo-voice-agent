from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ["VOICE_AI_SKIP_DOTENV"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent_spec
import brain


class FakeSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


class RealtimePrefetchTests(unittest.TestCase):
    def test_realtime_schema_exposes_model_selected_context_tools(self) -> None:
        names = {tool["name"] for tool in agent_spec.json_schemas()}
        self.assertEqual(names, {
            "recall_weak_concepts",
            "search_course_materials",
            "search_trusted_web",
            "show_visualization",
        })

    def test_prefetch_context_runs_memory_and_pdf_without_model_tool_calls(self) -> None:
        async def scenario() -> None:
            with (
                patch.object(
                    brain,
                    "recall_weak_concepts",
                    new=AsyncMock(return_value=json.dumps({"memories": [{"concept": "softmax"}]})),
                ) as recall,
                patch.object(
                    brain,
                    "search_course_materials",
                    return_value=json.dumps({"found": True, "results": [{"source": "week3.pdf p.4"}]}),
                ) as pdf,
            ):
                context = await brain.prefetch_context(" softmax scaling ")

            recall.assert_awaited_once_with("softmax scaling")
            pdf.assert_called_once_with("softmax scaling")
            self.assertEqual(context["student_question"], "softmax scaling")
            self.assertEqual(context["weak_concepts"]["memories"][0]["concept"], "softmax")
            self.assertTrue(context["course_materials"]["found"])

        asyncio.run(scenario())

    def test_session_persona_routes_simple_turns_without_filler(self) -> None:
        prompt = agent_spec.persona("socratic")
        self.assertIn("Answer greetings, thanks, casual conversation, and", prompt)
        self.assertIn("Only immediately before calling search_course_materials", prompt)
        self.assertIn("not say a filler before show_visualization", prompt)
        self.assertIn("Socratic mode:", prompt)


if __name__ == "__main__":
    unittest.main()

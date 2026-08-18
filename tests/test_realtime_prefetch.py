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
from grok_live import GrokTransport


class FakeSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


class RealtimePrefetchTests(unittest.TestCase):
    def test_realtime_schema_exposes_only_optional_tools(self) -> None:
        names = {tool["name"] for tool in agent_spec.json_schemas()}
        self.assertEqual(names, {"search_trusted_web", "show_visualization"})
        self.assertNotIn("recall_weak_concepts", names)
        self.assertNotIn("search_course_materials", names)

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
                context = await agent_spec.prefetch_context(" softmax scaling ")

            recall.assert_awaited_once_with("softmax scaling")
            pdf.assert_called_once_with("softmax scaling")
            self.assertEqual(context["student_question"], "softmax scaling")
            self.assertEqual(context["weak_concepts"]["memories"][0]["concept"], "softmax")
            self.assertTrue(context["course_materials"]["found"])

        asyncio.run(scenario())

    def test_final_response_uses_same_shared_policy_as_text(self) -> None:
        async def scenario() -> None:
            transport = GrokTransport(mode="socratic")
            transport._ws = FakeSocket()
            context = {
                "student_question": "attention scaling",
                "weak_concepts": {"memories": []},
                "course_materials": {
                    "found": True,
                    "results": [{"source": "week3.pdf p.7", "excerpt": "scaled dot product"}],
                },
            }
            transport._turn_context = context

            await transport._request_answer_response()

            self.assertEqual(transport._next_response_phase, "answer")
            payload = transport._ws.sent[-1]
            self.assertEqual(payload["type"], "response.create")
            instructions = payload["response"]["instructions"]
            self.assertEqual(instructions, brain.answer_instructions("socratic", context))
            self.assertIn("# Preloaded context", instructions)
            self.assertIn("week3.pdf p.7", instructions)
            self.assertNotIn("recall_weak_concepts", instructions)
            self.assertNotIn("search_course_materials", instructions)

        asyncio.run(scenario())

    def test_session_persona_is_filler_only_and_keeps_dynamic_topic_guidance(self) -> None:
        prompt = agent_spec.persona("socratic")
        self.assertIn("first response after each student turn", prompt)
        self.assertIn("Make it fit the student's topic", prompt)
        self.assertIn("Do not answer", prompt)
        self.assertNotIn("Socratic mode:", prompt)


if __name__ == "__main__":
    unittest.main()

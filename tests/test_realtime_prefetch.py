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
import brain_runtime


class RealtimePrefetchTests(unittest.TestCase):
    def test_realtime_schema_exposes_exactly_three_conversational_tools(self) -> None:
        names = {tool["name"] for tool in agent_spec.json_schemas()}
        self.assertEqual(names, {
            "search_course_materials",
            "search_trusted_web",
            "show_visualization",
        })
        self.assertNotIn("recall_weak_concepts", names)

    def test_voice_memory_prefetch_does_not_search_pdf(self) -> None:
        async def scenario() -> None:
            with (
                patch.object(
                    brain_runtime,
                    "recall_weak_concepts",
                    new=AsyncMock(return_value=json.dumps({"memories": [{"concept": "softmax"}]})),
                ) as recall,
                patch.object(brain_runtime, "search_course_materials") as pdf,
            ):
                context = await brain_runtime.prefetch_memory_context(
                    " softmax scaling ", "student-a"
                )

            recall.assert_awaited_once_with("softmax scaling", "student-a")
            pdf.assert_not_called()
            self.assertEqual(context["student_question"], "softmax scaling")
            self.assertEqual(context["weak_concepts"]["memories"][0]["concept"], "softmax")

        asyncio.run(scenario())

    def test_text_prefetch_still_runs_memory_and_pdf_in_parallel(self) -> None:
        async def scenario() -> None:
            with (
                patch.object(
                    brain_runtime,
                    "recall_weak_concepts",
                    new=AsyncMock(return_value=json.dumps({"memories": [{"concept": "softmax"}]})),
                ) as recall,
                patch.object(
                    brain_runtime,
                    "search_course_materials",
                    return_value=json.dumps({"found": True, "results": [{"source": "week3.pdf p.4"}]}),
                ) as pdf,
            ):
                context = await brain_runtime.prefetch_context(
                    " softmax scaling ", student_id="student-a"
                )

            recall.assert_awaited_once_with("softmax scaling", "student-a")
            pdf.assert_called_once_with("softmax scaling")
            self.assertTrue(context["course_materials"]["found"])

        asyncio.run(scenario())

    def test_voice_persona_has_memory_context_and_filler_backed_pdf_tool(self) -> None:
        prompt = agent_spec.persona(
            "socratic", {"found": True, "memories": [{"concept": "정상성"}]}
        )
        self.assertIn("There is\nno learner-memory retrieval tool", prompt)
        self.assertIn("call search_course_materials before the final answer", prompt)
        self.assertIn("Immediately before calling search_course_materials", prompt)
        self.assertIn("정상성", prompt)
        self.assertIn("Socratic mode:", prompt)


if __name__ == "__main__":
    unittest.main()

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


class CourseToolInstructionTests(unittest.TestCase):
    def test_successful_course_search_adds_socratic_and_visual_guidance(self) -> None:
        raw_result = {
            "found": True,
            "query": "scaled dot product attention",
            "retrieval_mode": "hybrid",
            "results": [
                {
                    "source": "lecture06_Transformer_Part1.pdf p.8",
                    "excerpt": "attention scaling excerpt",
                    "score": 0.91,
                }
            ],
            "instruction": "Use filename and page in the answer.",
        }

        with patch.object(
            agent_spec,
            "run_brain_tool",
            new=AsyncMock(return_value=json.dumps(raw_result, ensure_ascii=False)),
        ):
            result = asyncio.run(
                agent_spec.run_tool(
                    "search_course_materials",
                    {"query": "scaled dot product attention"},
                )
            )

        self.assertEqual(result["results"][0]["file"], "lecture06_Transformer_Part1.pdf")
        self.assertEqual(result["results"][0]["page"], 8)
        self.assertIn("Do not reveal", result["instruction"]["teaching"])
        self.assertIn("exactly one short reasoning question", result["instruction"]["teaching"])
        self.assertIn("show_visualization", result["instruction"]["visualization"])
        for kind in ("pdf", "formula", "flow", "plot"):
            self.assertIn(kind, result["instruction"]["visualization"])

    def test_missing_course_evidence_keeps_web_fallback_instruction(self) -> None:
        raw_result = {
            "found": False,
            "query": "unknown topic",
            "results": [],
            "instruction": "No PDF evidence found; trusted web search is now allowed.",
        }

        with patch.object(
            agent_spec,
            "run_brain_tool",
            new=AsyncMock(return_value=json.dumps(raw_result, ensure_ascii=False)),
        ):
            result = asyncio.run(
                agent_spec.run_tool("search_course_materials", {"query": "unknown topic"})
            )

        self.assertEqual(result, raw_result)

    def test_voice_prompt_defers_tool_result_behavior_to_course_instruction(self) -> None:
        persona = agent_spec.persona("socratic")
        self.assertIn(
            "Follow the teaching, grounding, and visualization instructions returned by",
            persona,
        )
        self.assertIn("Only immediately before calling", persona)
        self.assertIn("Do not say a\nfiller before show_visualization", persona)


if __name__ == "__main__":
    unittest.main()

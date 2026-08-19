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
        for kind in ("pdf", "formula", "flow"):
            self.assertIn(kind, result["instruction"]["visualization"])
        self.assertNotIn("plot", result["instruction"]["visualization"])

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

    def test_successful_trusted_web_search_adds_socratic_and_visual_guidance(self) -> None:
        raw_result = {
            "found": True,
            "answer": "trusted evidence summary",
            "sources": ["https://arxiv.org/abs/1234.5678"],
            "instruction": "The UI displays source URLs separately.",
        }

        with patch.object(
            agent_spec,
            "run_brain_tool",
            new=AsyncMock(return_value=json.dumps(raw_result, ensure_ascii=False)),
        ):
            result = asyncio.run(
                agent_spec.run_tool(
                    "search_trusted_web",
                    {"query": "attention scaling", "reason": "course evidence insufficient"},
                )
            )

        instruction = result["instruction"]
        self.assertIn("trusted-web evidence", instruction["grounding"])
        self.assertIn("Do not reveal", instruction["teaching"])
        self.assertIn("exactly one short reasoning question", instruction["teaching"])
        self.assertIn("show_visualization", instruction["visualization"])
        for kind in ("formula", "flow"):
            self.assertIn(kind, instruction["visualization"])
        self.assertNotIn("plot", instruction["visualization"])
        self.assertIn("Do not read or repeat raw URLs", instruction["sources"])

    def test_failed_trusted_web_search_is_not_enriched(self) -> None:
        raw_result = {
            "error": "trusted web search returned no citable sources",
            "query": "unknown topic",
        }

        with patch.object(
            agent_spec,
            "run_brain_tool",
            new=AsyncMock(return_value=json.dumps(raw_result, ensure_ascii=False)),
        ):
            result = asyncio.run(
                agent_spec.run_tool(
                    "search_trusted_web",
                    {"query": "unknown topic", "reason": "course evidence insufficient"},
                )
            )

        self.assertEqual(result, raw_result)

    def test_voice_visualization_schema_uses_three_minimal_variants(self) -> None:
        visual = next(
            tool for tool in agent_spec.json_schemas()
            if tool["name"] == "show_visualization"
        )
        variants = visual["parameters"]["oneOf"]
        self.assertEqual(
            {variant["properties"]["kind"]["const"] for variant in variants},
            {"formula", "flow", "pdf"},
        )

        required_by_kind = {
            variant["properties"]["kind"]["const"]: set(variant["required"])
            for variant in variants
        }
        self.assertEqual(
            required_by_kind["formula"],
            {"kind", "title", "caption", "latex"},
        )
        self.assertEqual(
            required_by_kind["flow"],
            {"kind", "title", "caption", "labels"},
        )
        self.assertEqual(
            required_by_kind["pdf"],
            {"kind", "title", "caption", "file", "page"},
        )
        schema_text = json.dumps(visual, ensure_ascii=False)
        for removed in ("plot", "points", "x_label", "y_label"):
            self.assertNotIn(removed, schema_text)

    def test_voice_prompt_defers_search_behavior_to_tool_result_instructions(self) -> None:
        persona = agent_spec.persona("socratic")
        self.assertIn("Follow instructions returned by search tools", persona)
        self.assertIn("Only immediately before calling", persona)
        self.assertIn("filler before show_visualization", persona)


if __name__ == "__main__":
    unittest.main()

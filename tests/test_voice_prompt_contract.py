from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ["VOICE_AI_SKIP_DOTENV"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent_spec
from brain import MODE_PROMPTS


class VoicePromptContractTests(unittest.TestCase):
    def test_always_on_voice_prompt_stays_compact_but_strong_on_visuals(self) -> None:
        prompt = agent_spec.VOICE_SYSTEM_PROMPT
        self.assertLess(len(prompt), 1200)
        self.assertIn("Follow instructions returned by search tools", prompt)
        self.assertIn("Visualization is part of teaching, not decoration", prompt)
        self.assertIn("MUST call show_visualization before responding", prompt)
        self.assertIn("without\nrevealing the final answer", prompt)
        self.assertNotIn("Greek letter", prompt)
        self.assertNotIn("fraction", prompt)
        self.assertNotIn("plot", prompt)

    def test_socratic_mode_keeps_strong_non_revealing_behavior(self) -> None:
        prompt = MODE_PROMPTS["socratic"]
        self.assertIn("never explain or summarize the answer", prompt)
        self.assertIn("Ask exactly one question", prompt)
        self.assertIn("give one minimal hint", prompt)
        self.assertIn("Give a direct explanation only after", prompt)
        self.assertIn("End every response with exactly one question", prompt)
        self.assertIn("Maximum two short spoken sentences", prompt)

    def test_search_results_still_reinforce_socratic_behavior(self) -> None:
        for instruction in (
            agent_spec.COURSE_RESULT_INSTRUCTION,
            agent_spec.WEB_RESULT_INSTRUCTION,
        ):
            teaching = instruction["teaching"]
            self.assertIn("Do not reveal", teaching)
            self.assertIn("one minimal hint", teaching)
            self.assertIn("show_visualization", instruction["visualization"])
            self.assertIn("next reasoning step", instruction["visualization"])


if __name__ == "__main__":
    unittest.main()

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
    def test_always_on_voice_prompt_stays_compact_and_prefers_visuals(self) -> None:
        prompt = agent_spec.VOICE_SYSTEM_PROMPT
        self.assertLess(len(prompt), 1200)
        self.assertIn("Follow instructions returned by search tools", prompt)
        self.assertIn("Visualization is part of teaching, not decoration", prompt)
        self.assertIn("Prefer show_visualization", prompt)
        self.assertNotIn("MUST call show_visualization", prompt)
        self.assertNotIn("MUST use show_visualization", prompt)
        self.assertIn("not the\nfinal answer", prompt)
        self.assertIn("MUST be Korean", prompt)
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

    def test_realtime_socratic_prompt_treats_visual_as_clue(self) -> None:
        prompt = agent_spec.persona("socratic")
        self.assertIn("# Socratic visual priority", prompt)
        self.assertIn("MUST call show_visualization before the one Socratic", prompt)
        self.assertIn("no-answer/worked-example rule applies to speech", prompt)
        self.assertLess(
            prompt.index("# Socratic visual priority"),
            prompt.index("# Recent weak concepts"),
        )
        self.assertNotIn("# Socratic visual priority", agent_spec.persona("explain"))

    def test_search_results_only_remind_the_current_mode_and_visual_preference(self) -> None:
        for instruction in (
            agent_spec.COURSE_RESULT_INSTRUCTION,
            agent_spec.WEB_RESULT_INSTRUCTION,
        ):
            teaching = instruction["teaching"]
            self.assertIn("Keep the current teaching mode", teaching)
            self.assertIn("one next-step question", teaching)
            self.assertNotIn("Give no hint", teaching)
            self.assertNotIn("wrong answer", teaching)
            self.assertIn("Prefer show_visualization", instruction["visualization"])


if __name__ == "__main__":
    unittest.main()

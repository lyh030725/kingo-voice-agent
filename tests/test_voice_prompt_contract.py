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
    def test_always_on_voice_prompt_stays_compact(self) -> None:
        self.assertLess(len(agent_spec.VOICE_SYSTEM_PROMPT), 1200)
        self.assertIn("Follow the teaching, grounding, and visualization instructions", agent_spec.VOICE_SYSTEM_PROMPT)
        self.assertIn("Use show_visualization when a visual materially helps learning", agent_spec.VOICE_SYSTEM_PROMPT)
        self.assertNotIn("Greek letter", agent_spec.VOICE_SYSTEM_PROMPT)
        self.assertNotIn("fraction", agent_spec.VOICE_SYSTEM_PROMPT)

    def test_socratic_mode_keeps_strong_non_revealing_behavior(self) -> None:
        prompt = MODE_PROMPTS["socratic"]
        self.assertIn("never explain or summarize the answer", prompt)
        self.assertIn("Ask exactly one question", prompt)
        self.assertIn("give one minimal hint", prompt)
        self.assertIn("Give a direct explanation only after", prompt)
        self.assertIn("End every response with exactly one question", prompt)
        self.assertIn("Maximum two short spoken sentences", prompt)

    def test_course_result_still_reinforces_socratic_behavior(self) -> None:
        teaching = agent_spec.COURSE_RESULT_INSTRUCTION["teaching"]
        self.assertIn("Do not reveal, summarize, or paraphrase", teaching)
        self.assertIn("Give no hint on the first attempt", teaching)
        self.assertIn("one minimal hint", teaching)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ["VOICE_AI_SKIP_DOTENV"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import brain
import server


class MvpTests(unittest.TestCase):
    def test_professor_can_upload_pdf_and_refresh_search_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
            brain, "COURSE_SRCS_DIR", Path(directory)
        ):
            brain.PDF_PAGE_CACHE = [{"stale": True}]
            saved = brain.add_course_material("week-03.pdf", b"%PDF-1.4\nlesson")

            self.assertEqual(saved["name"], "week-03.pdf")
            self.assertEqual(brain.list_course_materials(), [saved])
            self.assertIsNone(brain.PDF_PAGE_CACHE)
            with self.assertRaises(ValueError):
                brain.add_course_material("../secret.pdf", b"%PDF-1.4")
            with self.assertRaises(ValueError):
                brain.add_course_material("notes.txt", b"notes")

    def test_professor_can_add_and_delete_trusted_sites(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
            brain, "TRUSTED_SITES_FILE", Path(directory) / "trusted-sites.json"
        ):
            sites = brain.add_trusted_domain("https://KOSIS.kr/statHtml")
            self.assertIn("kosis.kr", sites)
            self.assertNotIn("kosis.kr", brain.remove_trusted_domain("kosis.kr"))
            with self.assertRaises(ValueError):
                brain.add_trusted_domain("not-a-domain")

    def test_text_chat_forwards_selected_mode(self) -> None:
        mocked = AsyncMock(return_value=("풀이", ["search_course_materials"], ["https://example.com/lesson"]))
        with patch.object(server, "think", mocked):
            result = asyncio.run(
                server.answer_text(brain.TextQuestion(text="ARIMA를 풀어줘", mode="explain"))
            )

        self.assertEqual(result["reply"], "풀이")
        self.assertEqual(result["sources"], ["https://example.com/lesson"])
        self.assertEqual(mocked.await_args.args[2], "explain")
        self.assertEqual(server.VALID_MODES, {"explain", "socratic"})


if __name__ == "__main__":
    unittest.main()

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
    def test_system_prompt_prefers_spoken_korean(self) -> None:
        self.assertIn("polite 해요 style", brain.SYSTEM_PROMPT)
        self.assertIn("Avoid written declarative endings", brain.SYSTEM_PROMPT)
        self.assertIn("textbook or report-like prose", brain.SYSTEM_PROMPT)

    def test_server_lifespan_survives_moss_quota(self) -> None:
        class QuotaClient:
            def __init__(self, _project_id, _project_key):
                pass

            async def session(self, **_kwargs):
                raise RuntimeError("Moss usage limit exceeded")

        async def scenario(store) -> None:
            with (
                patch.object(brain, "MOSS_MEMORY", store),
                patch.object(brain, "_pdf_pages", return_value=[]),
            ):
                async with server.lifespan(server.app):
                    self.assertTrue(store.is_local_mode)

        with tempfile.TemporaryDirectory() as directory:
            store = brain.MossMemoryStore(
                "project-1",
                "key-1",
                local_path=Path(directory) / "weak-concepts.json",
                sdk_loader=lambda: type("Sdk", (), {"MossClient": QuotaClient}),
            )
            asyncio.run(scenario(store))

    def test_course_agent_symbol_states_are_wired(self) -> None:
        root = Path(__file__).resolve().parents[1]
        component = (root / "static" / "course-agent-symbol.js").read_text(encoding="utf-8")
        page = (root / "static" / "index.html").read_text(encoding="utf-8")

        for state in ("presence", "resonance", "flow", "bloom", "sustain", "error"):
            self.assertIn(f'"{state}"', component)
        self.assertIn("prefers-reduced-motion: reduce", component)
        self.assertIn("animation: none !important", component)
        self.assertIn('id="course-agent-petal-gradient"', component)
        self.assertIn("@keyframes course-agent-grow", component)
        self.assertIn('class="growth-petal"', component)
        self.assertIn('id="course-agent-fan-mask"', component)
        self.assertIn("transform-origin: 50px 92px", component)
        self.assertIn('customElements.define("course-agent-symbol"', component)
        self.assertIn('setSymbolState(message, "sustain"), 500', page)
        self.assertIn('setSymbolState(message, "flow"), 300', page)
        self.assertIn('state="presence" size="32"', page)
        self.assertIn('state="presence" size="16"', page)

    def test_professor_can_upload_pdf_and_refresh_search_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
            brain, "COURSE_SRCS_DIR", Path(directory)
        ):
            brain.PDF_PAGE_CACHE = [{"stale": True}]
            saved = brain.add_course_material("week-03.pdf", b"%PDF-1.4\nlesson")
            second = brain.add_course_material("week-04.pdf", b"%PDF-1.4\nlesson")

            self.assertEqual(saved["name"], "week-03.pdf")
            self.assertEqual(brain.list_course_materials(), [saved, second])
            brain.PDF_PAGE_CACHE = [{"stale": True}]
            brain.remove_course_material(saved["name"])
            self.assertEqual(brain.list_course_materials(), [second])
            self.assertIsNone(brain.PDF_PAGE_CACHE)
            with self.assertRaises(ValueError):
                brain.add_course_material("../secret.pdf", b"%PDF-1.4")
            with self.assertRaises(ValueError):
                brain.add_course_material("notes.txt", b"notes")
            with self.assertRaises(FileNotFoundError):
                brain.remove_course_material("missing.pdf")

    def test_professor_multi_pdf_dropzone_is_wired(self) -> None:
        page = (Path(__file__).resolve().parents[1] / "static" / "index.html").read_text(encoding="utf-8")

        self.assertIn('id="material-dropzone"', page)
        self.assertIn('type="file" accept="application/pdf,.pdf" multiple', page)
        self.assertIn('materialDropzone.addEventListener("drop"', page)
        self.assertIn("selectedMaterialFiles", page)
        self.assertIn("for (const [index, file] of files.entries())", page)
        self.assertIn('id="cancel-material"', page)
        self.assertIn('id="selected-material-list"', page)
        self.assertIn("removeSelectedMaterial(index)", page)
        self.assertIn('remove.textContent = "선택 취소"', page)
        self.assertIn('method: "DELETE"', page)

    def test_trusted_sites_render_as_regular_links(self) -> None:
        page = (Path(__file__).resolve().parents[1] / "static" / "index.html").read_text(encoding="utf-8")

        self.assertIn('document.createElement("a")', page)
        self.assertIn('domain.href = "https://" + site', page)
        self.assertIn('domain.rel = "noopener noreferrer"', page)
        self.assertIn(".site-link {", page)
        self.assertNotIn('document.createElement("code")', page)

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

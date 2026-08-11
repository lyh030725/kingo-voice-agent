from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pypdf import PdfWriter

os.environ["VOICE_AI_SKIP_DOTENV"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import brain
import server


class MvpTests(unittest.TestCase):
    def test_system_prompt_prefers_spoken_korean(self) -> None:
        for heading in (
            "# Role",
            "# Language and style",
            "# Tool usage",
            "# Evidence and source rules",
            "# Visualization rules",
            "# Output format",
        ):
            self.assertIn(heading, brain.SYSTEM_PROMPT)
        for tool_name in (
            "recall_weak_concepts",
            "search_course_materials",
            "search_trusted_web",
            "save_weak_concept",
            "review_weak_concept",
            "show_visualization",
        ):
            self.assertIn(f"{tool_name}:", brain.SYSTEM_PROMPT)
        self.assertIn("Sungkyunkwan University student", brain.SYSTEM_PROMPT)
        self.assertIn("Speak in Korean unless asked otherwise", brain.SYSTEM_PROMPT)
        self.assertIn("At the start of every student turn", brain.SYSTEM_PROMPT)
        self.assertIn("Base factual claims only on tool results", brain.SYSTEM_PROMPT)
        self.assertIn("Return source URLs through the separate sources field", brain.SYSTEM_PROMPT)
        self.assertIn("Ordinary questions are not weaknesses", brain.SYSTEM_PROMPT)
        self.assertIn("memory id", brain.SYSTEM_PROMPT)
        self.assertIn("one to three short conversational sentences", brain.SYSTEM_PROMPT)
        self.assertIn("Never\nread raw JSON aloud", brain.SYSTEM_PROMPT)
        self.assertIn("polite 해요 style", brain.SYSTEM_PROMPT)
        self.assertIn("Avoid written declarative endings", brain.SYSTEM_PROMPT)
        self.assertIn("textbook or report-like prose", brain.SYSTEM_PROMPT)
        self.assertIn("Never put raw equations", brain.SYSTEM_PROMPT)
        self.assertIn("show_visualization first", brain.SYSTEM_PROMPT)
        self.assertIn("When unsure whether visual support is useful, prefer calling", brain.SYSTEM_PROMPT)
        self.assertIn("Do not send the final conversational answer until that tool", brain.SYSTEM_PROMPT)
        self.assertIn("paraphrase any equation instead of quoting or reading it", brain.SYSTEM_PROMPT)
        self.assertIn("repeat its LaTeX, symbols, equation, or coordinates", brain.SYSTEM_PROMPT)
        self.assertIn("제가 보여드린 그림처럼", brain.SYSTEM_PROMPT)

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

    def test_visualizations_are_rendered_without_raw_html(self) -> None:
        page = (Path(__file__).resolve().parents[1] / "static" / "index.html").read_text(encoding="utf-8")

        self.assertIn("mathjax@3.2.2/es5/tex-chtml.js", page)
        self.assertIn("function addVisualization(visualization)", page)
        self.assertIn("function buildPlot(visualization, points)", page)
        self.assertIn('message.type === "visualization"', page)
        self.assertIn('message.type === "visualization_error"', page)
        self.assertIn('formula.textContent = "$$" + visualization.latex + "$$"', page)
        self.assertIn('["formula", "flow", "plot", "pdf"]', page)
        self.assertIn("pdfjs-dist@6.2.108/build/pdf.min.mjs", page)
        self.assertIn("async function buildPdfPage", page)
        self.assertIn("await pdf.getPage(visualization.page)", page)
        self.assertIn("window.devicePixelRatio || 1", page)
        self.assertIn('document.createElement("canvas")', page)
        self.assertIn('encodeURIComponent(visualization.file)', page)
        self.assertNotIn('document.createElement("iframe")', page)

    def test_course_material_pdf_is_served_inline(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
            brain, "COURSE_SRCS_DIR", Path(directory)
        ):
            pdf_path = Path(directory) / "week-04.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=612, height=792)
            with pdf_path.open("wb") as stream:
                writer.write(stream)

            response = asyncio.run(server.course_material_file("week-04.pdf"))

            self.assertEqual(Path(response.path), pdf_path)
            self.assertEqual(response.media_type, "application/pdf")
            self.assertTrue(response.headers["content-disposition"].startswith("inline;"))
            with self.assertRaises(server.HTTPException) as missing:
                asyncio.run(server.course_material_file("missing.pdf"))
            self.assertEqual(missing.exception.status_code, 404)

    def test_professor_can_add_and_delete_trusted_sites(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
            brain, "TRUSTED_SITES_FILE", Path(directory) / "trusted-sites.json"
        ):
            sites = brain.add_trusted_domain("https://KOSIS.kr/statHtml")
            self.assertIn("kosis.kr", sites)
            self.assertNotIn("kosis.kr", brain.remove_trusted_domain("kosis.kr"))
            with self.assertRaises(ValueError):
                brain.add_trusted_domain("not-a-domain")

    def test_wikipedia_is_a_default_trusted_site(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
            brain, "TRUSTED_SITES_FILE", Path(directory) / "trusted-sites.json"
        ):
            self.assertIn("wikipedia.org", brain.get_trusted_domains())
            response = SimpleNamespace(
                model_dump=lambda: {},
                output_text="https://ko.wikipedia.org/wiki/시계열",
            )
            self.assertEqual(
                brain._trusted_urls(response),
                ["https://ko.wikipedia.org/wiki/시계열"],
            )

    def test_text_chat_forwards_selected_mode(self) -> None:
        mocked = AsyncMock(return_value=("풀이", ["search_course_materials"], ["https://example.com/lesson"], []))
        with patch.object(server, "think", mocked):
            result = asyncio.run(
                server.answer_text(brain.TextQuestion(text="ARIMA를 풀어줘", mode="explain"))
            )

        self.assertEqual(result["reply"], "풀이")
        self.assertEqual(result["sources"], ["https://example.com/lesson"])
        self.assertEqual(result["visualizations"], [])
        self.assertEqual(mocked.await_args.args[2], "explain")
        self.assertEqual(server.VALID_MODES, {"explain", "socratic"})


    def test_text_chat_streams_tokens_before_done(self) -> None:
        async def fake_think(_text, _timer, _mode, on_token=None):
            await on_token("첫 ")
            await on_token("토큰")
            return "첫 토큰", [], [], []

        async def scenario():
            with patch.object(server, "think", fake_think):
                response = await server.answer_text_stream(
                    brain.TextQuestion(text="질문", mode="explain")
                )
                body = "".join([chunk async for chunk in response.body_iterator])
            return [json.loads(line) for line in body.splitlines()]

        events = asyncio.run(scenario())

        self.assertEqual([event["type"] for event in events], ["token", "token", "done"])
        self.assertEqual([event["text"] for event in events[:2]], ["첫 ", "토큰"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ["VOICE_AI_SKIP_DOTENV"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pdf_retrieval
import session_state


class SessionStateTests(unittest.TestCase):
    def setUp(self) -> None:
        session_state._HISTORIES.clear()
        session_state._MEMORIES.clear()
        session_state._EXTERNAL_BRAINS.clear()

    def test_histories_are_isolated_by_student_and_session(self) -> None:
        a1 = session_state.history_for("student-a", "session-1")
        a2 = session_state.history_for("student-a", "session-2")
        b1 = session_state.history_for("student-b", "session-1")
        a1.append({"role": "user", "content": "A"})

        self.assertEqual(a2, [])
        self.assertEqual(b1, [])
        self.assertEqual(session_state.history_for("student-a", "session-1")[0]["content"], "A")

        session_state.clear_history("student-a", "session-1")
        self.assertEqual(session_state.history_for("student-a", "session-1"), [])

    def test_memory_store_is_shared_per_student_but_not_across_students(self) -> None:
        first = session_state.memory_for("student-a")
        same = session_state.memory_for("student-a")
        other = session_state.memory_for("student-b")

        self.assertIs(first, same)
        self.assertIsNot(first, other)
        self.assertEqual(first.student_id, "student-a")
        self.assertEqual(other.student_id, "student-b")
        self.assertNotEqual(first.local_path, other.local_path)


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        pdf_retrieval.clear_embedding_cache()

    def test_semantic_similarity_can_recover_zero_lexical_overlap(self) -> None:
        pages = [
            {"file": "a.pdf", "page": 1, "text": "stationarity and differencing"},
            {"file": "b.pdf", "page": 2, "text": "attention mechanism"},
        ]
        vectors = {
            "stationarity and differencing": [1.0, 0.0],
            "attention mechanism": [0.0, 1.0],
            "추세를 제거하면 왜 정상적이 돼": [1.0, 0.0],
        }

        def create(*, model, input):
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=vectors[text]) for text in input]
            )

        client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
        with patch.dict(os.environ, {"PDF_LEXICAL_WEIGHT": "0.4"}, clear=False):
            ranked, mode = pdf_retrieval.hybrid_rank(
                "추세를 제거하면 왜 정상적이 돼",
                pages,
                [0.0, 0.0],
                lambda: client,
            )

        self.assertEqual(mode, "hybrid")
        self.assertEqual(ranked[0][1]["file"], "a.pdf")

    def test_embedding_failure_falls_back_to_lexical_ranking(self) -> None:
        pages = [
            {"file": "a.pdf", "page": 1, "text": "softmax"},
            {"file": "b.pdf", "page": 2, "text": "attention"},
        ]

        def broken_client():
            raise RuntimeError("embedding unavailable")

        ranked, mode = pdf_retrieval.hybrid_rank(
            "softmax", pages, [3.0, 1.0], broken_client
        )

        self.assertEqual(mode, "lexical")
        self.assertEqual(ranked[0][1]["file"], "a.pdf")


if __name__ == "__main__":
    unittest.main()

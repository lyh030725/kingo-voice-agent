from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moss_memory import MossMemoryStore


class FakeDocumentInfo:
    def __init__(self, id, text, metadata=None, embedding=None, payload=None):
        self.id = id
        self.text = text
        self.metadata = metadata
        self.embedding = embedding
        self.payload = payload


class FakeGetDocumentsOptions:
    def __init__(self, doc_ids=None):
        self.doc_ids = doc_ids


class FakeQueryOptions:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeSession:
    def __init__(self):
        self.documents = {}
        self.last_query = None
        self.push_count = 0

    @property
    def doc_count(self):
        return len(self.documents)

    async def get_docs(self, options=None):
        if options and options.doc_ids is not None:
            return [
                self.documents[doc_id]
                for doc_id in options.doc_ids
                if doc_id in self.documents
            ]
        return list(self.documents.values())

    async def add_docs(self, docs):
        added = 0
        updated = 0
        for doc in docs:
            if doc.id in self.documents:
                updated += 1
            else:
                added += 1
            self.documents[doc.id] = doc
        return added, updated

    async def query(self, query, options):
        self.last_query = (query, options)
        return SimpleNamespace(docs=list(self.documents.values())[: options.top_k])

    async def push_index(self):
        self.push_count += 1
        return SimpleNamespace(doc_count=self.doc_count, status="completed")


class FakeClient:
    def __init__(self, project_id, project_key, session):
        self.project_id = project_id
        self.project_key = project_key
        self._session = session
        self.session_args = None

    async def session(self, *, index_name, model_id):
        self.session_args = (index_name, model_id)
        return self._session


class MossMemoryStoreTests(unittest.TestCase):
    def test_embed_deduplicate_recall_and_push(self):
        async def scenario(local_path):
            session = FakeSession()
            clients = []

            def make_client(project_id, project_key):
                client = FakeClient(project_id, project_key, session)
                clients.append(client)
                return client

            sdk = SimpleNamespace(
                DocumentInfo=FakeDocumentInfo,
                GetDocumentsOptions=FakeGetDocumentsOptions,
                MossClient=make_client,
                QueryOptions=FakeQueryOptions,
            )
            store = MossMemoryStore(
                "project-1",
                "key-1",
                index_name="student-memory",
                model_id="moss-minilm",
                student_id="student-1",
                sync_debounce_seconds=0,
                local_path=local_path,
                sdk_loader=lambda: sdk,
            )

            first = await store.save(
                "AI 개론",
                "Self-Attention",
                "Query와 Key를 왜 곱하나요?",
                "유사도 점수의 의미가 불명확함",
            )
            duplicate = await store.save(
                "AI 개론",
                "Self-Attention",
                "Query와 Key를 왜 곱하나요?",
                "중복 저장되지 않아야 함",
            )
            recalled = await store.recall("attention 유사도")
            stored = session.documents[first["memory_id"]]
            payload = json.loads(stored.payload)
            payload["next_review_at"] = 0
            stored.payload = json.dumps(payload)
            due = await store.next_review()
            failed = await store.review(first["memory_id"], False)
            await store.review(first["memory_id"], True)
            await store.review(first["memory_id"], True)
            mastered = await store.review(first["memory_id"], True)
            await store.flush()

            self.assertEqual(first["status"], "saved")
            self.assertEqual(duplicate["status"], "updated")
            self.assertEqual(len(session.documents), 1)
            self.assertEqual(recalled["memories"][0]["memory_id"], first["memory_id"] )
            self.assertEqual(recalled["memories"][0]["failure_count"], 2)
            self.assertFalse(failed["correct"])
            self.assertEqual(due["id"], first["memory_id"])
            self.assertEqual(mastered["status"], "mastered")
            self.assertEqual(clients[0].session_args, ("student-memory", "moss-minilm"))
            self.assertEqual(session.last_query[0], "attention 유사도")
            self.assertEqual(session.last_query[1].alpha, 0.85)
            self.assertEqual(
                session.last_query[1].filter["condition"],
                {"$eq": "student-1"},
            )
            self.assertEqual(session.push_count, 1)
            self.assertEqual(
                json.loads(local_path.read_text(encoding="utf-8"))[0]["concept"],
                "Self-Attention",
            )

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(scenario(Path(directory) / "weak-concepts.json"))

    def test_quota_uses_local_memory_without_retrying_moss(self):
        async def scenario(local_path):
            calls = 0

            class QuotaClient:
                def __init__(self, _project_id, _project_key):
                    pass

                async def session(self, **_kwargs):
                    nonlocal calls
                    calls += 1
                    raise RuntimeError("Moss usage limit exceeded")

            store = MossMemoryStore(
                "project-1",
                "key-1",
                student_id="student-1",
                local_path=local_path,
                sdk_loader=lambda: SimpleNamespace(MossClient=QuotaClient),
            )

            await store.initialize()
            saved = await store.save(
                "AI 개론",
                "Self-Attention",
                "Query와 Key를 왜 곱하나요?",
                "attention 유사도의 의미가 불명확함",
            )
            recalled = await store.recall("attention 유사도")
            reviewed = await store.review(saved["memory_id"], True)

            self.assertTrue(store.is_local_mode)
            self.assertEqual(calls, 1)
            self.assertEqual(saved["storage"], "local")
            self.assertEqual(recalled["storage"], "local")
            self.assertEqual(recalled["memories"][0]["concept"], "Self-Attention")
            self.assertEqual(reviewed["storage"], "local")
            self.assertTrue(local_path.exists())

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(scenario(Path(directory) / "weak-concepts.json"))

    def test_missing_credentials_fail_before_sdk_load(self):
        async def scenario():
            with patch.dict(
                os.environ,
                {"MOSS_PROJECT_ID": "", "MOSS_PROJECT_KEY": ""},
                clear=False,
            ):
                store = MossMemoryStore(
                    project_id="",
                    project_key="",
                    sdk_loader=lambda: self.fail(
                        "SDK must not load without credentials"
                    ),
                )
                with self.assertRaisesRegex(RuntimeError, "MOSS_PROJECT_ID"):
                    await store.initialize()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()

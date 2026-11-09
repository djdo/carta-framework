import unittest
from unittest.mock import MagicMock, patch
from carta_framework.vector_store import ChromaVectorStore
from carta_framework.models import TemporalScope

class TestVectorStore(unittest.TestCase):
    @patch("carta_framework.vector_store.chroma.chromadb.PersistentClient")
    def setUp(self, MockClient):
        self.mock_client = MockClient.return_value
        self.mock_collection = MagicMock()
        self.mock_client.get_or_create_collection.return_value = self.mock_collection
        self.mock_llm = MagicMock()
        # Initialize ChromaVectorStore
        self.store = ChromaVectorStore(
            llm_provider=self.mock_llm,
            persist_directory="dummy", 
            collection_name="test_col"
        )

    def test_add_documents(self):
        self.store.add_documents(
            documents=["test doc"],
            metadatas=[{"source": "test.txt"}],
            tenant_id="tenant-123",
            available_from_ts=500.0
        )
        self.mock_collection.add.assert_called_once()
        args, kwargs = self.mock_collection.add.call_args
        
        # Verify ids exist
        self.assertIn("ids", kwargs)
        # Verify metadata injection
        self.assertEqual(kwargs["metadatas"][0]["_carta_tenant_id"], "tenant-123")
        self.assertEqual(kwargs["metadatas"][0]["_carta_available_from_ts"], 500.0)

    def test_add_documents_leaves_caller_metadata_unchanged(self):
        metadatas = [{"source": "test.txt"}]
        self.store.add_documents(["test doc"], metadatas=metadatas, tenant_id="tenant-123")
        self.assertEqual(metadatas, [{"source": "test.txt"}])

    def test_get_unique_sources_pages_through_collection(self):
        pages = [
            {"metadatas": [{"source": "a/x.pdf"}, {"source": "y.pdf"}]},
            {"metadatas": [{"source": "x.pdf"}]},
        ]
        self.mock_collection.get.side_effect = pages
        self.assertEqual(self.store.get_unique_sources(page_size=2), ["x.pdf", "y.pdf"])
        self.assertEqual([c.kwargs["offset"] for c in self.mock_collection.get.call_args_list], [0, 2])

    def test_get_unique_sources_raises_store_errors(self):
        self.mock_collection.get.side_effect = RuntimeError("store down")
        with self.assertRaises(RuntimeError):
            self.store.get_unique_sources()

    def test_query_filtering(self):
        self.mock_collection.query.return_value = {"documents": [["res1"]], "metadatas": [[{}]], "distances": [[0.5]]}
        res = self.store.query(
            "hello",
            n_results=1,
            tenant_id="tenant-123",
            temporal_scope=TemporalScope(horizon_timestamp=1000.0)
        )
        
        self.mock_collection.query.assert_called_once()
        args, kwargs = self.mock_collection.query.call_args
        
        # Verify the where clause correctly builds $and condition
        expected_where = {
            "$and": [
                {"_carta_tenant_id": {"$eq": "tenant-123"}},
                {"_carta_available_from_ts": {"$lte": 1000.0}}
            ]
        }
        self.assertEqual(kwargs["where"], expected_where)


class TestEmbeddingSignature(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.dir = tempfile.mkdtemp()
        self.llm = MagicMock()
        self.llm.generate_embeddings.side_effect = lambda texts: [[0.1, 0.2, 0.3] for _ in texts]

    def _store(self, signature):
        from carta_framework.vector_store import ChromaVectorStore
        return ChromaVectorStore(self.llm, self.dir, "sig_test", embedding_signature=signature)

    def test_index_from_another_model_is_refused_until_cleared(self):
        self._store("google:a").add_documents(["doc"], metadatas=[{"source": "x"}])

        store = self._store("openai:b")
        self.assertIn("Re-ingest", store.embedding_mismatch)
        with self.assertRaises(RuntimeError):
            store.query("q")
        with self.assertRaises(RuntimeError):
            store.add_documents(["more"])

        store.clear()
        self.assertIsNone(store.embedding_mismatch)
        store.add_documents(["fresh"], metadatas=[{"source": "y"}])
        self.assertEqual(store.query("q")["documents"], ["fresh"])
        self.assertIsNone(self._store("openai:b").embedding_mismatch)

    def test_unsigned_existing_index_adopts_current_model(self):
        self._store(None).add_documents(["doc"], metadatas=[{"source": "x"}])
        self.assertIsNone(self._store("google:a").embedding_mismatch)
        self.assertIsNotNone(self._store("openai:b").embedding_mismatch)

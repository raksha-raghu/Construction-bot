"""Boundary conditions for the retrievers and index lifecycle."""

import unittest
from unittest.mock import Mock, patch

from backend.retrieval.bm25 import BM25Retriever
from backend.retrieval.semantic import search
from backend.storage.vector_store import chunk_text, rebuild_collection


class RetrievalValidationTests(unittest.TestCase):
    def test_empty_dense_collection(self):
        collection = Mock()
        collection.count.return_value = 0
        self.assertEqual(search(collection, "roof"), [])
        collection.query.assert_not_called()

    def test_bad_inputs_do_not_query_database(self):
        collection = Mock()
        for question, depth in [("", 5), (None, 5), ("roof", True), ("roof", -1)]:
            with self.assertRaises(ValueError):
                search(collection, question, depth)
        collection.query.assert_not_called()

    def test_bm25_empty_and_tokenless_corpus(self):
        for documents, error in [([], RuntimeError), (["!!!"], ValueError)]:
            collection = Mock()
            collection.get.return_value = {"ids": [str(i) for i in range(len(documents))], "documents": documents, "metadatas": [{}] * len(documents)}
            with self.assertRaises(error):
                BM25Retriever(collection)

    def test_chunk_boundaries(self):
        self.assertEqual(chunk_text("abcdefgh", 4, 1), ["abcd", "defg", "gh"])
        for size, overlap in [(0, 0), (4, -1), (4, 4)]:
            with self.assertRaises(ValueError):
                chunk_text("abc", size, overlap)

    def test_no_pdfs_preserves_existing_index(self):
        with patch("backend.storage.vector_store.glob.glob", return_value=[]), patch("backend.storage.vector_store.shutil.rmtree") as remove:
            with self.assertRaises(FileNotFoundError):
                rebuild_collection()
            remove.assert_not_called()


if __name__ == "__main__":
    unittest.main()

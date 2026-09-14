import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from backend.console import Query, print_comparison
from backend.retrieval.pipeline import RetrievalPipeline


class Catalog:
    def __init__(self):
        self.get_calls = 0
        self.documents = ["roof membrane", "steel bolt", "wood panel"]

    def get(self, include):
        self.get_calls += 1
        return {
            "ids": [str(i) for i in range(3)],
            "documents": self.documents,
            "metadatas": [
                {"source": f"{i}.pdf", "page": i + 1, "chunk_index": 0}
                for i in range(3)
            ],
        }

    def count(self):
        return len(self.documents)

    def query(self, query_texts, n_results, include):
        # Dense results deliberately differ from the keyword ranking.
        return {
            "ids": [["0"]],
            "documents": [[self.documents[0]]],
            "metadatas": [[{"source": "0.pdf", "page": 1, "chunk_index": 0}]],
            "distances": [[0.2]],
        }


class PipelineTests(unittest.TestCase):
    def test_independent_rankings_and_order(self):
        catalog = Catalog()
        pipeline = RetrievalPipeline(catalog)
        events = []
        dense = catalog.query
        keyword = pipeline.bm25.search

        def dense_call(**kwargs):
            events.append("rag")
            self.assertEqual(kwargs["n_results"], 3)
            return dense(**kwargs)

        def keyword_call(*args, **kwargs):
            events.append("bm25")
            return keyword(*args, **kwargs)

        with patch.object(catalog, "query", side_effect=dense_call), patch.object(
            pipeline.bm25, "search", side_effect=keyword_call
        ):
            result = pipeline.search("steel bolt", top_k=1)
        self.assertEqual(events, ["rag", "bm25"])
        self.assertEqual(result["rag"]["matches"][0]["source"], "0.pdf")
        self.assertEqual(result["bm25"]["matches"][0]["source"], "1.pdf")
        for method in ("rag", "bm25", "hybrid"):
            self.assertGreaterEqual(result[method]["latency_ms"], 0)
        output = io.StringIO()
        with redirect_stdout(output):
            print_comparison(result)
        self.assertLess(output.getvalue().index("RAG /"), output.getvalue().index("BM25 /"))
        self.assertIn("similarity=", output.getvalue())
        self.assertIn("bm25_score=", output.getvalue())
        self.assertIn("rrf_score=", output.getvalue())

    def test_validation_and_small_corpus(self):
        catalog = Catalog()
        pipeline = RetrievalPipeline(catalog)
        for question, top_k in [(" ", 5), ("roof", 0), ("roof", -1)]:
            with self.assertRaises(ValueError):
                pipeline.search(question, top_k)
        with patch.object(catalog, "query", wraps=catalog.query) as dense:
            result = pipeline.search("roof", top_k=100)
        self.assertEqual(dense.call_args.kwargs["n_results"], 3)
        self.assertEqual(len(result["bm25"]["matches"]), 3)

    def test_interactive_reuses_index(self):
        catalog = Catalog()
        output = io.StringIO()
        with patch("builtins.input", side_effect=["roof", "bolt", "exit"]), redirect_stdout(output):
            Query(catalog)
        self.assertEqual(catalog.get_calls, 1)
        self.assertEqual(output.getvalue().count("=== RAG /"), 2)
        self.assertEqual(output.getvalue().count("=== BM25 /"), 2)


if __name__ == "__main__":
    unittest.main()

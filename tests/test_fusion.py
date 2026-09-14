import copy
import unittest
from unittest.mock import patch

from backend.evaluation.runner import evaluate
from backend.retrieval.fusion import fuse_rankings
from backend.retrieval.pipeline import RetrievalPipeline
from test_pipeline import Catalog


def match(chunk_id):
    return {"id": chunk_id, "text": "roof membrane", "source": "roof.pdf",
            "page": 2, "chunk_index": 0}


class FusionTests(unittest.TestCase):
    def test_shared_chunk_overtakes_single_method_winners(self):
        rankings = {"rag": [match("a"), match("shared")],
                    "bm25": [match("b"), match("shared")]}
        original = copy.deepcopy(rankings)
        results = fuse_rankings(rankings, 3)
        self.assertEqual([item["id"] for item in results], ["shared", "a", "b"])
        self.assertAlmostEqual(results[0]["rrf_score"], 2 / 62)
        self.assertEqual(results[0]["retrieval_ranks"], {"rag": 2, "bm25": 2})
        self.assertEqual(rankings, original)

    def test_ids_preserve_distinct_chunks_and_duplicates_count_once(self):
        results = fuse_rankings({"rag": [match("a"), match("a"), match("b")]}, 5)
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0]["rrf_score"], 1 / 61)
        self.assertEqual(fuse_rankings({"rag": [], "bm25": []}, 5), [])
        with self.assertRaises(ValueError):
            fuse_rankings({"rag": [{"text": "missing ID"}]}, 5)
        for top_k, rrf_k in [(0, 60), (5, 0), (True, 60)]:
            with self.assertRaises(ValueError):
                fuse_rankings({}, top_k, rrf_k)

    def test_pipeline_fuses_before_truncation_and_evaluates_hybrid(self):
        pipeline = RetrievalPipeline(Catalog())
        with patch("backend.retrieval.pipeline.search", return_value=[match("a"), match("shared")]), \
             patch.object(pipeline.bm25, "search", return_value=[match("b"), match("shared")]):
            result = pipeline.search("roof", top_k=1)
            self.assertEqual(result["hybrid"]["matches"][0]["id"], "shared")
            self.assertEqual(result["rag"]["matches"][0]["id"], "a")
            self.assertEqual(result["bm25"]["matches"][0]["id"], "b")
            self.assertGreaterEqual(result["hybrid"]["latency_ms"],
                                    result["rag"]["latency_ms"] + result["bm25"]["latency_ms"])
            report = evaluate(pipeline, [{"id": "Q1", "query": "roof", "difficulty": "easy",
                                         "query_type": "keyword", "expected_sources": ["roof.pdf"],
                                         "expected_pages": [2]}], 1)
            self.assertEqual(set(report["methods"]), {"rag", "bm25", "hybrid"})
            self.assertEqual(report["methods"]["hybrid"]["overall"]["hit_at_1"], 1)

    def test_no_keyword_overlap_leaves_semantic_ranking(self):
        pipeline = RetrievalPipeline(Catalog())
        result = pipeline.search("unknownword")
        self.assertEqual([m["id"] for m in result["hybrid"]["matches"]], ["0"])
        self.assertEqual(result["hybrid"]["matches"][0]["retrieval_ranks"], {"rag": 1})
        baseline = RetrievalPipeline(Catalog(), include_hybrid=False).search("roof")
        self.assertNotIn("hybrid", baseline)


if __name__ == "__main__":
    unittest.main()

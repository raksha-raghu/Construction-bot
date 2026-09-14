"""Regression tests for evaluation labels, metrics, and standalone reports."""

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from backend.evaluation.cli import main
from backend.evaluation.dataset import validate_corpus, validate_dataset
from backend.evaluation.metrics import score_matches
from backend.evaluation.reports import write_reports
from backend.evaluation.runner import evaluate


def case():
    return {"id": "Q1", "query": "roof", "query_type": "application", "difficulty": "easy",
            "expected_sources": ["roof.pdf"], "expected_pages": [2]}


class Retriever:
    def __init__(self):
        self.calls = []

    def search(self, question, top_k=5):
        self.calls.append((question, top_k))
        return {"query": question,
                "rag": {"matches": [{"source": "wrong.pdf", "page": 2},
                                    {"source": "roof.pdf", "page": 2}], "latency_ms": 10.0},
                "bm25": {"matches": [], "latency_ms": 1.0}}


class EvaluationTests(unittest.TestCase):
    def test_corpus_must_contain_a_relevant_source_and_page(self):
        validate_corpus([case()], [{"source": "roof.pdf", "page": 2}])
        for metadata in ([], [{"source": "roof.pdf", "page": 1}], [None]):
            with self.assertRaisesRegex(ValueError, "Q1"):
                validate_corpus([case()], metadata)

    def test_known_ranking_and_cutoffs(self):
        matches = [{"source": "wrong.pdf", "page": 2},
                   {"source": "roof.pdf", "page": 1},
                   {"source": "roof.pdf", "page": 2}]
        metrics = score_matches(matches, case(), 3)
        self.assertEqual(metrics["first_relevant_rank"], 3)
        self.assertEqual(metrics["hit_at_1"], 0)
        self.assertEqual(metrics["hit_at_3"], 1)
        self.assertAlmostEqual(metrics["mrr"], 1 / 3)
        self.assertNotIn("hit_at_5", metrics)
        unrestricted = case()
        unrestricted["expected_pages"] = []
        self.assertEqual(score_matches(matches, unrestricted, 5)["first_relevant_rank"], 2)
        self.assertEqual(score_matches([], case(), 5)["mrr"], 0)

    def test_bad_datasets(self):
        invalid = [[], {}, [None], [case(), case()]]
        for field, value in [("query", " "), ("expected_sources", []),
                             ("expected_pages", [True]), ("expected_pages", [-1]),
                             ("expected_pages", "2")]:
            invalid.append([{**case(), field: value}])
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                validate_dataset(data)

    def test_comparison_aggregation_and_reports(self):
        retriever = Retriever()
        result = evaluate(retriever, [case()], 5)
        self.assertEqual(retriever.calls, [("roof", 5)])
        self.assertEqual(result["methods"]["rag"]["overall"]["mrr"], 0.5)
        self.assertEqual(result["methods"]["bm25"]["failed_queries"], 1)
        self.assertEqual(result["methods"]["rag"]["by_difficulty"]["easy"]["hit_at_5"], 1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            write_reports(result, output, {"dataset_sha256": "test"})
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["schema_version"], 1)
            self.assertEqual(len(json.loads((output / "rankings.json").read_text())), 2)
            with (output / "failure_cases.csv").open(encoding="utf-8-sig", newline="") as stream:
                failures = list(csv.DictReader(stream))
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["method"], "bm25")
            with self.assertRaises(FileExistsError):
                write_reports(result, output, {})

    def test_cli_validates_before_loading_models(self):
        with patch("backend.evaluation.cli.build_retriever") as build, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["--top-k", "0"])
            with tempfile.TemporaryDirectory() as directory:
                dataset = Path(directory) / "invalid.json"
                dataset.write_text("[]")
                with self.assertRaises(SystemExit):
                    main(["--dataset", str(dataset)])
            build.assert_not_called()

    def test_cli_warmup_and_complete_run(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "cases.json"
            dataset.write_text(json.dumps([case()]))
            output = Path(directory) / "reports"
            retriever = Retriever()
            with patch("backend.evaluation.cli.build_retriever", return_value=(retriever, object())), \
                 patch("backend.evaluation.cli.corpus_metadata", return_value={"chunk_count": 2}), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--dataset", str(dataset), "--output-dir", str(output), "--warmup"]), 0)
            self.assertEqual(len(retriever.calls), 2)
            self.assertTrue(json.loads((output / "summary.json").read_text())["warmup"])


if __name__ == "__main__":
    unittest.main()

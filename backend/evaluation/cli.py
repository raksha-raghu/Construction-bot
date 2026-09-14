"""Command-line evaluation against an existing index."""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

from .dataset import load_dataset, validate_corpus
from .reports import write_reports
from .runner import evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SingleMethodRetriever:
    def __init__(self, method, search_fn):
        self.method = method
        self.search_fn = search_fn

    def search(self, question, top_k=5):
        start = perf_counter()
        matches = self.search_fn(question, top_k=top_k)
        return {"query": question, self.method: {
            "matches": matches, "latency_ms": (perf_counter() - start) * 1000,
        }}


def build_retriever(method: str, dataset: list[dict]):
    # Lazy imports keep dataset validation, metrics, and --help model-independent.
    from backend.storage.vector_store import load_collection

    collection = load_collection(with_embeddings=False)
    if collection.count() == 0:
        raise ValueError("The index is empty. Add catalogs and run python -m backend --rebuild.")
    validate_corpus(dataset, collection.get(include=["metadatas"])["metadatas"])
    if method != "bm25":
        collection = load_collection(with_embeddings=True)
    if method in ("both", "all", "hybrid"):
        from backend.retrieval.pipeline import RetrievalPipeline
        retriever = RetrievalPipeline(collection, include_hybrid=method != "both")
        if method == "hybrid":
            pipeline = retriever
            retriever = SingleMethodRetriever("hybrid", lambda question, top_k: pipeline.search(question, top_k)["hybrid"]["matches"])
    elif method == "bm25":
        from backend.retrieval.bm25 import BM25Retriever
        retriever = SingleMethodRetriever("bm25", BM25Retriever(collection).search)
    else:
        from backend.retrieval.semantic import search
        retriever = SingleMethodRetriever("rag", lambda question, top_k: search(collection, question, top_k))
    return retriever, collection


def corpus_metadata(collection, dataset) -> dict:
    data = collection.get(include=["documents", "metadatas"])
    validate_corpus(dataset, data["metadatas"])
    records = sorted(zip(data["ids"], data["documents"], data["metadatas"]))
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {"collection": collection.name, "chunk_count": len(records),
            "corpus_sha256": fingerprint, "collection_metadata": collection.metadata}


def main(argv=None, default_method="all") -> int:
    parser = argparse.ArgumentParser(description="Evaluate catalog retrieval independently of interactive search.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "backend/evaluation/evaluation_dataset.json")
    parser.add_argument("--method", choices=("all", "both", "rag", "bm25", "hybrid"), default=default_method)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, help="New report directory; must not already exist.")
    parser.add_argument("--warmup", action="store_true", help="Run one unmeasured query before evaluation.")
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    started = datetime.now(timezone.utc)
    output = args.output_dir or PROJECT_ROOT / "backend/evaluation/results" / started.strftime("%Y%m%dT%H%M%S%fZ")
    try:
        dataset = load_dataset(args.dataset)
        if output.exists():
            raise ValueError(f"Report directory already exists: {output}. Choose a new directory.")
        print(f"Loaded {len(dataset)} questions. Loading {args.method} retrieval...", flush=True)
        start = perf_counter()
        retriever, collection = build_retriever(args.method, dataset)
        setup_ms = (perf_counter() - start) * 1000
        metadata = corpus_metadata(collection, dataset)
        if args.warmup:
            retriever.search(dataset[0]["query"], top_k=args.top_k)
        print("Evaluating questions...", flush=True)
        result = evaluate(retriever, dataset, args.top_k)
        packages = {}
        for package in ("chromadb", "sentence-transformers", "rank-bm25"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = "unavailable"
        from backend.storage.vector_store import EMBEDDING_MODEL
        from backend.retrieval.fusion import RRF_K, CANDIDATE_K
        metadata.update({
            "started_at_utc": started.isoformat(), "method": args.method,
            "dataset_path": str(args.dataset.resolve()),
            "dataset_sha256": hashlib.sha256(json.dumps(dataset, sort_keys=True).encode()).hexdigest(),
            "python_version": platform.python_version(), "packages": packages,
            "embedding_model": EMBEDDING_MODEL, "setup_ms": setup_ms,
            "warmup": args.warmup, "mrr_cutoff": args.top_k,
            "relevance_rule": "source in expected_sources AND (expected_pages empty OR page in expected_pages)",
            "latency_scope": "retrieval only; excludes setup and report writing",
            "fusion": {"method": "rrf", "rrf_k": RRF_K, "equal_weights": True,
                       "candidate_k": max(CANDIDATE_K, args.top_k),
                       "keyword_filter": "at least one query token overlaps"}
                      if args.method in ("all", "hybrid") else None,
        })
        destination = write_reports(result, output, metadata)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    for method, summary in result["methods"].items():
        print(f"\n{method.upper()}")
        for metric, value in summary["overall"].items():
            print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
    print(f"\nReports: {destination}")
    return 0

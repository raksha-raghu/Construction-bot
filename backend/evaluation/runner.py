"""Evaluation orchestration independent of databases and command-line I/O."""

from collections import defaultdict
from typing import Protocol

from .dataset import validate_dataset
from .metrics import aggregate, score_matches


class ComparisonRetriever(Protocol):
    def search(self, question: str, top_k: int = 5) -> dict: ...


def evaluate(retriever: ComparisonRetriever, dataset: list[dict], top_k: int = 5) -> dict:
    """Evaluate every case. MRR is truncated at the requested retrieval depth."""
    validate_dataset(dataset)
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    rows, rankings = [], []
    methods = None
    for case in dataset:
        result = retriever.search(case["query"], top_k=top_k)
        current_methods = tuple(key for key in result if key != "query")
        if not current_methods or (methods is not None and current_methods != methods):
            raise ValueError("Retriever must return consistent, non-empty method sections.")
        methods = current_methods
        for method in methods:
            section = result[method]
            matches = section["matches"][:top_k]
            rows.append({
                "id": case["id"], "query": case["query"], "method": method,
                "query_type": case["query_type"], "difficulty": case["difficulty"],
                "expected_sources": " | ".join(case["expected_sources"]),
                "expected_pages": " | ".join(map(str, case["expected_pages"])),
                **score_matches(matches, case, top_k),
                "latency_ms": section["latency_ms"],
                "returned_count": len(matches),
                "top1_source": matches[0].get("source") if matches else None,
                "top1_page": matches[0].get("page") if matches else None,
            })
            rankings.append({"id": case["id"], "method": method, "matches": matches})
    summaries = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        groups = {}
        for field in ("query_type", "difficulty"):
            grouped = defaultdict(list)
            for row in selected:
                grouped[row[field]].append(row)
            groups[f"by_{field}"] = {key: aggregate(value) for key, value in sorted(grouped.items())}
        summaries[method] = {
            "overall": aggregate(selected), **groups,
            "failed_queries": sum(row[f"hit_at_{top_k}"] == 0 for row in selected),
        }
    return {"top_k": top_k, "methods": summaries, "rows": rows, "rankings": rankings}

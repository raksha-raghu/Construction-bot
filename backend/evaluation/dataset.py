"""Validated labelled evaluation datasets."""

import json
from pathlib import Path


def validate_dataset(data: object) -> list[dict]:
    if not isinstance(data, list) or not data:
        raise ValueError("Dataset must be a non-empty JSON list.")
    seen = set()
    for index, case in enumerate(data, 1):
        if not isinstance(case, dict):
            raise ValueError(f"Case {index} must be an object.")
        for field in ("id", "query", "query_type", "difficulty"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f"Case {index}: {field} must be a non-empty string.")
        if case["id"] in seen:
            raise ValueError(f"Duplicate case ID: {case['id']}")
        seen.add(case["id"])
        sources = case.get("expected_sources")
        if not isinstance(sources, list) or not sources or any(
            not isinstance(source, str) or not source.strip() for source in sources
        ):
            raise ValueError(f"{case['id']}: expected_sources must contain source names.")
        pages = case.get("expected_pages")
        if not isinstance(pages, list) or any(type(page) is not int or page < 1 for page in pages):
            raise ValueError(f"{case['id']}: expected_pages must contain positive integers.")
    return data


def load_dataset(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig") as stream:
        return validate_dataset(json.load(stream))


def validate_corpus(dataset: list[dict], metadatas: list[dict]) -> None:
    """Reject cases whose relevance labels cannot match the indexed corpus."""
    pages_by_source = {}
    for metadata in metadatas:
        metadata = metadata or {}
        pages_by_source.setdefault(metadata.get("source"), set()).add(metadata.get("page"))
    missing = []
    for case in dataset:
        if not any(
            source in pages_by_source and (
                not case["expected_pages"]
                or pages_by_source[source].intersection(case["expected_pages"])
            ) for source in case["expected_sources"]
        ):
            missing.append(case["id"])
    if missing:
        raise ValueError(
            "Dataset labels have no matching indexed source/page for: " + ", ".join(missing)
            + ". Add the matching catalogs and rebuild, or select a dataset for this corpus."
        )

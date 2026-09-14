"""Reciprocal rank fusion (Cormack et al., SIGIR 2009)."""

RRF_K = 60
CANDIDATE_K = 20


def fuse_rankings(rankings: dict[str, list[dict]], top_k: int, rrf_k: int = RRF_K) -> list[dict]:
    """Merge by chunk ID, with equal weights and deterministic ID tie breaks.

    Each list is ordered best first. A chunk contributes once per method.
    Scores remain unrounded until display/serialization by callers.
    """
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if type(rrf_k) is not int or rrf_k < 1:
        raise ValueError("rrf_k must be a positive integer.")
    merged = {}
    for method, matches in rankings.items():
        seen = set()
        for position, match in enumerate(matches, 1):
            chunk_id = match.get("id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError("Fusion requires a non-empty chunk ID.")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            entry = merged.setdefault(chunk_id, {
                "id": chunk_id,
                **{key: match.get(key) for key in ("text", "source", "page", "chunk_index")},
                "rrf_score": 0.0, "retrieval_ranks": {},
            })
            entry["rrf_score"] += 1.0 / (rrf_k + position)
            entry["retrieval_ranks"][method] = position
    ordered = sorted(merged.values(), key=lambda item: (-item["rrf_score"], item["id"]))
    return [{**match, "rank": rank} for rank, match in enumerate(ordered[:top_k], 1)]

"""Pure ranking metrics using source and optional page relevance labels."""

from statistics import mean


def is_relevant(match: dict, case: dict) -> bool:
    return match.get("source") in case["expected_sources"] and (
        not case["expected_pages"] or match.get("page") in case["expected_pages"]
    )


def first_relevant_rank(matches: list[dict], case: dict) -> int | None:
    return next((rank for rank, match in enumerate(matches, 1) if is_relevant(match, case)), None)


def score_matches(matches: list[dict], case: dict, top_k: int) -> dict:
    rank = first_relevant_rank(matches[:top_k], case)
    cutoffs = sorted({k for k in (1, 3, 5, top_k) if k <= top_k})
    return {
        "first_relevant_rank": rank,
        **{f"hit_at_{k}": int(rank is not None and rank <= k) for k in cutoffs},
        "mrr": 1.0 / rank if rank else 0.0,
    }


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n_queries": 0}
    keys = [key for key in rows[0] if key.startswith("hit_at_")] + ["mrr"]
    return {
        "n_queries": len(rows),
        **{key: mean(row[key] for row in rows) for key in keys},
        "mean_latency_ms": mean(row["latency_ms"] for row in rows),
    }

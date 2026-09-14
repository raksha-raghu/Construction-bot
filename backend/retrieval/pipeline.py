"""Compare dense RAG retrieval and BM25 over the same catalog chunks."""

from time import perf_counter

from backend.retrieval.bm25 import BM25Retriever, tokenize
from backend.retrieval.fusion import CANDIDATE_K, fuse_rankings
from backend.retrieval.semantic import TOP_K, search


class RetrievalPipeline:
    def __init__(self, collection, include_hybrid: bool = True):
        self.collection = collection
        self.include_hybrid = include_hybrid
        # Build once per session, outside the per-query retrieval timings.
        self.bm25 = BM25Retriever(collection)

    def search(self, question: str, top_k: int = TOP_K):
        if not isinstance(question, str):
            raise ValueError("Question must be a string.")
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be at least 1.")

        depth = max(top_k, CANDIDATE_K) if self.include_hybrid else top_k
        limit = min(depth, len(self.bm25.documents))
        start = perf_counter()
        rag_matches = search(self.collection, question, top_k=limit)
        rag_ms = (perf_counter() - start) * 1000

        start = perf_counter()
        bm25_matches = self.bm25.search(question, top_k=limit)
        bm25_ms = (perf_counter() - start) * 1000

        result = {
            "query": question,
            "rag": {"matches": rag_matches[:top_k], "latency_ms": rag_ms},
            "bm25": {"matches": bm25_matches[:top_k], "latency_ms": bm25_ms},
        }
        if self.include_hybrid:
            start = perf_counter()
            query_tokens = set(tokenize(question))
            # No-overlap BM25 results are arbitrary ties, not keyword evidence.
            keyword_candidates = [match for match in bm25_matches
                                  if query_tokens.intersection(tokenize(match["text"]))]
            matches = fuse_rankings({"rag": rag_matches, "bm25": keyword_candidates}, top_k)
            fusion_ms = (perf_counter() - start) * 1000
            result["hybrid"] = {
                "matches": matches, "latency_ms": rag_ms + bm25_ms + fusion_ms,
                "fusion_ms": fusion_ms, "candidate_k": limit,
            }
        return result

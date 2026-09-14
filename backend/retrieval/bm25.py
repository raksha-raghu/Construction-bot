import re
from rank_bm25 import BM25Okapi


def tokenize(text: str):
    if not text:
        return []

    return re.findall(
        r"[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*",
        text.lower()
    )


class BM25Retriever:
    def __init__(self, collection):
        data = collection.get(include=["documents", "metadatas"])

        self.documents = data["documents"]
        self.ids = data["ids"]
        self.metadatas = data["metadatas"]

        if not self.documents:
            raise RuntimeError("No documents found in the Chroma collection.")

        tokenized_corpus = [tokenize(document) for document in self.documents]
        if not any(tokenized_corpus):
            raise ValueError("The corpus contains no searchable BM25 tokens.")
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Question must not be empty.")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for rank, index in enumerate(ranked_indices, start=1):
            metadata = self.metadatas[index] or {}

            results.append({
                "id": self.ids[index],
                "rank": rank,
                "text": self.documents[index],
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "chunk_index": metadata.get("chunk_index"),
                "bm25_score": round(float(scores[index]), 4),
            })

        return results

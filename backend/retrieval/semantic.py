import chromadb

TOP_K = 5


def search(collection: chromadb.Collection, query: str, top_k: int = TOP_K):
    """Return ranked catalog passages using cosine semantic similarity."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Question must not be empty.")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    count = collection.count()
    if not count:
        return []
    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    matches = []

    for rank, (doc, meta, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ):
        meta = meta or {}
        matches.append({
            "id": results["ids"][0][rank - 1],
            "rank": rank,
            "text": doc,
            "source": meta.get("source"),
            "page": meta.get("page"),
            "chunk_index": meta.get("chunk_index"),
            # Collection is explicitly configured for cosine distance.
            "similarity": round(1.0 - float(distance), 4),
        })

    return matches

import chromadb
from chromadb.utils import embedding_functions

from vectordb import Vectordb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = "./vector_db"
COLLECTION_NAME = "construction_catalogs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5


def search(collection: chromadb.Collection, query: str, top_k: int = TOP_K):
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    matches = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        matches.append({
            "text": doc,
            "source": meta.get("source"),
            "chunk_index": meta.get("chunk_index"),
            "similarity": round(1 - dist, 4),
        })

    return matches
import argparse

from backend.retrieval.bm25 import BM25Retriever
from backend.storage.vector_store import Vectordb


def print_matches(matches):
    if not matches:
        print("No BM25 matches found.")
        return

    for match in matches:
        print(
            f"\n#{match['rank']} "
            f"[{match['source']} — page {match['page']} — "
            f"chunk {match['chunk_index']}] "
            f"BM25={match['bm25_score']}"
        )

        preview = match["text"][:700]

        if len(match["text"]) > 700:
            preview += "..."

        print(preview)


def main():
    parser = argparse.ArgumentParser(
        description="Test BM25 retrieval against the existing construction catalogue corpus."
    )

    parser.add_argument(
        "query",
        nargs="+",
        help="Construction catalogue query"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return"
    )

    args = parser.parse_args()

    collection = Vectordb(False)

    print("Building BM25 index from the existing Chroma corpus...")
    retriever = BM25Retriever(collection)

    query = " ".join(args.query)

    print(f"\nQuery: {query}")
    print_matches(retriever.search(query, top_k=args.top_k))


if __name__ == "__main__":
    main()

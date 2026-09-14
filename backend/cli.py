import argparse

from backend.console import run_query_session
from backend.retrieval.semantic import TOP_K
from backend.storage.vector_store import Vectordb


def main():
    parser = argparse.ArgumentParser(
        description="Compare semantic retrieval, BM25, and hybrid rank fusion for each question."
    )

    parser.add_argument(
        "query",
        nargs="*",
        help="Optional query. If omitted, interactive mode starts.",
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and rebuild vector_db from the PDFs in ./catalogs.",
    )

    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help="Number of results per retrieval method (default: 5).",
    )
    parser.add_argument("--llm", action="store_true", help="Generate an answer from hybrid results using Ollama.")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama server base URL.")
    parser.add_argument("--llm-timeout", type=float, default=120, help="Ollama request timeout in seconds.")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    question = " ".join(args.query).strip() if args.query else None
    if question == "":
        parser.error("Question must not be empty")

    try:
        llm_config = None
        if args.llm:
            from backend.llm import OllamaConfig
            from backend.llm.ollama import load_system_prompt
            llm_config = OllamaConfig(base_url=args.ollama_url, timeout=args.llm_timeout)
            load_system_prompt()  # Validate the required prompt before loading the index.
        collection = Vectordb(args.rebuild)
        run_query_session(collection, question=question, top_k=args.top_k,
                          llm_config=llm_config)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Search failed: {exc}\n")


if __name__ == "__main__":
    main()

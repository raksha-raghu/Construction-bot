from backend.retrieval.pipeline import RetrievalPipeline
from backend.retrieval.semantic import TOP_K


def print_matches(matches, score_key="similarity"):
    if not matches:
        print("No matches found.")
        return
    for match in matches:
        print(
            f"\n#{match['rank']}  "
            f"[{match['source']} - page {match['page']} - "
            f"chunk {match['chunk_index']}]  "
            f"{score_key}={match[score_key]}"
        )
        preview = match["text"][:600]
        if len(match["text"]) > 600:
            preview += "..."
        print(preview)


def print_comparison(result):
    print(f"\nQuestion: {result['query']}")
    for method, title, score_key in (
        ("rag", "RAG / semantic retrieval", "similarity"),
        ("bm25", "BM25 / keyword retrieval", "bm25_score"),
        ("hybrid", "Hybrid / reciprocal rank fusion", "rrf_score"),
    ):
        if method not in result:
            continue
        section = result[method]
        print(f"\n=== {title} ({section['latency_ms']:.2f} ms) ===")
        print_matches(section["matches"], score_key=score_key)


def run_query_session(collection, question=None, top_k=TOP_K, *, llm_config=None):
    pipeline = RetrievalPipeline(collection)

    def handle_question(question):
        result = pipeline.search(question, top_k=top_k)
        print_comparison(result)
        if llm_config is not None:
            from backend.llm import generate_answer
            print("\nGenerating answer with Ollama...", flush=True)
            answer = generate_answer(result["query"], result["hybrid"]["matches"],
                                     config=llm_config)
            print(f"\n=== Answer ({answer['model']}) ===\n{answer['answer']}")
            if answer["sources"]:
                print("\nRetrieved sources supplied to the model:")
                for source in answer["sources"]:
                    print(f"[{source['citation']}] {source['source']} - page {source['page']} - chunk {source['chunk_index']}")

    if question is not None:
        handle_question(question)
        return
    print("Interactive RAG + BM25 + Hybrid comparison (type 'exit' to quit)")
    while True:
        try:
            question = input("\nQuery: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in ("exit", "quit", ""):
            break
        try:
            handle_question(question)
        except (ValueError, RuntimeError) as exc:
            print(f"Question failed: {exc}")


# Backward-compatible name for callers of the original interface.
Query = run_query_session

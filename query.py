import sys

import chromadb
from search import search

def print_matches(matches):
    if not matches:
        print("No matches found.")
        return
    for rank, m in enumerate(matches, start=1):
        print(f"\n#{rank}  [{m['source']} — chunk {m['chunk_index']}]  similarity={m['similarity']}")
        print(m["text"][:400] + ("..." if len(m["text"]) > 400 else ""))


def Query(collection: chromadb.Collection):
 
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print_matches(search(collection, query))
    else:
        print("Interactive search (type 'exit' to quit)")
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() in ("exit", "quit", ""):
                break
            print_matches(search(collection,query))



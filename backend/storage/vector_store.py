import os
import glob
import shutil
from pathlib import Path
import pdfplumber
import chromadb
from chromadb.utils import embedding_functions

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = str(PROJECT_ROOT / "catalogs")
DB_PATH = str(PROJECT_ROOT / "vector_db")
COLLECTION_NAME = "construction_catalogs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split one page of text into overlapping character chunks."""
    text = (text or "").strip()
    if not text:
        return []

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk size must be positive and overlap must be between 0 and chunk size - 1.")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------
def extract_pages_from_pdf(path: str):
    """
    Extract text page-by-page.

    Returns:
        list[dict]: [{"page": 1, "text": "..."}, ...]
    """
    pages = []

    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            page_text = page_text.strip()

            if page_text:
                pages.append({
                    "page": page_number,
                    "text": page_text,
                })

    return pages


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def _embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def load_collection(with_embeddings: bool = True):
    """Load an already-built collection."""
    client = chromadb.PersistentClient(path=DB_PATH)

    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_function() if with_embeddings else None,
    )


def rebuild_collection():
    """
    Rebuild the Chroma collection from the PDFs currently inside ./catalogs.

    The old vector_db directory is removed so deleted/renamed PDF chunks cannot
    remain in the evaluation corpus.
    """
    pdf_paths = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files were found in '{PDF_DIR}'. Add catalogs before rebuilding.")
    target = Path(DB_PATH).resolve()
    if target != PROJECT_ROOT / "vector_db" or Path(DB_PATH).is_symlink():
        raise ValueError("Rebuild is restricted to this project's vector_db directory.")
    embedding_fn = _embedding_function()
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0

    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        print(f"Processing: {filename}")

        pages = extract_pages_from_pdf(pdf_path)

        if not pages:
            print("  Warning: no extractable text found. The PDF may be scanned.")
            continue

        pdf_chunk_count = 0

        for page_data in pages:
            page_number = page_data["page"]
            chunks = chunk_text(page_data["text"])

            if not chunks:
                continue

            ids = [
                f"{filename}::page-{page_number}::chunk-{chunk_index}"
                for chunk_index in range(len(chunks))
            ]

            metadatas = [
                {
                    "source": filename,
                    "page": page_number,
                    "chunk_index": chunk_index,
                }
                for chunk_index in range(len(chunks))
            ]

            collection.upsert(
                ids=ids,
                documents=chunks,
                metadatas=metadatas,
            )

            pdf_chunk_count += len(chunks)
            total_chunks += len(chunks)

        print(f"  Added {pdf_chunk_count} chunks.")

    print("\nVector database rebuild complete.")
    print(f"Total stored chunks: {collection.count()}")

    return collection


def Vectordb(collection_regeneration: bool = False):
    """
    Compatibility wrapper used by the existing main.py design.

    True  -> rebuild the database from the current PDF folder.
    False -> load the existing database.
    """
    if collection_regeneration:
        return rebuild_collection()

    try:
        return load_collection()
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the vector database: {exc}. "
            "If no index exists, run 'python -m backend --rebuild' first."
        ) from exc

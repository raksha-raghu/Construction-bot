import os
import glob
import pdfplumber
import chromadb
from chromadb.utils import embedding_functions

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PDF_DIR = "./catalogs"                 # folder containing PDF files
DB_PATH = "./vector_db"            # persistent Chroma DB location
COLLECTION_NAME = "construction_catalogs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # runs locally via sentence-transformers
CHUNK_SIZE = 800                   # characters per chunk
CHUNK_OVERLAP = 150                # characters of overlap between chunks


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------
def extract_text_from_pdf(path: str) -> str:
    """Extract text from all pages of a PDF using pdfplumber."""
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping chunks so embeddings stay within a
    reasonable size and retrieval can return focused passages."""
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def Vectordb(collection_regeneration: bool):
    if collection_regeneration is False:
        # Delete the existing collection if it exists
        client = chromadb.PersistentClient(path=DB_PATH)
        if COLLECTION_NAME in client.list_collections():
            print(f"Existing collection '{COLLECTION_NAME}' is used.")
            return client.get_collection(name=COLLECTION_NAME)

            
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    pdf_paths = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if not pdf_paths:
        print(f"No PDFs found in {PDF_DIR}. Add some .pdf files and re-run.")
        return

    total_chunks = 0
    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        print(f"Processing {filename} ...")

        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            print(f"  Warning: no extractable text in {filename} (might be scanned). Skipping.")
            continue

        chunks = chunk_text(text)
        if not chunks:
            continue

        ids = [f"{filename}::chunk-{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

        # Chroma will call embedding_fn internally to embed these documents.
        collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
        )

        total_chunks += len(chunks)
        print(f"  Added {len(chunks)} chunks.")

    print("\nDone.")
    print("Total documents in collection:", collection.count())
    return collection

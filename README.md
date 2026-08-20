# Construction Catalog Search

A local semantic-search tool for construction product catalogs. It extracts text
from PDF files, splits the text into overlapping chunks, stores embeddings in a
persistent ChromaDB collection, and returns the passages most similar to a
natural-language query.

The project runs locally and does not require an API key.

## Features

- Reads every PDF in the `catalogs/` directory
- Creates embeddings with `all-MiniLM-L6-v2`
- Persists the vector index in `vector_db/`
- Supports command-line and interactive searches
- Shows the source file, chunk number, and similarity score for each result

## Requirements

- Python 3.10 or newer
- Internet access on the first run to download the embedding model
- Text-based PDF catalogs

Scanned image-only PDFs require OCR before this project can search them.

## Setup

Clone the repository, open a terminal in the project directory, and create a
virtual environment.

### Windows PowerShell

```powershell
py -m venv venv
venv\Scripts\Activate.ps1
python -m pip install chromadb pdfplumber sentence-transformers
```

If PowerShell prevents activation, you can run the environment's Python
directly, for example `venv\Scripts\python.exe main.py`.

### macOS or Linux

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install chromadb pdfplumber sentence-transformers
```

## Add catalogs

Place one or more `.pdf` files in the `catalogs/` directory:

```text
Construction-RAG-bot/
|-- catalogs/
|   |-- product-catalog.pdf
|   `-- technical-guide.pdf
|-- main.py
|-- query.py
|-- search.py
`-- vectordb.py
```

## Usage

Run a single search by passing the question after `main.py`:

```powershell
python main.py "Which waterproofing system is suitable for a flat roof?"
```

Run without a question to start an interactive session:

```powershell
python main.py
```

Then enter queries at the prompt. Enter `exit`, `quit`, or an empty line to
close the program.

Example result:

```text
#1  [product-catalog.pdf - chunk 42]  similarity=0.7312
Matching text from the catalog...
```

The five closest passages are returned by default. Similarity scores are most
useful for ranking the results relative to one another; they are not confidence
percentages.

## Index behavior

`main.py` currently calls `Vectordb(True)`, so the PDFs are processed when the
program starts. The resulting ChromaDB collection is stored in `vector_db/`.

To reuse an existing collection without processing the PDFs again, change the
line in `main.py` to:

```python
collection = Vectordb(False)
```

When catalogs are added, removed, or changed, rebuild the index by deleting the
local `vector_db/` directory and running the program again. The directory is
generated data and is excluded from Git.

## Configuration

The main settings are constants near the top of `vectordb.py` and `search.py`:

| Setting | Default | Purpose |
| --- | --- | --- |
| `PDF_DIR` | `./catalogs` | Directory containing PDF files |
| `DB_PATH` | `./vector_db` | Persistent ChromaDB directory |
| `COLLECTION_NAME` | `construction_catalogs` | ChromaDB collection name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformer model |
| `CHUNK_SIZE` | `800` | Maximum characters per text chunk |
| `CHUNK_OVERLAP` | `150` | Characters shared by adjacent chunks |
| `TOP_K` | `5` | Number of search matches returned |

Keep the collection name and embedding model consistent between indexing and
searching.

## Project structure

- `main.py` creates or loads the collection and starts the query interface.
- `vectordb.py` extracts PDF text, chunks it, and stores the embeddings.
- `search.py` queries ChromaDB and formats the matching data.
- `query.py` provides the one-shot and interactive command-line interfaces.
- `catalogs/` contains the source PDF files.
- `vector_db/` contains the generated local vector index.

## Troubleshooting

- **No PDFs found:** Add `.pdf` files directly inside `catalogs/`.
- **No extractable text:** The PDF is probably scanned; run OCR on it first.
- **Model download fails:** Check the internet connection and rerun the command.
- **Catalog changes are not reflected:** Remove `vector_db/` and rebuild it.
- **Duplicate document IDs:** Remove `vector_db/` before rebuilding the same PDFs,
  or use `Vectordb(False)` to load the existing collection.

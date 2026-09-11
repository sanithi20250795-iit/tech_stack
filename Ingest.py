import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

DATA_DIR = Path(__file__).parent / "data"
DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "notes"

CHUNK_SIZE = 500       # words per chunk (approx tokens * 0.75)
CHUNK_OVERLAP = 50     # words of overlap between consecutive chunks


def load_text_from_pdf(path: Path) -> list[tuple[str, int]]:
    """Returns list of (page_text, page_number) tuples."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((text, i + 1))
    return pages


def load_text_from_txt(path: Path) -> list[tuple[str, int]]:
    """Plain text / markdown files have no page numbers, so we use 0."""
    return [(path.read_text(encoding="utf-8", errors="ignore"), 0)]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Splits text into overlapping word-based chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  
    return chunks


def main():
    DATA_DIR.mkdir(exist_ok=True)
    DB_DIR.mkdir(exist_ok=True)

    files = list(DATA_DIR.glob("*.pdf")) + list(DATA_DIR.glob("*.txt")) + list(DATA_DIR.glob("*.md"))
    if not files:
        print(f"No files found in {DATA_DIR}. Drop your PDFs/notes there and re-run.")
        return

    print(f"Found {len(files)} file(s). Loading + chunking...")

    all_chunks = []      # the actual text
    all_ids = []          # unique id per chunk
    all_metadatas = []    # source file + page, so answers can be cited

    for file_path in files:
        if file_path.suffix == ".pdf":
            page_texts = load_text_from_pdf(file_path)
        else:
            page_texts = load_text_from_txt(file_path)

        for page_text, page_num in page_texts:
            chunks = chunk_text(page_text)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{file_path.stem}_p{page_num}_c{i}"
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                all_metadatas.append({
                    "source": file_path.name,
                    "page": page_num,
                })

    print(f"Created {len(all_chunks)} chunks. Embedding + storing in ChromaDB...")

    # "all-MiniLM-L6-v2" is small, fast, and good enough for this use case.
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=str(DB_DIR))
    # get_or_create so re-running ingest.py doesn't error if it already exists
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )

    # Chroma batches embedding + storage together when you call .add()
    # Add in batches to avoid overwhelming memory on large note sets.
    BATCH = 100
    for i in range(0, len(all_chunks), BATCH):
        collection.add(
            documents=all_chunks[i:i + BATCH],
            ids=all_ids[i:i + BATCH],
            metadatas=all_metadatas[i:i + BATCH],
        )

    print(f"Done. {collection.count()} chunks stored in {DB_DIR}")


if __name__ == "__main__":
    main()

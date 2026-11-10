"""
STEP 1 of RAG — INGEST (run once, and again whenever your documents change).

    python ingest.py

Reads every file in data/, splits it into overlapping chunks,
embeds each chunk, and saves everything in a local Chroma database (chroma_db/).
"""
import shutil
from pathlib import Path

import chromadb
from chromadb.config import Settings

import config
from ollama_client import embed


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")


def split_into_chunks(text, size=None, overlap=None):
    """Cut text into pieces of about `size` characters.

    Neighbouring pieces share `overlap` characters, so a rule that sits on a
    boundary still appears whole in at least one chunk. We try to cut at a
    paragraph or sentence break rather than in the middle of a word.
    """
    size = size or config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):  # look back for a nice place to cut
            for sep in ("\n\n", "\n", ". "):
                cut = text.rfind(sep, start + size // 2, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def get_collection(db_dir=None, reset=False):
    db_dir = Path(db_dir or config.DB_DIR)
    if reset and db_dir.exists():
        shutil.rmtree(db_dir)
    client = chromadb.PersistentClient(path=str(db_dir), settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})


def ingest(data_dir=None, db_dir=None, verbose=True):
    data_dir = Path(data_dir or config.DATA_DIR)
    files = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in {".md", ".txt", ".pdf"})
    if not files:
        raise SystemExit(f"No .md, .txt or .pdf files found in {data_dir}")

    collection = get_collection(db_dir, reset=True)  # start fresh every time
    total = 0
    for path in files:
        chunks = split_into_chunks(read_file(path))
        if not chunks:
            print(f"  ! {path.name}: no text found (a scanned PDF needs OCR first)")
            continue
        collection.add(
            ids=[f"{path.name}-{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embed(chunks, kind="document"),
            metadatas=[{"source": path.name, "chunk": i} for i in range(len(chunks))],
        )
        total += len(chunks)
        if verbose:
            print(f"  {path.name:35s} {len(chunks):3d} chunks")
    if verbose:
        print(f"\nDone: {len(files)} documents -> {total} chunks -> {db_dir or config.DB_DIR}")
    return total


if __name__ == "__main__":
    print(f"Ingesting {config.DATA_DIR} with {config.EMBED_MODEL} "
          f"(CHUNK_SIZE={config.CHUNK_SIZE}, CHUNK_OVERLAP={config.CHUNK_OVERLAP})\n")
    ingest()

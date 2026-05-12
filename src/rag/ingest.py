"""PDF ingestion into Chroma.

Each PDF in `datasets/` is chunked once and indexed into:
1. A per-paper collection (for paper-specific queries)
2. The shared `all_papers` collection (for cross-paper search)

Re-running is idempotent: papers already indexed (matched by paper_stem in
`all_papers` metadata) are skipped unless `force=True`.
"""
from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import SETTINGS
from src.rag.store import (
    ALL_PAPERS_COLLECTION,
    get_collection_name,
    get_vectorstore,
)

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128


def _already_indexed(paper_stem: str) -> bool:
    store = get_vectorstore(ALL_PAPERS_COLLECTION)
    try:
        existing = store.get(where={"paper_stem": paper_stem}, limit=1)
        return bool(existing.get("ids"))
    except Exception:
        return False


def ingest_pdf(pdf_path: Path, force: bool = False) -> dict:
    """Index a single PDF. Returns metadata about what happened."""
    paper_stem = pdf_path.stem

    if not force and _already_indexed(paper_stem):
        return {"paper": paper_stem, "status": "skipped", "chunks": 0}

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    for chunk in chunks:
        chunk.metadata["paper_stem"] = paper_stem
        chunk.metadata["source"] = str(pdf_path.name)

    per_paper = get_vectorstore(get_collection_name(paper_stem))
    shared = get_vectorstore(ALL_PAPERS_COLLECTION)

    if force:
        try:
            per_paper.delete(where={"paper_stem": paper_stem})
            shared.delete(where={"paper_stem": paper_stem})
        except Exception:
            pass

    per_paper.add_documents(chunks)
    shared.add_documents(chunks)

    return {"paper": paper_stem, "status": "indexed", "chunks": len(chunks)}


def ingest_directory(directory: Path | None = None, force: bool = False) -> list[dict]:
    """Ingest every PDF in the given directory (defaults to DATASETS_DIR)."""
    directory = directory or SETTINGS.datasets_dir
    results = []
    for pdf in sorted(directory.glob("*.pdf")):
        results.append(ingest_pdf(pdf, force=force))
    return results


if __name__ == "__main__":
    import sys

    force_flag = "--force" in sys.argv
    print(f"Ingesting from {SETTINGS.datasets_dir} (force={force_flag})...")
    for r in ingest_directory(force=force_flag):
        print(f"  [{r['status']:>7}] {r['paper']}  chunks={r['chunks']}")

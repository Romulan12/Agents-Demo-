"""Chroma vector store factory.

One persistent Chroma instance, multiple collections — one per paper plus a
shared "all_papers" collection for cross-paper search.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import SETTINGS
from src.llm import get_embeddings

ALL_PAPERS_COLLECTION = "all_papers"


def get_collection_name(paper_stem: str) -> str:
    """Chroma collection names: alphanumeric + underscores, 3–63 chars."""
    safe = "".join(c if c.isalnum() else "_" for c in paper_stem)
    safe = safe.strip("_")[:60]
    if len(safe) < 3:
        safe = f"doc_{safe}"
    return safe


@lru_cache(maxsize=None)
def get_vectorstore(collection_name: str = ALL_PAPERS_COLLECTION) -> Chroma:
    """Return a Chroma vector store for the given collection (cached per call)."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(SETTINGS.chroma_dir),
    )


def get_retriever(
    collection_name: str = ALL_PAPERS_COLLECTION,
    k: int = 5,
) -> VectorStoreRetriever:
    return get_vectorstore(collection_name).as_retriever(search_kwargs={"k": k})


def list_papers() -> list[str]:
    """Return paper stems known to the index (one per non-shared collection)."""
    store = get_vectorstore(ALL_PAPERS_COLLECTION)
    try:
        items = store.get(include=["metadatas"])
        stems = {m.get("paper_stem") for m in items.get("metadatas", []) if m}
        return sorted(s for s in stems if s)
    except Exception:
        return []

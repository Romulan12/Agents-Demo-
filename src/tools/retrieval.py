"""Retrieval tools — semantic search over Chroma collections."""
from __future__ import annotations

from langchain_core.tools import tool

from src.llm import get_chat_model
from src.rag.store import (
    ALL_PAPERS_COLLECTION,
    get_collection_name,
    get_retriever,
    get_vectorstore,
    list_papers,
)


def _format_docs(docs) -> str:
    if not docs:
        return "No relevant information found."
    chunks = []
    for d in docs:
        stem = d.metadata.get("paper_stem", "unknown")
        page = d.metadata.get("page", "?")
        chunks.append(f"[source: {stem} p.{page}]\n{d.page_content.strip()}")
    return "\n\n---\n\n".join(chunks)


@tool
def vector_search(query: str, paper_stem: str = "") -> str:
    """Semantic search over indexed research papers.

    Args:
        query: The information you're looking for.
        paper_stem: Optional. Restrict to one paper by its filename stem
            (e.g. 'AutonomousDataAgents'). Leave empty to search all papers.

    Returns the most relevant chunks with [source: paper p.N] citations.
    """
    collection = (
        get_collection_name(paper_stem) if paper_stem else ALL_PAPERS_COLLECTION
    )
    retriever = get_retriever(collection_name=collection, k=5)
    docs = retriever.invoke(query)
    return _format_docs(docs)


@tool
def summarize_paper(paper_stem: str) -> str:
    """Summarize a specific paper by its filename stem.

    Args:
        paper_stem: The paper's filename without extension.

    Returns a concise summary grounded in the paper's content.
    """
    store = get_vectorstore(get_collection_name(paper_stem))
    try:
        items = store.get(include=["documents", "metadatas"])
    except Exception as e:
        return f"Error reading paper '{paper_stem}': {e}"

    docs = items.get("documents") or []
    if not docs:
        return f"No content indexed for paper '{paper_stem}'."

    full_text = "\n\n".join(docs)[:60_000]
    llm = get_chat_model(temperature=0.0)
    response = llm.invoke(
        [
            (
                "system",
                "Summarize the provided research paper. Capture: (1) the core "
                "contribution, (2) methods, (3) key findings, (4) limitations. "
                "Use only the supplied text. 200-300 words.",
            ),
            ("user", f"Paper: {paper_stem}\n\nContent:\n{full_text}"),
        ]
    )
    return f"[source: {paper_stem}]\n{response.content}"


@tool
def list_indexed_papers() -> str:
    """List the papers currently indexed in the vector store."""
    papers = list_papers()
    if not papers:
        return "No papers are currently indexed."
    return "Indexed papers:\n" + "\n".join(f"- {p}" for p in papers)

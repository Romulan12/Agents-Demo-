"""arXiv search and download — auto-ingests into Chroma after download."""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import arxiv
from langchain_core.tools import tool

from src.config import SETTINGS
from src.rag.ingest import ingest_pdf


def _safe_filename(text: str, max_length: int = 40) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", text)
    cleaned = re.sub(r"[-\s]+", "_", cleaned).strip("_")
    return cleaned[:max_length]


@tool
def arxiv_download(topic: str, max_results: int = 3, months_back: int = 12) -> str:
    """Search arXiv for recent papers on a topic, download PDFs, and index them.

    Args:
        topic: Search topic (e.g. "retrieval augmented generation").
        max_results: How many papers to download (default 3, max 5).
        months_back: Only include papers from the last N months (default 12).

    Returns a summary of downloaded + indexed papers.
    """
    max_results = min(max(int(max_results), 1), 5)
    cutoff = datetime.now() - timedelta(days=months_back * 30)
    SETTINGS.datasets_dir.mkdir(parents=True, exist_ok=True)

    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=f'"{topic}"',
            max_results=max_results * 2,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        results = list(client.results(search))
    except Exception as e:
        return f"arXiv search failed: {e}"

    downloaded: list[str] = []
    indexed: list[dict] = []

    for r in results:
        if len(downloaded) >= max_results:
            break
        if r.published.replace(tzinfo=None) < cutoff:
            continue

        first_author = r.authors[0].name.split()[-1] if r.authors else "unknown"
        title_short = _safe_filename(r.title, 30)
        date_str = r.published.strftime("%Y-%m-%d")
        filename = f"arxiv_{date_str}_{first_author}_{title_short}.pdf"
        filepath = SETTINGS.datasets_dir / filename

        try:
            if not filepath.exists():
                r.download_pdf(
                    dirpath=str(SETTINGS.datasets_dir), filename=filename
                )
            downloaded.append(filename)
            indexed.append(ingest_pdf(filepath))
        except Exception as e:
            indexed.append({"paper": filename, "status": f"error: {e}", "chunks": 0})

    if not downloaded:
        return f"No papers found on arXiv for '{topic}' within the last {months_back} months."

    lines = [f"Downloaded and indexed {len(downloaded)} papers on '{topic}':"]
    for entry in indexed:
        lines.append(
            f"  - {entry['paper']}: {entry['status']} ({entry.get('chunks', 0)} chunks)"
        )
    return "\n".join(lines)


@tool
def arxiv_search(topic: str, max_results: int = 5, months_back: int = 12) -> str:
    """Search arXiv without downloading. Useful for previewing papers.

    Args:
        topic: Search query.
        max_results: Number of results (default 5, max 10).
        months_back: Only include papers from the last N months.

    Returns titles, authors, dates, and abstracts.
    """
    max_results = min(max(int(max_results), 1), 10)
    cutoff = datetime.now() - timedelta(days=months_back * 30)

    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=f'"{topic}"',
            max_results=max_results * 2,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        results = list(client.results(search))
    except Exception as e:
        return f"arXiv search failed: {e}"

    out = [f"arXiv results for '{topic}':"]
    found = 0
    for r in results:
        if found >= max_results:
            break
        if r.published.replace(tzinfo=None) < cutoff:
            continue
        authors = ", ".join(a.name for a in r.authors[:3])
        if len(r.authors) > 3:
            authors += " et al."
        out.append(
            f"\n- {r.title}\n  Authors: {authors}\n  Published: {r.published.strftime('%Y-%m-%d')}\n  {r.summary[:240].strip()}...\n  {r.entry_id}"
        )
        found += 1

    return "\n".join(out) if found else f"No recent results for '{topic}'."

"""Tool registry — grouped by agent so guardrails can whitelist per agent."""
from __future__ import annotations

from src.tools.arxiv import arxiv_download, arxiv_search
from src.tools.retrieval import list_indexed_papers, summarize_paper, vector_search
from src.tools.web import scrape_url, web_search

QA_TOOLS = [vector_search, summarize_paper, list_indexed_papers, web_search, scrape_url]
BLOG_TOOLS = [vector_search, summarize_paper, list_indexed_papers, web_search, arxiv_search, arxiv_download]
ACADEMIC_TOOLS = [vector_search, summarize_paper, list_indexed_papers, arxiv_search, arxiv_download]

ALL_TOOLS = [
    vector_search,
    summarize_paper,
    list_indexed_papers,
    web_search,
    scrape_url,
    arxiv_search,
    arxiv_download,
]

__all__ = [
    "QA_TOOLS",
    "BLOG_TOOLS",
    "ACADEMIC_TOOLS",
    "ALL_TOOLS",
    "vector_search",
    "summarize_paper",
    "list_indexed_papers",
    "web_search",
    "scrape_url",
    "arxiv_search",
    "arxiv_download",
]

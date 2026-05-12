"""Web search and scraping tools — proper TLS via certifi."""
from __future__ import annotations

import certifi
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
TIMEOUT = 10
MAX_SCRAPE_CHARS = 6000


def _get(url: str) -> requests.Response:
    return requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT,
        verify=certifi.where(),
    )


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current information not in the indexed papers.

    Args:
        query: Search query.
        max_results: How many results to return (default 5, max 10).

    Returns titles, snippets, and URLs.
    """
    max_results = min(max(int(max_results), 1), 10)
    try:
        from googlesearch import search
    except ImportError:
        return "Error: googlesearch-python package is not installed."

    try:
        urls = list(search(query, num_results=max_results, lang="en"))
    except Exception as e:
        return f"Error performing search: {e}"

    if not urls:
        return f"No results for: {query}"

    out = [f"Web search results for: '{query}'"]
    for i, url in enumerate(urls, 1):
        try:
            resp = _get(url)
            soup = BeautifulSoup(resp.content, "html.parser")
            title = soup.find("title")
            title_text = title.get_text(strip=True) if title else url
            meta = soup.find("meta", attrs={"name": "description"})
            snippet = (
                meta.get("content", "").strip()
                if meta
                else "(no description)"
            )
            out.append(f"\n{i}. {title_text}\n   {snippet[:240]}\n   {url}")
        except Exception:
            out.append(f"\n{i}. {url}\n   (could not fetch)")
    return "\n".join(out)


@tool
def scrape_url(url: str) -> str:
    """Fetch and extract the main text from a webpage.

    Args:
        url: Full URL to fetch.

    Returns cleaned text content (truncated for length).
    """
    try:
        resp = _get(url)
        resp.raise_for_status()
    except Exception as e:
        return f"Error fetching {url}: {e}"

    soup = BeautifulSoup(resp.content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned = "\n".join(lines)

    if len(cleaned) > MAX_SCRAPE_CHARS:
        cleaned = cleaned[:MAX_SCRAPE_CHARS] + "\n[...truncated]"
    return f"[source: {url}]\n{cleaned}"

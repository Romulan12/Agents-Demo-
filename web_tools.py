from typing import List, Dict
import requests
from bs4 import BeautifulSoup
from llama_index.core.tools import FunctionTool


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using Google for current information.
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        Formatted string with search results including titles, snippets, and URLs
    """
    try:
        from googlesearch import search
        
        # Get search results (returns URLs)
        search_results = []
        for url in search(query, num_results=max_results, lang="en"):
            search_results.append(url)
        
        if not search_results:
            return f"No results found for query: {query}"
        
        formatted_results = f"Google Search Results for: '{query}'\n\n"
        
        # For each URL, try to get title and snippet by scraping
        for idx, url in enumerate(search_results, 1):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=5)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Get title
                title = soup.find('title')
                title = title.get_text() if title else url
                
                # Get meta description as snippet
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                snippet = meta_desc.get('content', 'No description available') if meta_desc else 'No description available'
                
                formatted_results += f"{idx}. {title}\n"
                formatted_results += f"   {snippet[:200]}...\n"
                formatted_results += f"   URL: {url}\n\n"
                
            except Exception:
                # If scraping fails, just show the URL
                formatted_results += f"{idx}. {url}\n"
                formatted_results += f"   (Unable to fetch page details)\n\n"
        
        return formatted_results
        
    except ImportError:
        return "Error: googlesearch-python package not installed. Please run: pip install googlesearch-python"
    except Exception as e:
        return f"Error performing web search: {str(e)}"


def scrape_webpage(url: str) -> str:
    """
    Extract and return the main text content from a webpage.
    
    Args:
        url: The URL of the webpage to scrape
        
    Returns:
        Extracted text content from the webpage
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Limit length
        max_length = 5000
        if len(text) > max_length:
            text = text[:max_length] + "...\n[Content truncated]"
        
        return f"Content from {url}:\n\n{text}"
        
    except requests.exceptions.RequestException as e:
        return f"Error fetching webpage: {str(e)}"
    except Exception as e:
        return f"Error scraping webpage: {str(e)}"


def get_web_search_tools() -> List[FunctionTool]:
    """
    Create and return web search and scraping tools for the agent.
    
    Returns:
        List of FunctionTool objects for web operations
    """
    web_search_tool = FunctionTool.from_defaults(
        fn=web_search,
        name="web_search",
        description=(
            "Search the web for current information, news, trends, and real-world "
            "applications. Use this when you need up-to-date information that may not "
            "be in the research papers, such as recent implementations, industry adoption, "
            "latest benchmarks, or current events."
        )
    )
    
    scrape_tool = FunctionTool.from_defaults(
        fn=scrape_webpage,
        name="scrape_webpage",
        description=(
            "Extract and read the full content from a specific webpage URL. "
            "Use this after web_search to get detailed information from a specific "
            "source that looks relevant."
        )
    )
    
    return [web_search_tool, scrape_tool]

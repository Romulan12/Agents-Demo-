"""
arXiv Paper Downloader
Downloads latest research papers from arXiv based on topic search
"""

import arxiv
import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta
import re


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """
    Convert text to a safe filename.
    
    Args:
        text: Text to sanitize
        max_length: Maximum length of filename
        
    Returns:
        Sanitized filename string
    """
    # Remove special characters
    text = re.sub(r'[^\w\s-]', '', text)
    # Replace spaces with underscores
    text = re.sub(r'[-\s]+', '_', text)
    # Truncate to max length
    return text[:max_length]


def download_arxiv_papers(
    topic: str,
    max_results: int = 3,
    download_dir: str = "./datasets",
    months_back: int = 12
) -> List[str]:
    """
    Download latest papers from arXiv on a specific topic.
    
    Args:
        topic: Search query (e.g., "RAG", "retrieval augmented generation")
        max_results: Number of papers to download (default: 3)
        download_dir: Directory to save papers (default: "./datasets")
        months_back: Only include papers from last N months (default: 12)
        
    Returns:
        List of paths to downloaded papers
    """
    # Create download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    # Calculate date threshold (papers from last N months)
    date_threshold = datetime.now() - timedelta(days=months_back * 30)
    
    print(f"\n{'='*60}")
    print(f"Searching arXiv for papers on: '{topic}'")
    print(f"Date range: Last {months_back} months")
    print(f"Max results: {max_results}")
    print(f"{'='*60}\n")
    
    downloaded_papers = []
    
    try:
        # First attempt: Exact topic search
        print(f"🔍 Searching for: '{topic}'...")
        search = arxiv.Search(
            query=f'"{topic}"',
            max_results=max_results * 2,  # Get more to filter by date
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        results = list(search.results())
        
        # If not enough results, try broader search with keywords
        if len(results) < max_results:
            print(f"⚠️  Only found {len(results)} papers with exact match.")
            print(f"🔍 Trying broader search with keywords...")
            
            # Extract keywords from topic
            keywords = topic.split()
            keyword_query = " OR ".join([f'"{kw}"' for kw in keywords if len(kw) > 3])
            
            search = arxiv.Search(
                query=keyword_query,
                max_results=max_results * 3,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            results = list(search.results())
        
        print(f"✅ Found {len(results)} total papers\n")
        
        # Filter by date and download
        papers_downloaded = 0
        for result in results:
            if papers_downloaded >= max_results:
                break
            
            # Check if paper is within date range
            if result.published.replace(tzinfo=None) < date_threshold:
                continue
            
            # Create filename
            first_author = result.authors[0].name.split()[-1] if result.authors else "Unknown"
            title_short = sanitize_filename(result.title, max_length=30)
            date_str = result.published.strftime("%Y-%m-%d")
            filename = f"arxiv_{date_str}_{first_author}_{title_short}.pdf"
            filepath = os.path.join(download_dir, filename)
            
            # Skip if already exists
            if os.path.exists(filepath):
                print(f"⏭️  Skipping (already exists): {filename}")
                downloaded_papers.append(filepath)
                papers_downloaded += 1
                continue
            
            # Download paper
            try:
                print(f"📥 Downloading: {result.title[:60]}...")
                print(f"   Authors: {', '.join([a.name for a in result.authors[:3]])}...")
                print(f"   Published: {date_str}")
                print(f"   Saving to: {filename}")
                
                result.download_pdf(filename=filepath)
                downloaded_papers.append(filepath)
                papers_downloaded += 1
                print(f"   ✅ Downloaded successfully!\n")
                
            except Exception as e:
                print(f"   ❌ Error downloading: {str(e)}\n")
                continue
        
        print(f"{'='*60}")
        print(f"✅ Successfully processed {papers_downloaded} papers")
        print(f"{'='*60}\n")
        
        return downloaded_papers
        
    except Exception as e:
        print(f"❌ Error searching arXiv: {str(e)}")
        return []


def search_arxiv_papers(
    topic: str,
    max_results: int = 10,
    months_back: int = 12
) -> List[dict]:
    """
    Search arXiv papers without downloading (for preview).
    
    Args:
        topic: Search query
        max_results: Number of results to return
        months_back: Only include papers from last N months
        
    Returns:
        List of paper metadata dictionaries
    """
    date_threshold = datetime.now() - timedelta(days=months_back * 30)
    
    try:
        search = arxiv.Search(
            query=f'"{topic}"',
            max_results=max_results * 2,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        results = []
        for result in search.results():
            if result.published.replace(tzinfo=None) < date_threshold:
                continue
            
            results.append({
                'title': result.title,
                'authors': [a.name for a in result.authors],
                'published': result.published.strftime("%Y-%m-%d"),
                'summary': result.summary[:200] + "...",
                'pdf_url': result.pdf_url
            })
            
            if len(results) >= max_results:
                break
        
        return results
        
    except Exception as e:
        print(f"Error searching arXiv: {str(e)}")
        return []


if __name__ == "__main__":
    # Example usage
    print("arXiv Paper Downloader - Example Usage\n")
    
    # Download papers on RAG
    papers = download_arxiv_papers(
        topic="Retrieval Augmented Generation",
        max_results=3,
        download_dir="./datasets"
    )
    
    print(f"\nDownloaded {len(papers)} papers:")
    for paper in papers:
        print(f"  - {paper}")

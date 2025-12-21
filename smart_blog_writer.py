#!/usr/bin/env python3
"""
Smart Blog Writer - Intelligent blog generation with automatic source selection
Automatically decides whether to use existing papers, download new ones, or use web search
"""

import os
import asyncio
from pathlib import Path
from typing import List, Tuple, Optional
import time
import httpx
import openai

# Fix SSL certificate verification issues
import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context

from research_agent import ResearchAssistant
from arxiv_downloader import download_arxiv_papers


class SmartBlogWriter:
    """
    Intelligent blog writer that automatically decides information sources.
    
    Features:
    - Extracts topic from natural language query
    - Searches for relevant papers in datasets folder
    - Decides whether to use existing papers, download new ones, or use web
    - Generates comprehensive blog posts
    """
    
    def __init__(self, datasets_dir: str = "./datasets", openai_api_key: str = None):
        """
        Initialize smart blog writer.
         
        Args:
            datasets_dir: Directory containing PDF papers
            openai_api_key: OpenAI API key (optional, will use env var if not provided)
        """
        self.datasets_dir = Path(datasets_dir)
        
        # Get API key
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            raise ValueError("OpenAI API key required")
        
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        
        # Create HTTP client with SSL verification disabled
        self.http_client = httpx.Client(verify=False)
        
        # Create OpenAI client with SSL bypass
        self.client = openai.OpenAI(
            api_key=openai_api_key,
            http_client=self.http_client
        )
    
    def extract_topic(self, query: str) -> str:
        """
        Extract main topic from user query using LLM.
        
        Examples:
        - "Write a blog on computer vision" → "computer vision"
        - "Create blog about RAG systems" → "RAG systems"
        - "Blog on neural networks" → "neural networks"
        
        Args:
            query: User's natural language query
            
        Returns:
            Extracted topic string
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract the main topic from the user's query. Return ONLY the topic, nothing else. Be concise."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                temperature=0,
                max_tokens=50
            )
            
            # Add rate limiting
            time.sleep(0.2)
            
            topic = response.choices[0].message.content.strip()
            # Remove quotes if present
            topic = topic.strip('"').strip("'")
            return topic
            
        except Exception as e:
            print(f"⚠️  Error extracting topic: {e}")
            # Fallback: simple extraction
            return self._simple_topic_extraction(query)
    
    def _simple_topic_extraction(self, query: str) -> str:
        """Fallback simple topic extraction using keywords."""
        query_lower = query.lower()
        
        # Remove common phrases
        for phrase in ["write a blog on", "write blog on", "create blog about", 
                       "blog on", "blog about", "write about"]:
            query_lower = query_lower.replace(phrase, "")
        
        return query_lower.strip()
    
    def find_relevant_papers(self, topic: str) -> List[str]:
        """
        Search datasets folder for papers relevant to topic.
        
        Matching strategy:
        1. Check if topic keywords appear in filename
        2. Case-insensitive matching
        3. Partial word matching
        
        Args:
            topic: Topic to search for
            
        Returns:
            List of paths to relevant papers
        """
        if not self.datasets_dir.exists():
            print(f"⚠️  Datasets directory not found: {self.datasets_dir}")
            return []
        
        relevant_papers = []
        
        # Extract keywords from topic
        topic_keywords = [word.lower() for word in topic.split() if len(word) > 2]
        
        print(f"\n🔍 Searching for papers on '{topic}'...")
        print(f"   Keywords: {topic_keywords}")
        
        # Search through all PDFs
        for pdf_path in self.datasets_dir.glob("*.pdf"):
            filename = pdf_path.stem.lower()
            
            # Check if any keyword matches
            matches = [kw for kw in topic_keywords if kw in filename]
            
            if matches:
                relevant_papers.append(str(pdf_path))
                print(f"   ✅ Found: {pdf_path.name} (matched: {matches})")
        
        if not relevant_papers:
            print(f"   ❌ No papers found matching '{topic}'")
        else:
            print(f"   📚 Total: {len(relevant_papers)} relevant papers")
        
        return relevant_papers
    
    def decide_source(
        self,
        topic: str,
        relevant_papers: List[str],
        download_enabled: bool
    ) -> Tuple[str, Optional[List[str]]]:
        """
        Decide where to get information from.
        
        Decision logic:
        1. If relevant papers exist → use them
        2. If no papers and download enabled → download from arXiv
        3. If no papers and download disabled → use web search
        
        Args:
            topic: Topic to write about
            relevant_papers: List of relevant papers found
            download_enabled: Whether arXiv download is enabled
            
        Returns:
            Tuple of (source_type, resources)
            - source_type: "papers", "arxiv", or "web"
            - resources: List of paper paths or None
        """
        print(f"\n🤔 Deciding information source...")
        
        if relevant_papers:
            print(f"   ✅ Decision: Use {len(relevant_papers)} existing papers")
            return "papers", relevant_papers
        
        if download_enabled:
            print(f"   📥 Decision: Download papers from arXiv")
            return "arxiv", None
        
        print(f"   🌐 Decision: Use web search (no papers, download disabled)")
        return "web", None
    
    async def write_smart_blog(
        self,
        user_query: str,
        download_enabled: bool = False,
        num_papers_to_download: int = 3,
        style: str = "professional",
        word_count: int = 500
    ) -> Tuple[str, dict]:
        """
        Intelligently write blog based on available resources.
        
        This is the main method that orchestrates the entire process:
        1. Extract topic from query
        2. Search for relevant papers
        3. Decide information source
        4. Generate blog
        
        Args:
            user_query: Natural language query (e.g., "Write blog on computer vision")
            download_enabled: Whether to download papers if none found
            num_papers_to_download: Number of papers to download from arXiv
            style: Writing style ("technical", "professional", "casual")
            word_count: Target word count
            
        Returns:
            Tuple of (blog_content, metadata)
            - blog_content: Generated blog post
            - metadata: Dict with decision info (source, papers_used, etc.)
        """
        print("=" * 60)
        print("🧠 SMART BLOG WRITER")
        print("=" * 60)
        
        # Step 1: Extract topic
        print(f"\n📝 User Query: {user_query}")
        topic = self.extract_topic(user_query)
        print(f"🎯 Extracted Topic: {topic}")
        
        # Step 2: Find relevant papers
        relevant_papers = self.find_relevant_papers(topic)
        
        # Step 3: Decide source
        source_type, resources = self.decide_source(
            topic, relevant_papers, download_enabled
        )
        
        # Step 4: Generate blog based on source
        metadata = {
            "topic": topic,
            "source_type": source_type,
            "papers_used": [],
            "query": user_query
        }
        
        if source_type == "papers":
            # Use existing papers
            print(f"\n📚 Using existing papers...")
            blog_content = await self._write_from_papers(
                topic, resources, style, word_count
            )
            metadata["papers_used"] = [Path(p).name for p in resources]
            
        elif source_type == "arxiv":
            # Download from arXiv then use
            print(f"\n📥 Downloading papers from arXiv...")
            try:
                downloaded_papers = download_arxiv_papers(
                    topic=topic,
                    max_results=num_papers_to_download,
                    download_dir=str(self.datasets_dir),
                    months_back=12
                )
                
                if downloaded_papers:
                    print(f"   ✅ Downloaded {len(downloaded_papers)} papers")
                    blog_content = await self._write_from_papers(
                        topic, downloaded_papers, style, word_count
                    )
                    metadata["papers_used"] = [Path(p).name for p in downloaded_papers]
                else:
                    print(f"   ⚠️  No papers downloaded, falling back to web search")
                    blog_content = await self._write_from_web(topic, style, word_count)
                    metadata["source_type"] = "web"
                    
            except Exception as e:
                print(f"   ❌ Error downloading papers: {e}")
                print(f"   🌐 Falling back to web search")
                blog_content = await self._write_from_web(topic, style, word_count)
                metadata["source_type"] = "web"
        
        else:  # web
            # Use web search
            print(f"\n🌐 Using web search...")
            blog_content = await self._write_from_web(topic, style, word_count)
        
        print(f"\n✅ Blog generation complete!")
        print("=" * 60)
        
        return blog_content, metadata
    
    async def _write_from_papers(
        self,
        topic: str,
        papers: List[str],
        style: str,
        word_count: int
    ) -> str:
        """Generate blog from papers using ResearchAssistant."""
        print(f"   📝 Generating blog from {len(papers)} papers...")
        
        # Create research assistant with papers
        assistant = ResearchAssistant(papers=papers, openai_api_key=self.openai_api_key)
        assistant.setup()
        
        # Generate blog
        blog_content = await assistant.write_blog_post(
            topic=topic,
            style=style,
            word_count=word_count,
            save_to_file=False  # Don't save, return content
        )
        
        return blog_content
    
    async def _write_from_web(
        self,
        topic: str,
        style: str,
        word_count: int
    ) -> str:
        """Generate blog using web search (fallback method)."""
        print(f"   🌐 Generating blog from web search...")
        
        # For now, use a simple prompt-based approach
        # In future, could integrate web_tools.py for actual web search
        
        style_guides = {
            "technical": "technical, academic style with precise terminology",
            "professional": "professional, business-oriented style",
            "casual": "casual, conversational style"
        }
        
        style_desc = style_guides.get(style, "professional style")
        
        prompt = f"""Write a comprehensive blog post about "{topic}" in a {style_desc}.

Requirements:
- Length: approximately {word_count} words
- Include an engaging introduction
- Cover key concepts and recent developments
- Provide practical insights
- End with a conclusion
- Use clear section headings

Write the blog post now:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert blog writer who creates engaging, informative content."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=word_count * 2  # Rough estimate
            )
            
            # Add rate limiting
            time.sleep(0.2)
            
            blog_content = response.choices[0].message.content.strip()
            return blog_content
            
        except Exception as e:
            return f"Error generating blog from web: {str(e)}"


# Example usage
async def main():
    """Example demonstrating smart blog writer."""
    
    # Initialize
    writer = SmartBlogWriter()
    
    # Example 1: Topic with existing papers
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Topic with existing papers")
    print("=" * 60)
    
    blog, metadata = await writer.write_smart_blog(
        user_query="Write a blog on autonomous agents",
        download_enabled=False,
        style="professional",
        word_count=300
    )
    
    print(f"\n📄 Generated Blog:\n{blog[:500]}...")
    print(f"\n📊 Metadata: {metadata}")
    
    # Example 2: Topic without papers, download enabled
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Topic without papers, download enabled")
    print("=" * 60)
    
    blog, metadata = await writer.write_smart_blog(
        user_query="Write a blog on quantum computing",
        download_enabled=True,
        num_papers_to_download=2,
        style="professional",
        word_count=300
    )
    
    print(f"\n📄 Generated Blog:\n{blog[:500]}...")
    print(f"\n📊 Metadata: {metadata}")


if __name__ == "__main__":
    asyncio.run(main())

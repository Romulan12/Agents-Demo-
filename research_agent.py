"""
Research Assistant Agent - Multi-Agent RAG with Web Search
Combines paper analysis with live web search capabilities
"""

import asyncio
from pathlib import Path
from typing import List, Dict
import os
import openai
import nest_asyncio
import httpx

# Fix SSL certificate verification issues
import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context

# Apply nest_asyncio to allow nested event loops (like in Jupyter)
nest_asyncio.apply()

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import QueryEngineTool

# Import our custom utilities
from utils import get_doc_tools
from web_tools import get_web_search_tools
from arxiv_downloader import download_arxiv_papers


class ResearchAssistant:
    """
    Multi-agent research assistant that combines paper analysis with web search.
    """
    
    def __init__(self, papers: List[str], openai_api_key: str = None):
        """
        Initialize the research assistant.
        
        Args:
            papers: List of paths to PDF papers
            openai_api_key: OpenAI API key (optional, will use env var if not provided)
        """
        import os
        
        # Get API key from parameter or environment variable
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            raise ValueError(
                "OpenAI API key not found. Please either:\n"
                "1. Pass it as a parameter: ResearchAssistant(papers, 'your-key')\n"
                "2. Set OPENAI_API_KEY environment variable"
            )
        
        self.papers = papers
        self.paper_to_tools_dict = {}
        self.openai_api_key = openai_api_key
        
        # Set OpenAI API key at module level (like in the working notebook)
        openai.api_key = openai_api_key
        
        # Create BOTH sync and async HTTP clients with SSL verification disabled
        # Sync client is used for embeddings, async client is used for agent queries
        http_client = httpx.Client(verify=False)
        async_http_client = httpx.AsyncClient(verify=False)
        
        # Configure LLM and embeddings GLOBALLY before any tool creation
        Settings.llm = OpenAI(
            model="gpt-3.5-turbo",
            api_key=openai_api_key,
            http_client=http_client,
            async_http_client=async_http_client
        )
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-ada-002",
            api_key=openai_api_key,
            http_client=http_client,
            async_http_client=async_http_client
        )
        
        self.llm = Settings.llm
        self.agent = None
        self.memory = ChatMemoryBuffer.from_defaults()
        
    def setup_tools(self):
        """
        Create tools for all papers and web search.
        """
        print("Setting up paper tools...")
        all_tools = []
        
        # Create tools for each paper
        for paper in self.papers:
            print(f"  Creating tools for: {paper}")
            path = Path(paper)
            vector_tool, summary_tool = get_doc_tools(paper, path.stem)
            self.paper_to_tools_dict[path.stem] = [vector_tool, summary_tool]
            all_tools.extend([vector_tool, summary_tool])
        
        # Add web search tools
        print("Adding web search tools...")
        web_tools = get_web_search_tools()
        all_tools.extend(web_tools)
        
        print(f"Total tools available: {len(all_tools)}")
        return all_tools
    
    def create_agent(self, tools: List):
        """
        Create the main research agent with all tools.
        
        Args:
            tools: List of all available tools
        """
        system_prompt = """You are an expert research assistant with access to:
1. Academic papers (via vector search and summarization tools)
2. Live web search (for current information and trends)

Your role is to:
- Answer questions by combining information from papers and web sources
- Compare historical research findings with current real-world applications
- Provide comprehensive, well-cited answers
- Use web search for recent developments, implementations, and benchmarks
- Use paper tools for theoretical foundations and original research

Always cite your sources and distinguish between paper findings and web information."""

        self.agent = FunctionAgent(
            tools=tools,
            llm=self.llm,
            system_prompt=system_prompt,
            verbose=True,
        )
        
        print("Research agent created successfully!")
    
    async def query(self, question: str, use_memory: bool = True) -> str:
        """
        Query the research assistant.
        
        Args:
            question: The research question
            use_memory: Whether to use conversation memory
            
        Returns:
            The agent's response
        """
        if self.agent is None:
            raise ValueError("Agent not initialized. Call setup() first.")
        
        if use_memory:
            response = await self.agent.run(question, memory=self.memory)
        else:
            response = await self.agent.run(question)
        
        return str(response)
    
    async def write_blog_post(
        self,
        topic: str,
        style: str = "professional",
        word_count: int = 200,
        save_to_file: bool = True,
        output_dir: str = "./blog_posts"
    ) -> str:
        """
        Generate a blog post based on research papers.
        
        Args:
            topic: Blog post topic/title
            style: Writing style - "technical", "casual", or "professional" (default: "professional")
            word_count: Target word count (default: 200)
            save_to_file: Whether to save to .txt file (default: True)
            output_dir: Directory to save blog posts (default: "./blog_posts")
            
        Returns:
            The generated blog post as plain text
            
        Example:
            blog = await assistant.write_blog_post(
                topic="The Future of RAG Systems",
                style="professional",
                word_count=200
            )
        """
        if self.agent is None:
            raise ValueError("Agent not initialized. Call setup() first.")
        
        # Define style guidelines
        style_guides = {
            "technical": "Write in a technical, academic style with precise terminology and detailed explanations.",
            "casual": "Write in a casual, conversational style that's easy to understand for general readers.",
            "professional": "Write in a professional, business-oriented style that's informative yet accessible."
        }
        
        style_guide = style_guides.get(style.lower(), style_guides["professional"])
        
        # Get list of papers for reference
        paper_list = "\n".join([f"- {Path(p).stem}" for p in self.papers])
        
        # Create the blog writing prompt
        prompt = f"""Write a blog post about "{topic}" based on the research papers you have access to.

Requirements:
- Style: {style_guide}
- Length: Approximately {word_count} words
- Structure: Title, introduction, body with key insights, conclusion
- Include citations to specific papers when mentioning findings
- Include references section at the end with paper titles and links (if available)
- Focus on synthesizing insights from multiple papers
- Make it engaging and informative

Available papers:
{paper_list}

Please write the blog post now."""

        print(f"\n{'='*60}")
        print(f"Generating blog post: '{topic}'")
        print(f"Style: {style}")
        print(f"Target length: ~{word_count} words")
        print(f"{'='*60}\n")
        
        # Generate the blog post
        blog_content = await self.query(prompt, use_memory=False)
        
        # Save to file if requested
        if save_to_file:
            import os
            from datetime import datetime
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Create filename
            date_str = datetime.now().strftime("%Y-%m-%d")
            topic_slug = topic.replace(" ", "_").replace("/", "_")[:50]
            filename = f"blog_{date_str}_{topic_slug}.txt"
            filepath = os.path.join(output_dir, filename)
            
            # Save the blog post
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(blog_content)
            
            print(f"\n✅ Blog post saved to: {filepath}\n")
        
        return blog_content
    
    def setup(self):
        """
        Complete setup: create tools and initialize agent.
        """
        tools = self.setup_tools()
        self.create_agent(tools)
    
    @classmethod
    def from_arxiv(
        cls,
        topic: str,
        num_papers: int = 3,
        existing_papers: List[str] = None,
        openai_api_key: str = None,
        download_dir: str = "./datasets",
        months_back: int = 12
    ):
        """
        Create ResearchAssistant with papers auto-downloaded from arXiv.
        
        Args:
            topic: Search topic (e.g., "RAG", "retrieval augmented generation")
            num_papers: Number of papers to download from arXiv (default: 3)
            existing_papers: Additional papers to include (optional)
            openai_api_key: OpenAI API key (optional, will use env var if not provided)
            download_dir: Directory to save downloaded papers (default: "./datasets")
            months_back: Only download papers from last N months (default: 12)
            
        Returns:
            ResearchAssistant instance with downloaded + existing papers
            
        Example:
            # Download 3 latest RAG papers
            assistant = ResearchAssistant.from_arxiv(
                topic="Retrieval Augmented Generation",
                num_papers=3
            )
            assistant.setup()
            
            # Or combine with existing papers
            assistant = ResearchAssistant.from_arxiv(
                topic="RAG",
                num_papers=3,
                existing_papers=["./datasets/my_paper.pdf"]
            )
        """
        print(f"\n{'='*60}")
        print(f"Creating Research Assistant with arXiv papers")
        print(f"Topic: {topic}")
        print(f"{'='*60}")
        
        # Download papers from arXiv
        downloaded_papers = download_arxiv_papers(
            topic=topic,
            max_results=num_papers,
            download_dir=download_dir,
            months_back=months_back
        )
        
        # Combine with existing papers if provided
        all_papers = downloaded_papers.copy()
        if existing_papers:
            print(f"\nAdding {len(existing_papers)} existing papers...")
            all_papers.extend(existing_papers)
        
        print(f"\nTotal papers to analyze: {len(all_papers)}")
        for i, paper in enumerate(all_papers, 1):
            print(f"  {i}. {Path(paper).name}")
        
        # Create and return ResearchAssistant instance
        return cls(papers=all_papers, openai_api_key=openai_api_key)


# Example usage
async def main():
    """
    Example demonstrating the research assistant.
    """
    import os
    
    # Configuration - reads from environment variable
    # Set your API key: export OPENAI_API_KEY="your-key-here"
    papers = [
        "./datasets/AutonomousDataAgents.pdf"
    ]
    
    # Initialize research assistant
    print("=" * 60)
    print("Initializing Research Assistant...")
    print("=" * 60)
    
    # API key will be read from OPENAI_API_KEY environment variable
    assistant = ResearchAssistant(papers)
    assistant.setup()
    
    print("\n" + "=" * 60)
    print("Research Assistant Ready!")
    print("=" * 60)
    
    # Example queries
    queries = [
        "What is the Autonomous Data Agents paper about?",
        "What are recent real-world implementations of autonomous agents in 2024?",
        "Compare the approaches in the papers with current industry trends",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 60}")
        print(f"Query {i}: {query}")
        print("=" * 60)
        
        response = await assistant.query(query)
        print(f"\nResponse:\n{response}\n")
        
        # Add delay between queries
        if i < len(queries):
            await asyncio.sleep(2)


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())

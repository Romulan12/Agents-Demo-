# 🧠 Agentic RAG System - Complete Overview

**Multi-Agent Research Assistant with Intelligent Routing**

---

## 🏗️ System Architecture

Your system is a **Multi-Agent RAG (Retrieval Augmented Generation)** platform with intelligent routing, combining academic paper analysis with live web search capabilities.

### **Core Components:**

```
┌─────────────────────────────────────────────────────────┐
│                    GRADIO UI                            │
│         (gradio_app_multi_agent.py)                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                 ROUTER AGENT                            │
│         (Intelligent Query Routing)                     │
│   Decides: Blog Writer vs Q&A Agent                     │
└────────────┬────────────────────────┬───────────────────┘
             │                        │
             ▼                        ▼
    ┌────────────────┐      ┌────────────────────┐
    │  BLOG WRITER   │      │    Q&A AGENT       │
    │   (Content)    │      │  (Custom ReAct)    │
    └────────┬───────┘      └─────────┬──────────┘
             │                        │
             └────────────┬───────────┘
                          ▼
              ┌───────────────────────┐
              │  RESEARCH ASSISTANT   │
              │   (Base System)       │
              └───────────┬───────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌─────────┐    ┌──────────┐    ┌──────────┐
    │ PAPER   │    │   WEB    │    │  ARXIV   │
    │ TOOLS   │    │  TOOLS   │    │DOWNLOADER│
    └─────────┘    └──────────┘    └──────────┘
```

---

## 📚 1. Query Tools (The Foundation)

### **A. Paper Tools** (`utils.py` - `get_doc_tools()`)

For **each PDF paper**, two specialized tools are created:

#### **1. Vector Tool** (`{paper_name}_vector_tool`)
```python
# Purpose: Retrieve specific context from papers
# Technology: VectorStoreIndex with OpenAI embeddings
# Use case: "What does paper X say about Y?"

Features:
- Chunks documents into 1024-token pieces (SentenceSplitter)
- Creates vector embeddings (text-embedding-ada-002)
- Semantic search for relevant passages
- Anti-hallucination prompt: Returns "No relevant information found" if context doesn't match
```

**Custom Prompt Template:**
```python
text_qa_template = PromptTemplate(
    """Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    If the context does not contain relevant information to answer the query, 
    respond with: "No relevant information found in this document."
    Do NOT use your general knowledge to answer.
    Query: {query_str}
    Answer: """
)
```

#### **2. Summary Tool** (`{paper_name}_summary_tool`)
```python
# Purpose: Summarize entire papers or sections
# Technology: SummaryIndex with tree summarization
# Use case: "Summarize the main contributions of paper X"

Features:
- Tree-based summarization (hierarchical)
- Async processing for speed
- Good for high-level overviews
- response_mode="tree_summarize"
```

**Example Tools Created:**
```
AutonomousDataAgents_vector_tool
AutonomousDataAgents_summary_tool
arxiv_2025-11-24_Fleming_vector_tool
arxiv_2025-11-24_Fleming_summary_tool
```

---

### **B. Web Tools** (`web_tools.py`)

Two tools for accessing current information:

#### **1. Web Search Tool** (`web_search`)
```python
# Purpose: Google search for current information
# Technology: googlesearch-python + BeautifulSoup
# Use case: "What are recent implementations of X?"

Features:
- Searches Google (up to 5 results by default)
- Extracts titles, snippets, URLs
- Scrapes meta descriptions from each result
- Returns formatted results with titles and descriptions
- Handles errors gracefully
```

**Function Signature:**
```python
def web_search(query: str, max_results: int = 5) -> str
```

#### **2. Web Scraper Tool** (`scrape_webpage`)
```python
# Purpose: Extract full content from specific URLs
# Technology: BeautifulSoup + requests
# Use case: Follow-up after web_search to get details

Features:
- Fetches and cleans webpage content
- Removes scripts, styles, navigation, footer, header
- Truncates to 5000 chars to avoid token limits
- Returns clean text content
- User-Agent spoofing to avoid blocks
```

**Function Signature:**
```python
def scrape_webpage(url: str) -> str
```

---

## 🤖 2. Agent System

### **A. Research Assistant** (`research_agent.py`)

**The base system** that manages all tools and provides the foundation.

```python
class ResearchAssistant:
    """
    Multi-agent research assistant that combines paper analysis with web search.
    """
    
    # Core Methods:
    def __init__(papers, openai_api_key)     # Initialize with papers
    def setup_tools()                        # Create paper + web tools
    def create_agent(tools)                  # Initialize FunctionAgent
    def setup()                              # Complete setup (tools + agent)
    def query(question, use_memory=True)     # Answer questions
    def write_blog_post(...)                 # Generate blog content
    
    # Class Method:
    @classmethod
    def from_arxiv(topic, num_papers, ...)  # Auto-download from arXiv
```

**Key Features:**
- Uses **LlamaIndex FunctionAgent** for tool orchestration
- Manages `paper_to_tools_dict` mapping papers to their tools
- Configures OpenAI LLM (gpt-3.5-turbo) and embeddings (ada-002)
- Handles SSL/async HTTP clients with verification disabled
- Memory management with `ChatMemoryBuffer`
- Blog post generation with customizable style and length

**System Prompt:**
```
"You are an expert research assistant with access to:
1. Academic papers (via vector search and summarization)
2. Live web search (for current information)

Your role is to:
- Combine information from papers and web sources
- Compare historical research with current applications
- Provide comprehensive, well-cited answers
- Use web search for recent developments
- Use paper tools for theoretical foundations"
```

**Configuration:**
```python
# Global Settings
Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=...)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key=...)

# HTTP Clients (SSL disabled)
http_client = httpx.Client(verify=False)
async_http_client = httpx.AsyncClient(verify=False)
```

---

### **B. Custom ReAct Agent** (`react_agent.py`)

**⚠️ IMPORTANT:** You're using a **custom ReAct implementation** (not LlamaIndex's built-in) because the built-in version gave errors.

```python
class ReActAgent:
    """
    Custom ReAct (Reasoning + Acting) Agent
    
    Implements autonomous multi-step reasoning:
    1. Thought: Reason about what to do next
    2. Action: Select and execute a tool
    3. Observation: Observe the result
    4. Repeat until task is complete
    """
    
    def __init__(tools, llm, max_iterations=5, verbose=True)
    def run(question) -> Dict[str, Any]
    
    # Internal Methods:
    def _create_react_prompt(question, history)
    def _parse_llm_response(response)
    def _execute_tool(tool_name, query)
    def _generate_final_answer(question, history)
```

**ReAct Loop:**
```python
for iteration in range(1, max_iterations + 1):
    # 1. Generate reasoning prompt with history
    prompt = self._create_react_prompt(question, history)
    
    # 2. Get LLM response
    response = await self.llm.acomplete(prompt)
    
    # 3. Parse: Thought, Action, Action Input
    thought, action, action_input = await self._parse_llm_response(response.text)
    
    # 4. Check if finished
    if action.upper() == "FINISH":
        break
    
    # 5. Execute tool
    observation = await self._execute_tool(action, action_input)
    
    # 6. Record step
    history.append(ReActStep(...))

# 7. Generate final answer from history
final_answer = await self._generate_final_answer(question, history)
```

**Anti-Hallucination System Prompt:**
```
CRITICAL RULES - MUST FOLLOW:
- ONLY report information that is EXPLICITLY in the tool output
- If a tool returns no relevant information, acknowledge it and try another tool
- DO NOT use your general knowledge to fill in gaps or make assumptions
- If you cannot find information in papers, use web search tools
- ALWAYS cite the source of your information (which tool/paper provided it)
- If all paper tools fail, you MUST try web search before finishing

Verification After Each Observation:
- Ask yourself: "Does this observation actually contain relevant information?"
- If observation is irrelevant, empty, or off-topic, acknowledge it and try different tool
- If you've tried 3+ paper tools without finding relevant info, switch to web search
- Never claim to have found information if the observation doesn't explicitly contain it

Format your response EXACTLY as:
Thought: [your reasoning about what to do next]
Action: [tool_name OR "FINISH"]
Action Input: [query for the tool, or "N/A" if FINISH]
```

**Wrapper Class:**
```python
class ReActResearchAssistant:
    """
    Wrapper that integrates custom ReAct agent with ResearchAssistant.
    """
    
    def __init__(base_assistant, max_iterations=5, verbose=True)
    def setup()                              # Create ReAct agent with tools
    def query(question) -> Dict[str, Any]    # Query with reasoning trace
    def query_simple(question) -> str        # Query, return just answer
```

**Return Format:**
```python
{
    "answer": "Final synthesized answer with citations",
    "reasoning_steps": [
        {
            "iteration": 1,
            "thought": "I need to find...",
            "action": "AutonomousDataAgents_vector_tool",
            "action_input": {"query": "..."},
            "observation": "According to the paper..."
        },
        # ... more steps
    ],
    "total_iterations": 3
}
```

---

### **C. Router Agent** (`router_agent.py`)

**Intelligent query routing** using LLM reasoning.

```python
class RouterAgent:
    """
    Autonomous router that decides which specialized agent 
    should handle a user query using LLM reasoning.
    """
    
    def __init__(openai_api_key)
    async def route(user_query) -> Dict
    def explain_routing(decision, user_query) -> str
    
    # Internal:
    def _format_agents_info() -> str
    def _fallback_route(query) -> Dict  # Keyword-based backup
```

**Available Agents:**
```python
agents = {
    "blog_writer": {
        "description": "Generates blog posts, articles, and written content",
        "capabilities": ["Create blog posts", "Generate articles", ...],
        "keywords": ["write", "blog", "article", "post", "create", ...],
        "examples": ["Write a blog on X", "Create an article about Y"]
    },
    "qa_agent": {
        "description": "Answers questions using document analysis",
        "capabilities": ["Answer questions", "Analyze documents", ...],
        "keywords": ["what", "how", "why", "explain", "compare", ...],
        "examples": ["What are X?", "How does Y work?"]
    }
}
```

**Routing Decision:**
```json
{
    "agent": "blog_writer" or "qa_agent",
    "reasoning": "Detailed explanation of why this agent was chosen...",
    "confidence": 0.95,
    "alternative": "backup_agent_name"
}
```

**Routing Logic:**

1. **LLM-Powered (Primary):**
   - Analyzes user intent with GPT-3.5
   - Considers full context, not just keywords
   - Provides detailed reasoning
   - Returns confidence score (0.0-1.0)

2. **Fallback (If LLM Fails):**
   - Priority 1: Summary/explanation keywords → Q&A
   - Priority 2: Explicit blog keywords → Blog Writer
   - Priority 3: Question words → Q&A
   - Priority 4: Default to Q&A

**Key Insight:**
```python
# "write a summary" → Q&A Agent (NOT Blog Writer!)
# The word "write" alone doesn't mean blog - considers full context
```

---

## 🎯 3. Complete Tool Inventory

### **Per-Paper Tools (2 per paper):**
```
For each PDF in datasets/:
  1. {paper_name}_vector_tool    - Semantic search
  2. {paper_name}_summary_tool   - Summarization
```

### **Web Tools (2 total):**
```
1. web_search        - Google search
2. scrape_webpage    - URL content extraction
```

### **Example with 4 Papers:**
```
Total Tools: 10
├── Paper Tools: 8
│   ├── AutonomousDataAgents_vector_tool
│   ├── AutonomousDataAgents_summary_tool
│   ├── arxiv_2025-11-20_Janjusevic_vector_tool
│   ├── arxiv_2025-11-20_Janjusevic_summary_tool
│   ├── arxiv_2025-11-24_Fleming_vector_tool
│   ├── arxiv_2025-11-24_Fleming_summary_tool
│   ├── arxiv_2025-11-25_Errico_vector_tool
│   └── arxiv_2025-11-25_Errico_summary_tool
└── Web Tools: 2
    ├── web_search
    └── scrape_webpage
```

---

## 🔄 4. Query Flow Example

**User Query:** *"What are autonomous data agents and how are they used in industry?"*

### **Step 1: Router Agent**
```
🎯 LLM Analysis:
- Intent: Information seeking (question format)
- Expected output: Explanation/answer
- Keywords: "what", "how"

Decision:
  Agent: qa_agent
  Confidence: 90%
  Reasoning: "Question seeking explanation about technical topic"
  Alternative: blog_writer
```

### **Step 2: Custom ReAct Agent (Multi-Step Reasoning)**
```
Iteration 1:
  Thought: "Need to understand autonomous data agents from paper"
  Action: AutonomousDataAgents_vector_tool
  Action Input: "autonomous data agents definition"
  Observation: "According to the paper, autonomous data agents are 
                systems that can independently perform data-related 
                tasks such as collection, processing, and analysis..."

Iteration 2:
  Thought: "Got theory from paper. Now need current industry applications.
            Paper might not have recent info, should try web search."
  Action: web_search
  Action Input: "autonomous data agents industry applications 2024"
  Observation: "Found 5 results:
                1. 'How Companies Use AI Agents in Production'
                2. 'Autonomous Agents in Enterprise Systems'
                ..."

Iteration 3:
  Thought: "Found relevant article about production use. Get full details."
  Action: scrape_webpage
  Action Input: "https://example.com/agents-in-production"
  Observation: "Article content: Companies like Salesforce and Microsoft 
                are implementing autonomous agents for customer service,
                data processing, and workflow automation..."

Final Answer:
  "According to the AutonomousDataAgents paper, autonomous data agents 
   are systems that can independently perform data-related tasks such as 
   collection, processing, and analysis [AutonomousDataAgents_vector_tool].
   
   In industry, web search reveals that companies like Salesforce and 
   Microsoft are implementing these systems for customer service, data 
   processing, and workflow automation [web_search, scrape_webpage]."
```

---

## 📊 5. Technology Stack

### **Core Framework:**
- **LlamaIndex** - RAG orchestration and agent framework
- **OpenAI** - LLM (gpt-3.5-turbo) + Embeddings (text-embedding-ada-002)
- **Gradio** - Web UI (ports 7861, 7862)

### **Document Processing:**
- **SimpleDirectoryReader** - PDF loading
- **SentenceSplitter** - Chunking (1024 tokens)
- **VectorStoreIndex** - Semantic search with embeddings
- **SummaryIndex** - Hierarchical summarization

### **Web Integration:**
- **googlesearch-python** - Google search API
- **BeautifulSoup4** - HTML parsing and cleaning
- **requests** - HTTP client for web scraping
- **httpx** - Async HTTP client (with SSL disabled)

### **Agent Framework:**
- **FunctionAgent** - LlamaIndex's tool orchestration agent
- **Custom ReActAgent** - Manual ReAct implementation (~400 lines)
- **ChatMemoryBuffer** - Conversation memory management

### **Utilities:**
- **nest_asyncio** - Nested event loop support (Jupyter compatibility)
- **asyncio** - Async/await support
- **ssl/certifi** - SSL certificate handling (disabled for compatibility)

---

## 🎨 6. User Interfaces

### **Multi-Agent App** (Port 7862) - **RECOMMENDED** ⭐
```python
# gradio_app_multi_agent.py
Features:
✅ LLM-powered Router Agent with reasoning
✅ Confidence scores and explanations
✅ Blog generation with style/length options
✅ Q&A with Custom ReAct reasoning (5 iterations)
✅ arXiv auto-download integration
✅ Web search integration
✅ Reasoning trace display (optional)
✅ Download generated blogs as .txt
```

### **Single-Agent App** (Port 7861) - **Simpler**
```python
# gradio_app_single_agent.py
Features:
✅ Keyword-based routing (no LLM call)
✅ Blog generation
✅ Q&A with Custom ReAct reasoning
✅ arXiv auto-download
✅ Faster (one less API call)
✅ Good for testing/prototyping
```

---

## 📁 7. File Structure

```
/Users/misharana/Desktop/Work/Agentic/
├── utils.py                      # Tool creation (get_doc_tools)
├── research_agent.py             # Base ResearchAssistant class
├── react_agent.py                # Custom ReAct implementation
├── router_agent.py               # LLM-powered routing
├── web_tools.py                  # Web search & scraping
├── arxiv_downloader.py           # Auto-download from arXiv
├── smart_blog_writer.py          # Blog generation logic
├── gradio_app_multi_agent.py    # Main UI (port 7862)
├── gradio_app_single_agent.py   # Simple UI (port 7861)
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── notes.md                      # This file!
│
├── datasets/                     # PDF papers storage
│   ├── AutonomousDataAgents.pdf
│   ├── arxiv_2025-11-24_Fleming_*.pdf
│   └── ...
│
├── blog_posts/                   # Generated blog posts
│   ├── blog_2025-11-27_*.txt
│   └── ...
│
├── docs/                         # Additional documentation
│   ├── AGENTS.md
│   ├── ARCHITECTURE.md
│   ├── EXAMPLES.md
│   └── FALLBACKS.md
│
└── backup/                       # Backup of all files
    └── (mirrors main directory)
```

---

## 🚀 8. Usage Examples

### **A. Basic Q&A**
```python
from research_agent import ResearchAssistant
import asyncio

# Initialize
papers = ["./datasets/AutonomousDataAgents.pdf"]
assistant = ResearchAssistant(papers, openai_api_key="your-key")
assistant.setup()

# Query
response = await assistant.query("What are autonomous data agents?")
print(response)
```

### **B. With Custom ReAct**
```python
from research_agent import ResearchAssistant
from react_agent import ReActResearchAssistant

# Initialize base assistant
papers = ["./datasets/AutonomousDataAgents.pdf"]
base_assistant = ResearchAssistant(papers, openai_api_key="your-key")
base_assistant.setup()

# Wrap with ReAct
react_assistant = ReActResearchAssistant(
    base_assistant=base_assistant,
    max_iterations=5,
    verbose=True
)
react_assistant.setup()

# Query with reasoning trace
result = await react_assistant.query("What are the main contributions?")
print(result["answer"])
print(f"Iterations: {result['total_iterations']}")
```

### **C. Auto-Download from arXiv**
```python
# Download papers and create assistant in one step
assistant = ResearchAssistant.from_arxiv(
    topic="Retrieval Augmented Generation",
    num_papers=3,
    openai_api_key="your-key"
)
assistant.setup()

response = await assistant.query("What are recent advances in RAG?")
```

### **D. Generate Blog Post**
```python
blog = await assistant.write_blog_post(
    topic="The Future of RAG Systems",
    style="professional",  # or "technical", "casual"
    word_count=500,
    save_to_file=True
)
print(blog)
```

### **E. Using Gradio UI**
```bash
# Multi-agent app (recommended)
python gradio_app_multi_agent.py
# Open browser to http://localhost:7862

# Single-agent app (simpler)
python gradio_app_single_agent.py
# Open browser to http://localhost:7861
```

---

## 🎯 9. Key Features Summary

### **🧠 Intelligence**
- ✅ Multi-step autonomous reasoning (Custom ReAct)
- ✅ LLM-powered query routing with confidence scores
- ✅ Anti-hallucination protection (strict source citation)
- ✅ Fallback mechanisms (keyword-based routing, web search)

### **📚 Knowledge Sources**
- ✅ Academic papers (vector search + summarization)
- ✅ Live web search (Google + webpage scraping)
- ✅ arXiv auto-download (recent papers)
- ✅ Hybrid knowledge (papers + current web info)

### **🛠️ Tools**
- ✅ 2 tools per paper (vector + summary)
- ✅ 2 web tools (search + scrape)
- ✅ Automatic tool creation from PDFs
- ✅ Tool name truncation (OpenAI 64-char limit)

### **🤖 Agents**
- ✅ Router Agent (LLM-powered intent detection)
- ✅ Blog Writer (content generation)
- ✅ Q&A Agent (Custom ReAct reasoning)
- ✅ Research Assistant (base orchestration)

### **💡 User Experience**
- ✅ Two Gradio UIs (simple + advanced)
- ✅ Reasoning trace display (optional)
- ✅ Blog download as .txt
- ✅ Configurable parameters (style, length, iterations)
- ✅ Progress indicators

---

## 🔧 10. Configuration

### **Environment Variables**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### **LLM Settings**
```python
# In research_agent.py
Settings.llm = OpenAI(
    model="gpt-3.5-turbo",
    api_key=openai_api_key,
    temperature=0.7  # Default for generation
)

# In router_agent.py
self.llm = OpenAI(
    model="gpt-3.5-turbo",
    temperature=0.1  # Low for consistent routing
)
```

### **ReAct Settings**
```python
# In react_agent.py
ReActAgent(
    tools=all_tools,
    llm=llm,
    max_iterations=5,    # Max reasoning steps
    verbose=True         # Print reasoning trace
)
```

### **Document Chunking**
```python
# In utils.py
splitter = SentenceSplitter(
    chunk_size=1024,     # Tokens per chunk
    chunk_overlap=20     # Overlap between chunks
)
```

---

## 📈 11. Performance Considerations

### **Token Usage**
- Vector search: ~1000-2000 tokens per query
- Summary: ~2000-3000 tokens per query
- ReAct reasoning: ~500-1000 tokens per iteration
- Final answer generation: ~1000-2000 tokens
- **Total per query:** ~5000-15000 tokens (depending on iterations)

### **API Calls**
- Router Agent: 1 call (if using multi-agent app)
- ReAct iterations: 1 call per iteration (max 5)
- Final answer: 1 call
- **Total:** 2-7 API calls per query

### **Speed**
- Simple query: 5-10 seconds
- Complex query (5 iterations): 15-30 seconds
- Blog generation: 30-60 seconds
- arXiv download: 10-30 seconds per paper

---

## 🎓 12. Best Practices

### **For Accurate Results:**
1. ✅ Use specific, clear questions
2. ✅ Enable web search for current information
3. ✅ Increase max_iterations for complex queries
4. ✅ Review reasoning trace to understand agent's process
5. ✅ Provide relevant papers in datasets/

### **For Blog Generation:**
1. ✅ Be specific about topic
2. ✅ Choose appropriate style (technical/professional/casual)
3. ✅ Set realistic word count (300-800 words)
4. ✅ Include relevant papers for better content
5. ✅ Enable arXiv download for current research

### **For Development:**
1. ✅ Use verbose=True to debug reasoning
2. ✅ Check reasoning_steps in response
3. ✅ Monitor token usage
4. ✅ Test with simple queries first
5. ✅ Use single-agent app for faster iteration

---

## 🐛 13. Known Issues & Solutions

### **Issue: LlamaIndex ReActAgent Errors**
**Solution:** Using custom ReAct implementation instead (current setup)

### **Issue: SSL Certificate Verification**
**Solution:** Disabled SSL verification in HTTP clients
```python
ssl._create_default_https_context = ssl._create_unverified_context
http_client = httpx.Client(verify=False)
```

### **Issue: Tool Name Too Long**
**Solution:** Truncate paper names to 50 chars in utils.py
```python
max_name_length = 50
doc_name = doc_name[:max_name_length]
```

### **Issue: Web Search Blocked**
**Solution:** User-Agent spoofing in web_tools.py
```python
headers = {'User-Agent': 'Mozilla/5.0 ...'}
```

---

## 🎯 Summary

Your **Agentic RAG System** is a sophisticated multi-agent platform that:

1. **Intelligently routes queries** using LLM-powered Router Agent
2. **Combines academic papers with live web search** for comprehensive answers
3. **Uses custom ReAct reasoning** for multi-step autonomous problem solving
4. **Prevents hallucinations** through strict source citation requirements
5. **Generates blog posts** with customizable style and length
6. **Auto-downloads papers** from arXiv for current research
7. **Provides transparent reasoning** with detailed trace display
8. **Offers two UIs** (simple keyword-based + advanced LLM-powered)

**Key Strength:** Bridges the gap between academic research and current real-world applications! 🚀

---

**Created:** November 27, 2025  
**System Version:** Custom ReAct Implementation  
**Status:** Production Ready ✅

# 🏗️ System Architecture

This document provides a detailed overview of the Advanced Agentic RAG system architecture, including multi-agent design, data flow, and component interactions.

---

## 📊 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                        │
│                      (Gradio Web UI)                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    ROUTER AGENT (Meta-Agent)                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  LLM-Powered Intent Detection                          │ │
│  │  - Analyzes user query                                 │ │
│  │  - Determines intent (Q&A vs Blog)                     │ │
│  │  - Provides confidence score                           │ │
│  │  - Fallback: Keyword-based routing                     │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────┬───────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
┌──────────────────────┐    ┌──────────────────────┐
│   Q&A AGENT          │    │   BLOG WRITER AGENT  │
│   (ReAct Pattern)    │    │   (SmartBlogWriter)  │
└──────────────────────┘    └──────────────────────┘
              │                       │
              ▼                       ▼
┌──────────────────────┐    ┌──────────────────────┐
│  ResearchAssistant   │    │  Topic Extraction    │
│  (Base)              │    │  Source Selection    │
│  - Paper loading     │    │  Content Generation  │
│  - Tool creation     │    │                      │
│  - Web search        │    │  Fallback: Web-based │
└──────────┬───────────┘    └──────────────────────┘
           │
           ▼
┌──────────────────────┐
│  ReActResearchAsst   │
│  (Wrapper)           │
│  - Multi-step        │
│  - Reasoning trace   │
│  - Tool selection    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────┐
│         TOOL ECOSYSTEM               │
│  ┌────────────────────────────────┐  │
│  │  Paper Tools (24)              │  │
│  │  - vector_search × 12          │  │
│  │  - summary × 12                │  │
│  └────────────────────────────────┘  │
│  ┌────────────────────────────────┐  │
│  │  Web Tools (2)                 │  │
│  │  - search_web                  │  │
│  │  - search_news                 │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
```

---

## 🎯 Multi-Agent System Design

### Agent Hierarchy

The system implements a **hierarchical multi-agent architecture** with three distinct agents:

```
Level 1: Meta-Agent (Orchestration)
    └─ Router Agent

Level 2: Specialized Agents (Execution)
    ├─ Q&A Agent (ReAct)
    └─ Blog Writer Agent

Level 3: Tools (Resources)
    ├─ Paper Tools (24)
    └─ Web Tools (2)
```

### Agent Characteristics

| Agent | Type | Pattern | Iterations | Tools | Purpose |
|-------|------|---------|------------|-------|---------|
| **Router** | Meta | Single-shot LLM | 1 | None | Route queries |
| **Q&A** | Reasoning | ReAct | 3-10 | 26 | Answer questions |
| **Blog** | Generation | Direct | 1 | Search | Generate content |

---

## 🔄 Data Flow

### 1. Query Processing Flow

```
User Query
    ↓
[1] Router Agent
    ├─ Parse query
    ├─ LLM reasoning
    ├─ Intent detection
    └─ Confidence scoring
    ↓
[2] Route Decision
    ├─ If Q&A: → Q&A Agent
    └─ If Blog: → Blog Writer
    ↓
[3] Agent Execution
    ├─ Q&A: Multi-step reasoning
    └─ Blog: Content generation
    ↓
[4] Result Synthesis
    └─ Format output
    ↓
User Interface
```

### 2. Q&A Agent Flow (ReAct Pattern)

```
Question
    ↓
[1] Initialize
    ├─ Load papers
    ├─ Create tools (24 paper + 2 web)
    └─ Setup ReAct agent
    ↓
[2] ReAct Loop (Iteration 1-N)
    ├─ THINK: Analyze what's needed
    ├─ ACT: Select and execute tool
    ├─ OBSERVE: Process tool result
    └─ DECIDE: Continue or finish?
    ↓
[3] Synthesis
    ├─ Combine observations
    ├─ Generate final answer
    └─ Include reasoning trace
    ↓
Formatted Answer
```

### 3. Blog Writer Flow

```
Blog Request
    ↓
[1] Topic Extraction
    └─ LLM extracts topic from query
    ↓
[2] Source Selection
    ├─ Search for relevant papers
    ├─ If found: Use papers
    └─ If not: Use web search
    ↓
[3] Content Generation
    ├─ Apply style (Technical/Professional/Casual)
    ├─ Apply length (300/500/800 words)
    └─ Generate structured content
    ↓
Blog Post
```

---

## 🧩 Component Breakdown

### 1. Router Agent

**File**: `router_agent.py`

**Architecture**:
```python
RouterAgent
    ├─ LLM (OpenAI)
    ├─ Agent Definitions
    │   ├─ blog_writer
    │   └─ qa_agent
    └─ Routing Logic
        ├─ Primary: LLM reasoning
        └─ Fallback: Keyword matching
```

**Decision Process**:
```
Input: User query
    ↓
LLM Prompt:
    "Analyze query step-by-step:
     1. What is the user's intent?
     2. What output do they expect?
     3. Which agent is best suited?"
    ↓
LLM Response:
    {
        "agent": "qa_agent",
        "reasoning": "...",
        "confidence": 0.95
    }
    ↓
Output: Routing decision
```

**Fallback Logic**:
```python
if LLM_fails:
    if "blog" in query or "write" in query:
        return "blog_writer"
    else:
        return "qa_agent"
```

---

### 2. Q&A Agent (ReAct)

**Files**: `research_agent.py`, `react_agent.py`

**Architecture**:
```
ReActResearchAssistant (Wrapper)
    └─ wraps
        ↓
    ResearchAssistant (Base)
        ├─ Paper Management
        ├─ Tool Creation
        └─ Web Search Integration
```

**Wrapper Pattern**:
```python
# Base provides infrastructure
base = ResearchAssistant(papers)
base.setup()  # Creates tools

# Wrapper adds ReAct reasoning
react = ReActResearchAssistant(base)
react.setup()  # Extracts tools, creates ReAct agent

# Use wrapper for queries
result = await react.query(question)
```

**ReAct Loop**:
```
for iteration in range(1, max_iterations + 1):
    # Step 1: Think
    thought = LLM.reason(question, history)
    
    # Step 2: Act
    action = LLM.select_tool(thought)
    observation = execute_tool(action)
    
    # Step 3: Record
    history.append(thought, action, observation)
    
    # Step 4: Decide
    if action == "FINISH":
        break

# Step 5: Synthesize
final_answer = LLM.synthesize(history)
```

---

### 3. Blog Writer Agent

**File**: `smart_blog_writer.py`

**Architecture**:
```
SmartBlogWriter
    ├─ Topic Extractor (LLM)
    ├─ Paper Searcher
    │   └─ Filename-based matching
    ├─ Source Selector
    │   ├─ Papers available? → Use papers
    │   └─ No papers? → Use web
    └─ Content Generator
        ├─ ResearchAssistant.write_blog_post()
        └─ Web-based generation
```

**Decision Tree**:
```
Blog Request
    ↓
Extract Topic
    ↓
Search Papers (filename match)
    ↓
Papers Found?
    ├─ YES → Use ResearchAssistant
    │         └─ Generate from papers
    └─ NO → Check Download Enabled?
              ├─ YES → Download from arXiv
              │         └─ Generate from papers
              └─ NO → Use Web Search
                        └─ Generate from web
```

---

## 🛠️ Tool Ecosystem

### Tool Types

**1. Paper Tools (24 total)**
```
For each paper (12 papers):
    ├─ vector_search_tool
    │   └─ Semantic search in paper content
    └─ summary_tool
        └─ Get paper summary
```

**2. Web Tools (2 total)**
```
├─ search_web
│   └─ General web search (DuckDuckGo)
└─ search_news
    └─ Recent news search
```

### Tool Creation Process

```python
# In ResearchAssistant.setup()
for paper in papers:
    # Create vector index
    index = VectorStoreIndex.from_documents(paper)
    
    # Create vector search tool
    vector_tool = QueryEngineTool(
        query_engine=index.as_query_engine(),
        metadata=ToolMetadata(
            name=f"{paper_name}_vector_tool",
            description=f"Search {paper_name} content"
        )
    )
    
    # Create summary tool
    summary_tool = QueryEngineTool(
        query_engine=index.as_query_engine(
            response_mode="tree_summarize"
        ),
        metadata=ToolMetadata(
            name=f"{paper_name}_summary_tool",
            description=f"Get summary of {paper_name}"
        )
    )
    
    tools.append(vector_tool)
    tools.append(summary_tool)
```

---

## 🔗 Integration Points

### 1. LlamaIndex Integration

```
LlamaIndex Components:
    ├─ SimpleDirectoryReader
    │   └─ Load PDFs
    ├─ VectorStoreIndex
    │   └─ Create embeddings
    ├─ QueryEngine
    │   └─ Search and retrieve
    └─ ToolMetadata
        └─ Tool descriptions
```

### 2. OpenAI Integration

```
OpenAI APIs:
    ├─ Chat Completions (gpt-4o-mini)
    │   ├─ Router reasoning
    │   ├─ ReAct reasoning
    │   └─ Content generation
    └─ Embeddings (text-embedding-3-small)
        └─ Vector search
```

### 3. Gradio Integration

```
Gradio UI:
    ├─ Input Components
    │   ├─ Textbox (query)
    │   ├─ Checkboxes (options)
    │   ├─ Sliders (parameters)
    │   └─ File upload
    └─ Output Components
        ├─ Textbox (results)
        └─ Download button
```

---

## 📈 Scalability Considerations

### Adding New Agents

To add a new specialized agent:

```python
# 1. Define in router_agent.py
self.agents = {
    "blog_writer": {...},
    "qa_agent": {...},
    "code_generator": {  # New agent
        "name": "code_generator",
        "description": "Generates code based on requirements",
        "capabilities": ["code generation", "debugging"]
    }
}

# 2. Implement agent class
class CodeGeneratorAgent:
    def generate(self, requirements):
        # Implementation
        pass

# 3. Add routing logic in gradio_app
if routing_decision["agent"] == "code_generator":
    agent = CodeGeneratorAgent()
    result = agent.generate(user_query)
```

### Horizontal Scaling

```
Current: Single instance
    └─ All agents in one process

Future: Distributed
    ├─ Router Agent (API Gateway)
    ├─ Q&A Agent (Worker Pool)
    ├─ Blog Agent (Worker Pool)
    └─ Load Balancer
```

### Performance Optimization

**Current Bottlenecks**:
1. Sequential tool execution in ReAct
2. Paper loading on every request
3. No caching of embeddings

**Potential Improvements**:
```python
# 1. Parallel tool execution
async def execute_tools_parallel(tools):
    results = await asyncio.gather(*[
        tool.execute() for tool in tools
    ])
    return results

# 2. Cache paper embeddings
@lru_cache(maxsize=100)
def load_paper_index(paper_path):
    return VectorStoreIndex.from_documents(paper_path)

# 3. Persistent vector store
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=StorageContext.from_defaults(
        persist_dir="./storage"
    )
)
```

---

## 🔒 Security Considerations

### API Key Management
```python
# Environment variable (current)
api_key = os.getenv("OPENAI_API_KEY")

# Future: Secrets manager
from secrets_manager import get_secret
api_key = get_secret("openai_api_key")
```

### Input Validation
```python
# Validate user input
def validate_query(query: str) -> bool:
    if not query or len(query) > 10000:
        return False
    # Add more validation
    return True
```

### Rate Limiting
```python
# Future: Add rate limiting
from ratelimit import limits

@limits(calls=10, period=60)
def process_query(query):
    # Process query
    pass
```

---

## 📊 System Metrics

### Performance Metrics

| Component | Metric | Target | Current |
|-----------|--------|--------|---------|
| Router | Decision time | <2s | ~1s |
| Q&A Agent | Query time | <30s | 10-30s |
| Blog Writer | Generation time | <45s | 15-45s |
| Paper Loading | Load time | <2s/paper | ~1s/paper |

### Resource Usage

```
Memory:
    ├─ Base: ~500MB
    ├─ Per paper: ~50MB
    └─ Peak: ~2GB (12 papers)

API Calls:
    ├─ Router: 1 call
    ├─ Q&A: 5-10 calls (ReAct iterations)
    └─ Blog: 3-5

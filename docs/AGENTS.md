# 🤖 Agent Documentation

This document provides detailed information about each agent in the Advanced Agentic RAG system, including their capabilities, decision-making processes, and implementation details.

---

## 📋 Table of Contents

1. [Router Agent](#router-agent)
2. [Q&A Agent (ReAct)](#qa-agent-react)
3. [Blog Writer Agent](#blog-writer-agent)
4. [Tool Ecosystem](#tool-ecosystem)
5. [Agent Comparison](#agent-comparison)

---

## 🎯 Router Agent

### Overview

The Router Agent is a **meta-agent** that orchestrates the entire system by intelligently routing user queries to the appropriate specialized agent.

### Key Characteristics

| Property | Value |
|----------|-------|
| **Type** | Meta-agent (Orchestrator) |
| **Pattern** | Single-shot LLM reasoning |
| **File** | `router_agent.py` |
| **LLM Calls** | 1 per query |
| **Tools** | None |
| **Fallback** | Keyword-based routing |

### Capabilities

✅ **LLM-Powered Intent Detection**
- Analyzes user query semantically
- Understands context and nuance
- No hardcoded rules

✅ **Confidence Scoring**
- Provides confidence percentage (0-100%)
- Helps identify ambiguous queries
- Enables monitoring and improvement

✅ **Explainable Routing**
- Provides reasoning for decisions
- Transparent decision-making
- Helps users understand routing

✅ **Fallback Mechanism**
- Keyword-based routing if LLM fails
- Ensures system reliability
- Graceful degradation

### Decision Process

```python
async def route(self, user_query: str) -> Dict:
    """
    Route user query to appropriate agent.
    
    Returns:
        {
            "agent": "qa_agent" | "blog_writer",
            "reasoning": "Explanation of decision",
            "confidence": 0.0-1.0
        }
    """
```

**Step-by-Step**:

```
1. Receive user query
   ↓
2. Build LLM prompt with:
   - Agent definitions
   - Capabilities
   - User query
   ↓
3. LLM analyzes:
   - User's intent
   - Expected output type
   - Best suited agent
   ↓
4. Parse LLM response (JSON)
   ↓
5. Validate decision
   ↓
6. Return routing decision
```

### Agent Definitions

```python
self.agents = {
    "blog_writer": {
        "name": "blog_writer",
        "description": "Generates blog posts, articles, and written content on topics",
        "capabilities": [
            "blog post generation",
            "article writing",
            "content creation",
            "topic research"
        ],
        "best_for": [
            "requests to write blogs",
            "content generation tasks",
            "article creation"
        ]
    },
    "qa_agent": {
        "name": "qa_agent",
        "description": "Answers questions using document analysis and multi-step reasoning",
        "capabilities": [
            "question answering",
            "document analysis",
            "information synthesis",
            "multi-step reasoning"
        ],
        "best_for": [
            "questions about documents",
            "information requests",
            "analysis tasks"
        ]
    }
}
```

### Routing Examples

**Example 1: Blog Request**
```
Query: "Write a blog about computer vision"

LLM Reasoning:
- User explicitly requests "write a blog"
- Output should be a blog post
- Content creation task

Decision:
- Agent: blog_writer
- Confidence: 95%
- Reasoning: "User explicitly requested blog generation"
```

**Example 2: Question**
```
Query: "What are the main approaches in RAG systems?"

LLM Reasoning:
- User asks a question ("What are...")
- Expects informational answer
- Requires document analysis

Decision:
- Agent: qa_agent
- Confidence: 95%
- Reasoning: "Question seeking information from documents"
```

**Example 3: Ambiguous**
```
Query: "Tell me about transformers"

LLM Reasoning:
- Could be question or blog request
- "Tell me" suggests information seeking
- No explicit content creation request

Decision:
- Agent: qa_agent
- Confidence: 70%
- Reasoning: "Informational request, though ambiguous"
```

### Fallback Logic

```python
def _fallback_route(self, user_query: str) -> Dict:
    """Keyword-based fallback routing"""
    query_lower = user_query.lower()
    
    # Blog keywords
    blog_keywords = ["write", "blog", "article", "post", "create", "generate"]
    
    if any(keyword in query_lower for keyword in blog_keywords):
        return {
            "agent": "blog_writer",
            "reasoning": "Fallback: Detected content creation keywords",
            "confidence": 0.6
        }
    else:
        return {
            "agent": "qa_agent",
            "reasoning": "Fallback: Default to Q&A agent",
            "confidence": 0.5
        }
```

### Performance

- **Speed**: ~1 second per routing decision
- **Accuracy**: ~95% correct routing (based on testing)
- **Reliability**: 100% (fallback ensures no failures)

---

## 🔍 Q&A Agent (ReAct)

### Overview

The Q&A Agent uses the **ReAct (Reasoning + Acting) pattern** to answer complex questions through multi-step autonomous reasoning.

### Key Characteristics

| Property | Value |
|----------|-------|
| **Type** | Reasoning agent |
| **Pattern** | ReAct (iterative) |
| **Files** | `research_agent.py`, `react_agent.py` |
| **LLM Calls** | 5-10 per query |
| **Tools** | 26 (24 paper + 2 web) |
| **Iterations** | 3-10 (configurable) |

### Architecture

The Q&A Agent uses a **wrapper pattern**:

```
ReActResearchAssistant (Wrapper)
    └─ Adds ReAct reasoning
        ↓
ResearchAssistant (Base)
    └─ Provides infrastructure
```

### Components

#### 1. ResearchAssistant (Base)

**Purpose**: Provides core infrastructure

**Responsibilities**:
- Load and manage papers
- Create tools (vector search, summary)
- Add web search tools
- Provide LLM instance

**Key Methods**:
```python
class ResearchAssistant:
    def setup(self):
        """Initialize papers and create tools"""
        
    async def query(self, question: str) -> str:
        """Simple query (no ReAct)"""
        
    async def write_blog_post(self, topic: str, ...) -> str:
        """Generate blog post"""
```

#### 2. ReActResearchAssistant (Wrapper)

**Purpose**: Add ReAct reasoning capabilities

**Responsibilities**:
- Extract tools from base assistant
- Create ReAct agent
- Manage reasoning iterations
- Format output with reasoning trace

**Key Methods**:
```python
class ReActResearchAssistant:
    def __init__(self, base_assistant, max_iterations=5, verbose=True):
        """Wrap base assistant with ReAct"""
        
    def setup(self):
        """Extract tools and create ReAct agent"""
        
    async def query(self, question: str) -> Dict:
        """Query with ReAct reasoning"""
```

### ReAct Pattern

**The ReAct Loop**:

```
for iteration in range(1, max_iterations + 1):
    
    # STEP 1: THINK
    thought = LLM.reason(
        question=question,
        history=previous_steps,
        available_tools=tools
    )
    
    # STEP 2: ACT
    action = LLM.select_action(thought)
    
    if action == "FINISH":
        break
    
    # STEP 3: EXECUTE
    observation = execute_tool(action, action_input)
    
    # STEP 4: RECORD
    history.append({
        "thought": thought,
        "action": action,
        "observation": observation
    })

# STEP 5: SYNTHESIZE
final_answer = LLM.synthesize(question, history)
```

### Reasoning Example

**Query**: "What are the main approaches in RAG systems?"

```
Iteration 1:
  💭 Thought: "I need to search for information about RAG approaches"
  🔧 Action: vector_search_AutonomousDataAgents
  📥 Input: "RAG approaches and methods"
  👁️ Observation: "RAG systems use retrieval-augmented generation..."

Iteration 2:
  💭 Thought: "I should get a comprehensive summary of RAG concepts"
  🔧 Action: summary_AutonomousDataAgents
  📥 Input: "RAG system approaches"
  👁️ Observation: "The paper discusses three main approaches..."

Iteration 3:
  💭 Thought: "Let me check if there are recent developments"
  🔧 Action: search_web
  📥 Input: "latest RAG system approaches 2024"
  👁️ Observation: "Recent advances include hybrid retrieval..."

Iteration 4:
  💭 Thought: "I have comprehensive information from papers and web"
  🔧 Action: FINISH
  📥 Input: N/A

Final Answer:
  "RAG systems employ three main approaches:
   1. Dense retrieval using embeddings...
   2. Hybrid methods combining dense and sparse...
   3. Advanced techniques like HyDE and self-RAG...
   
   Recent developments in 2024 include..."
```

### Tool Selection

The agent autonomously selects tools based on:

1. **Query requirements**: What information is needed?
2. **Previous observations**: What has been learned?
3. **Tool descriptions**: Which tool is most relevant?
4. **Reasoning history**: What has already been tried?

**Tool Selection Logic**:
```python
# LLM prompt includes:
prompt = f"""
Available tools:
- vector_search_paper1: Search paper1 content
- summary_paper1: Get summary of paper1
- vector_search_paper2: Search paper2 content
- summary_paper2: Get summary of paper2
- search_web: Search the web
- search_news: Search recent news

Based on your thought, which tool should you use?
"""
```

### Capabilities

✅ **Multi-Step Reasoning**
- Break complex queries into steps
- Iterative information gathering
- Self-directed research

✅ **Autonomous Tool Selection**
- Agent decides which tools to use
- No predefined workflow
- Adapts to query complexity

✅ **Information Synthesis**
- Combines information from multiple sources
- Cross-references papers
- Integrates web search results

✅ **Transparent Reasoning**
- Shows thought process
- Displays tool usage
- Explains decisions

✅ **Flexible Iteration**
- 3-10 reasoning steps (configurable)
- Stops when sufficient information gathered
- Efficient resource usage

### Performance

| Metric | Value |
|--------|-------|
| **Average iterations** | 4-6 |
| **Query time** | 10-30 seconds |
| **Tools used per query** | 3-5 |
| **Success rate** | ~90% |

### Configuration

```python
# In gradio_app
assistant = ReActResearchAssistant(
    base_assistant=base_assistant,
    max_iterations=5,      # Max reasoning steps
    verbose=True           # Show reasoning trace
)
```

---

## ✍️ Blog Writer Agent

### Overview

The Blog Writer Agent generates high-quality blog posts by intelligently selecting information sources and applying style preferences.

### Key Characteristics

| Property | Value |
|----------|-------|
| **Type** | Generation agent |
| **Pattern** | Direct generation |
| **File** | `smart_blog_writer.py` |
| **LLM Calls** | 3-5 per blog |
| **Tools** | Paper search, web search |
| **Iterations** | 1 (single generation) |

### Capabilities

✅ **Automatic Topic Extraction**
- Extracts topic from user query
- Handles various phrasings
- Cleans and normalizes topics

✅ **Intelligent Source Selection**
- Searches for relevant papers
- Falls back to web if needed
- Downloads from arXiv if enabled

✅ **Style Customization**
- Technical/Academic
- Professional/Business
- Casual/Conversational

✅ **Length Control**
- Short (300 words)
- Medium (500 words)
- Long (800 words)

✅ **Source Attribution**
- Lists papers used
- Provides metadata
- Transparent sourcing

### Workflow

```
1. Topic Extraction
   ↓
2. Paper Search
   ↓
3. Source Decision
   ├─ Papers found? → Use papers
   ├─ Download enabled? → Download from arXiv
   └─ Otherwise → Use web search
   ↓
4. Content Generation
   ├─ Apply style
   ├─ Apply length
   └─ Structure content
   ↓
5. Output
   └─ Blog post + metadata
```

### Topic Extraction

```python
async def _extract_topic(self, user_query: str) -> str:
    """Extract topic from user query using LLM"""
    
    prompt = f"""
    Extract the main topic from this request:
    "{user_query}"
    
    Return ONLY the topic (2-5 words), nothing else.
    
    Examples:
    - "Write a blog about computer vision" → "computer vision"
    - "Create an article on RAG systems" → "RAG systems"
    """
    
    response = await self.llm.acomplete(prompt)
    return response.text.strip()
```

### Paper Search

**Current Implementation** (Filename-based):
```python
def _search_papers(self, topic: str) -> List[str]:
    """Search for papers matching topic"""
    
    # Extract keywords
    keywords = topic.lower().split()
    
    # Search filenames
    relevant_papers = []
    for pdf_path in self.datasets_dir.glob("*.pdf"):
        filename = pdf_path.stem.lower()
        
        # Check if any keyword matches
        if any(kw in filename for kw in keywords):
            relevant_papers.append(str(pdf_path))
    
    return relevant_papers
```

**Limitation**: Only searches filenames, not content

**Potential Improvement**: Content-based search using embeddings

### Source Selection Logic

```python
async def write_smart_blog(self, user_query, download_enabled, ...):
    # 1. Extract topic
    topic = await self._extract_topic(user_query)
    
    # 2. Search for papers
    papers = self._search_papers(topic)
    
    # 3. Decide source
    if papers:
        # Use found papers
        return await self._write_from_papers(topic, papers, ...)
    
    elif download_enabled:
        # Download from arXiv
        papers = download_arxiv_papers(topic, ...)
        if papers:
            return await self._write_from_papers(topic, papers, ...)
    
    # Fallback: Web-based generation
    return await self._write_from_web(topic, ...)
```

### Generation Methods

#### 1. Paper-Based Generation

```python
async def _write_from_papers(self, topic, papers, style, word_count):
    # Create research assistant with papers
    assistant = ResearchAssistant(papers=papers)
    assistant.setup()
    
    # Generate blog using assistant
    blog_content = await assistant.write_blog_post(
        topic=topic,
        style=style,
        word_count=word_count,
        save_to_file=False
    )
    
    return blog_content, metadata
```

#### 2. Web-Based Generation

```python
async def _write_from_web(self, topic, style, word_count):
    # Search web for information
    web

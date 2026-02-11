# 🧠 Advanced Agentic RAG System

An intelligent multi-agent system for research paper analysis, question answering, and blog generation using advanced reasoning patterns.

## ✨ Features

### 🤖 **Multi-Agent Architecture**
- **Router Agent**: LLM-powered intent detection with explainable routing decisions
- **Q&A Agent**: Multi-step autonomous reasoning using ReAct pattern
- **Blog Writer Agent**: Intelligent content generation with automatic source selection

### 🧠 **Advanced Reasoning**
- **ReAct Pattern**: Reasoning + Acting for complex multi-step queries
- **Autonomous Tool Selection**: Agent decides which tools to use and when
- **Transparent Reasoning**: See the agent's thought process step-by-step
- **Iterative Refinement**: Up to 10 reasoning iterations for complex queries

### 🛡️ **Anti-Hallucination Protection**
- **Relevance Checking**: Vector tools verify query relevance before responding
- **Explicit Acknowledgment**: System states when information isn't found in papers
- **No False Information**: Tools return "No relevant information found" instead of making up answers
- **Source Verification**: Final answers only include information from actual tool outputs
- **Smart Fallback**: Automatically uses web search when papers don't contain relevant information

### 📚 **Flexible Paper Sources**
- Load existing PDFs from local directory
- Auto-download papers from arXiv
- Upload custom PDFs
- Web search fallback when papers unavailable

### 🔍 **Intelligent Search**
- Vector search across paper content
- Semantic similarity matching
- Web search integration
- Multi-source information synthesis

### ✍️ **Smart Content Generation**
- Automatic topic extraction
- Style customization (Technical/Professional/Casual)
- Length control (300-800 words)
- Source attribution

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Agentic
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up API key**
```bash
export OPENAI_API_KEY='your-api-key-here'
```

4. **Run the application**
```bash
python gradio_app_single_agent.py
```

5. **Open in browser**
```
http://127.0.0.1:7860
```

---

## 💻 Usage

### Question Answering

Ask questions about your research papers:

```
"What are the main approaches in RAG systems?"
"Compare autonomous agents across these papers"
"Explain the key concepts in this research"
```

The system will:
1. Route to Q&A Agent
2. Load relevant papers
3. Perform multi-step reasoning
4. Synthesize comprehensive answer

### Blog Generation

Request blog posts on topics:

```
"Write a blog on computer vision applications"
"Create a technical article about transformers"
"Generate a professional post on RAG systems"
```

The system will:
1. Route to Blog Writer Agent
2. Extract topic from request
3. Find relevant papers or use web search
4. Generate structured blog post

---

## 📊 System Architecture

```
User Query
    ↓
┌─────────────────────────────────────┐
│   Router Agent (LLM-Powered)       │
│   - Intent detection                │
│   - Confidence scoring              │
│   - Explainable routing             │
└─────────────┬───────────────────────┘
              │
    ┌─────────┴─────────┐
    │                   │
    ▼                   ▼
┌─────────┐      ┌──────────────┐
│ Q&A     │      │ BLOG WRITER  │
│ AGENT   │      │ AGENT        │
└─────────┘      └──────────────┘
    │                   │
    ▼                   ▼
ReAct Agent      SmartBlogWriter
(26 tools)       (Source selection)
```

For detailed architecture, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## 🎯 Key Components

### Router Agent
- **Purpose**: Intelligent query routing
- **Method**: LLM-powered intent detection
- **Output**: Routing decision with confidence score
- **Fallback**: Keyword-based classification

### Q&A Agent (ReAct)
- **Purpose**: Answer complex questions
- **Method**: Multi-step reasoning with tools
- **Tools**: 24 paper tools + 2 web search tools
- **Iterations**: 3-10 reasoning steps

### Blog Writer Agent
- **Purpose**: Generate blog posts
- **Method**: Topic extraction + source selection
- **Sources**: Papers or web search
- **Fallback**: Web-based generation

For detailed agent documentation, see [docs/AGENTS.md](docs/AGENTS.md)

---

## ⚙️ Configuration

### Environment Variables

```bash
# Required
export OPENAI_API_KEY='your-api-key-here'

# Optional
export OPENAI_MODEL='gpt-4'  # Default: gpt-4o-mini
```

### Paper Sources

Place PDF files in the `datasets/` directory:
```
datasets/
├── paper1.pdf
├── paper2.pdf
└── paper3.pdf
```

### UI Settings

- **Max Reasoning Steps**: 3-10 (default: 5)
- **Writing Style**: Technical/Professional/Casual
- **Output Length**: 300/500/800 words
- **Show Reasoning**: Enable/disable reasoning trace

---

## 📁 Project Structure

```
Agentic/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── gradio_app_single_agent.py    # Main application
├── gradio_app_multi_agent.py     # Alternative UI
│
├── router_agent.py               # Router Agent implementation
├── research_agent.py             # ResearchAssistant (base)
├── react_agent.py                # ReAct Agent wrapper
├── smart_blog_writer.py          # Blog Writer Agent
├── web_tools.py                  # Web search tools
├── arxiv_downloader.py           # arXiv integration
├── utils.py                      # Utility functions
│
├── datasets/                     # PDF storage
│   ├── paper1.pdf
│   └── paper2.pdf
│
├── blog_posts/                   # Generated blogs
│   └── blog_2025-11-27_topic.txt
│
└── docs/                         # Documentation
    ├── ARCHITECTURE.md           # System architecture
    ├── AGENTS.md                 # Agent details
    ├── FALLBACKS.md              # Fallback mechanisms
    └── EXAMPLES.md               # Usage examples
```

---

## 🔧 Advanced Features

### ReAct Reasoning

The Q&A agent uses the ReAct (Reasoning + Acting) pattern:

```
Iteration 1:
  💭 Thought: "I need to understand what RAG is"
  🔧 Action: Use vector_search tool
  👁️ Observation: [Results from paper]

Iteration 2:
  💭 Thought: "Now I need specific examples"
  🔧 Action: Use summary tool
  👁️ Observation: [Summary results]

Iteration 3:
  💭 Thought: "I have enough information"
  🔧 Action: FINISH
  ✅ Generate final answer
```

### Fallback Mechanisms

1. **Router Agent**: Falls back to keyword-based routing if LLM fails
2. **Blog Writer**: Falls back to web search if no papers found
3. **Error Handling**: Graceful degradation with informative messages

See [docs/FALLBACKS.md](docs/FALLBACKS.md) for details.

---

## 📚 Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Detailed system architecture
- **[AGENTS.md](docs/AGENTS.md)**: Agent implementations and capabilities
- **[FALLBACKS.md](docs/FALLBACKS.md)**: Fallback mechanisms and error handling
- **[EXAMPLES.md](docs/EXAMPLES.md)**: Usage examples and best practices

---

## 🎓 Examples

### Complex Q&A Query

```
Query: "Compare the autonomous agent approaches across papers 
        and explain their real-world applications"

Result:
- Router → Q&A Agent (95% confidence)
- ReAct performs 5 reasoning iterations
- Uses 4 different tools across 2 papers
- Synthesizes comprehensive comparison
```

### Blog Generation

```
Query: "Write a professional blog about RAG systems"

Result:
- Router → Blog Writer (95% confidence)
- Extracts topic: "RAG systems"
- Finds 3 relevant papers
- Generates 500-word professional blog
```

See [docs/EXAMPLES.md](docs/EXAMPLES.md) for more examples.

---

## 🛠️ Troubleshooting

### Common Issues

**Issue**: "OPENAI_API_KEY not set"
```bash
export OPENAI_API_KEY='your-key-here'
```

**Issue**: "No papers found"
- Enable "Load existing papers" checkbox
- Or enable arXiv download
- Or upload a PDF manually

**Issue**: "SSL Certificate Error"
- System automatically bypasses SSL verification
- Check internet connection

**Issue**: "Agent takes too long"
- Reduce max reasoning steps (3-5)
- Use fewer papers
- Check API rate limits

---

## 🔬 Technical Details

### Technologies Used

- **LLM**: OpenAI GPT-4o-mini
- **Framework**: LlamaIndex
- **UI**: Gradio
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Store**: In-memory (LlamaIndex)
- **Web Search**: DuckDuckGo

### Performance

- **Router Decision**: ~1 second
- **Q&A Query**: 10-30 seconds (depends on iterations)
- **Blog Generation**: 15-45 seconds (depends on sources)
- **Paper Loading**: ~1 second per paper

---

## 📈 Future Enhancements

- [ ] Add more specialized agents (Code Generator, Data Analyst)
- [ ] Implement parallel tool execution
- [ ] Add conversation memory
- [ ] Support more LLM providers
- [ ] Add paper summarization caching
- [ ] Implement agent collaboration
- [ ] Add evaluation metrics

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 🙏 Acknowledgments

- Built with [LlamaIndex](https://www.llamaindex.ai/)
- UI powered by [Gradio](https://gradio.app/)
- Inspired by the ReAct paper: [Yao et al., 2023](https://arxiv.org/abs/2210.03629)


---

## 🎯 Quick Reference

### Commands

```bash
# Run main app
python gradio_app_single_agent.py

# Run alternative UI
python gradio_app_multi_agent.py

# Download papers from arXiv
python arxiv_downloader.py
```

### Query Templates

**Q&A:**
- "What are [concept] in [topic]?"
- "Compare [approach A] and [approach B]"
- "Explain [concept] from the papers"

**Blog:**
- "Write a [style] blog about [topic]"
- "Create an article on [topic]"
- "Generate a post about [topic]"

---



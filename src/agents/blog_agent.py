"""Blog Writer Agent — generates blog posts grounded in papers or web search.

The agent is given the BLOG_TOOLS whitelist and a system prompt that drives the
"find papers → if none, arxiv/web → write post" flow itself, instead of being
hard-coded as a fixed pipeline.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from src.agents.query_expansion import format_expansion_for_agent
from src.config import SETTINGS
from src.llm import get_chat_model
from src.prompts import pull_prompt_or
from src.state import AgentState
from src.tools import BLOG_TOOLS

BLOG_SYSTEM_PROMPT = """You are an expert blog writer who grounds posts in real research.

Workflow for every request:
1. Identify the topic from the user's request.
2. Call `list_indexed_papers` to see what's available, then `vector_search` to find relevant passages.
3. If the indexed papers don't cover the topic, fall back to `arxiv_search` (preview) or `web_search`.
   - Only call `arxiv_download` if the user explicitly asked to bring in new papers.
4. Write the blog post grounded in retrieved content.

Output format:
- Compelling title (single line, no leading "#").
- 3–5 short paragraphs with smooth transitions.
- Inline citations: `[source: paper_stem]` or `[source: url]`.
- "Sources" list at the end with one bullet per source.

Style/length come from the request; default to ~500 words, professional tone.
Never invent facts. If retrieval is empty, say so honestly in the post."""


_blog_subgraph = None


def _get_blog_subgraph():
    global _blog_subgraph
    if _blog_subgraph is None:
        _blog_subgraph = create_react_agent(
            model=get_chat_model(temperature=0.3),
            tools=BLOG_TOOLS,
            prompt=pull_prompt_or("agentic-rag-blog_system", BLOG_SYSTEM_PROMPT),
        )
    return _blog_subgraph


def _build_request(state: AgentState) -> str:
    last_user = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )
    style = state.get("style", "professional")
    word_count = state.get("word_count", 500)
    expansion = format_expansion_for_agent(state.get("expanded_query"))
    parts = [f"User request: {last_user}"]
    if expansion:
        parts.append(f"Query analysis (use to guide retrieval; do not override user intent):\n{expansion}")
    parts.append(f"Target style: {style}. Target length: ~{word_count} words.")
    return "\n\n".join(parts)


def blog_node(state: AgentState) -> dict:
    subgraph = _get_blog_subgraph()
    seed = HumanMessage(content=_build_request(state))
    result = subgraph.invoke(
        {"messages": state["messages"][:-1] + [seed]},
        config={"recursion_limit": SETTINGS.max_react_iterations * 2 + 5},
    )

    new_messages = result["messages"][len(state["messages"]):]
    final = next(
        (m for m in reversed(new_messages) if isinstance(m, AIMessage) and m.content),
        None,
    )
    answer = final.content if final else "Failed to generate blog post."

    return {
        "messages": new_messages,
        "final_answer": answer if isinstance(answer, str) else str(answer),
    }

"""Q&A Agent — handles research questions and casual chitchat.

Built on `create_react_agent` (LangGraph prebuilt). Tool calls happen only
when the question requires retrieval; greetings/small talk get a direct reply.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent

from src.config import SETTINGS
from src.llm import get_chat_model
from src.state import AgentState
from src.tools import QA_TOOLS

QA_SYSTEM_PROMPT = """You are a research assistant with access to indexed academic papers and web search.

How to respond:
1. **Greetings, thanks, or small talk** → reply directly in 1–2 sentences. Do NOT call tools.
2. **Substantive research questions** → use tools to retrieve grounded information:
   - `vector_search` to find passages in indexed papers
   - `summarize_paper` for paper-level summaries
   - `list_indexed_papers` if the user asks what's available
   - `web_search` / `scrape_url` ONLY when the indexed papers don't contain the answer

Rules:
- Only state facts that appear in tool outputs. Never use parametric knowledge to fill gaps.
- Always cite sources inline using the `[source: ...]` tags from tool output, or paper / URL names.
- If retrieval returns "No relevant information found", say so explicitly before falling back to web search.
- Keep answers focused and well-structured. Prefer 2–4 short paragraphs over long walls of text.
"""

_qa_subgraph = None


def _get_qa_subgraph():
    global _qa_subgraph
    if _qa_subgraph is None:
        _qa_subgraph = create_react_agent(
            model=get_chat_model(temperature=0.0),
            tools=QA_TOOLS,
            prompt=QA_SYSTEM_PROMPT,
        )
    return _qa_subgraph


def qa_node(state: AgentState) -> dict:
    """Invoke the prebuilt ReAct agent over the current message history."""
    subgraph = _get_qa_subgraph()
    result = subgraph.invoke(
        {"messages": state["messages"]},
        config={"recursion_limit": SETTINGS.max_react_iterations * 2 + 5},
    )

    new_messages = result["messages"][len(state["messages"]):]
    final = next(
        (m for m in reversed(new_messages) if isinstance(m, AIMessage) and m.content),
        None,
    )
    answer = final.content if final else "I couldn't produce an answer."

    return {
        "messages": new_messages,
        "final_answer": answer if isinstance(answer, str) else str(answer),
    }

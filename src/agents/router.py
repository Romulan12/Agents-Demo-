"""Router / supervisor node — chooses which agent handles the query.

Uses structured output for a single, deterministic decision. The decision is
written to `state['route']`; conditional edges in `graph.py` dispatch from there.
"""
from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.llm import get_fast_chat_model
from src.state import AgentState

ROUTER_SYSTEM = """You are a router for a multi-agent research system. Decide which specialized agent should handle the user's message.

Agents:
- **qa**: Answers research questions ("what is X?", "compare A and B"), summarizes papers, AND handles greetings / chitchat / casual conversation. This is the default for anything that is not explicitly a blog or academic paper request.
- **blog**: Writes blog posts or articles ("write a blog about...", "draft an article on...").
- **academic**: Generates a critical-analysis academic paper following a thesis-driven structure ("write an academic paper", "critical analysis of...", "generate a scholarly paper").

Heuristics:
- "Write a summary" / "summarize this paper" → qa (NOT blog).
- "Write a blog" / "draft an article" → blog.
- "Critical analysis", "academic paper", "scholarly", "thesis on..." → academic.
- Greetings, "what can you do?", thanks → qa."""


class RouteDecision(BaseModel):
    agent: Literal["qa", "blog", "academic"] = Field(
        description="Which agent should handle this query."
    )
    reason: str = Field(description="One-sentence justification.")


def router_node(state: AgentState) -> dict:
    user_msg = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )

    llm = get_fast_chat_model(temperature=0.0).with_structured_output(RouteDecision)
    decision: RouteDecision = llm.invoke(
        [("system", ROUTER_SYSTEM), ("user", str(user_msg))]
    )

    return {"route": decision.agent, "route_reason": decision.reason}

"""Graph state shared across all nodes."""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

Route = Literal["qa", "blog", "academic", "blocked"]


class AgentState(TypedDict, total=False):
    """State passed between nodes. `messages` is reducer-merged; everything else
    is replaced on write."""

    messages: Annotated[list[BaseMessage], add_messages]

    # Query expansion (set by query_expansion_node, before router)
    expanded_query: dict

    # Routing
    route: Route
    route_reason: str

    # Guardrail outputs
    input_guard_passed: bool
    input_guard_reason: str
    output_guard_passed: bool
    output_guard_reason: str

    # Per-agent context
    topic: str
    style: str
    word_count: int

    # Academic agent (human-in-loop)
    outline: dict
    outline_approved: bool

    # Final answer surfaced to user
    final_answer: str
    citations: list[str]

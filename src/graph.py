"""Main LangGraph wiring.

Flow:
  START → input_guard → (blocked? → END)
       → router
       → qa                                     (fast path, no expansion)
       → blog | academic  (via query_expansion) (expansion-enriched)
       → output_guard → END

Persistence: SqliteSaver (cross-session memory keyed by `thread_id`).
Tracing: LangSmith (auto-enabled if `LANGSMITH_TRACING=true`).
"""
from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.agents.academic_agent import build_academic_subgraph
from src.agents.blog_agent import blog_node
from src.agents.qa_agent import qa_node
from src.agents.query_expansion import query_expansion_node
from src.agents.router import router_node
from src.config import SETTINGS
from src.guardrails.input_guard import input_guard_node
from src.guardrails.output_guard import output_guard_node
from src.state import AgentState


def _route_from_input_guard(state: AgentState) -> Literal["router", "end"]:
    return "router" if state.get("input_guard_passed") else "end"


def _route_from_router(state: AgentState) -> Literal["qa", "expand"]:
    """QA skips expansion (chitchat/simple Q&A doesn't benefit). Blog and
    academic both go through expansion first."""
    return "qa" if state.get("route", "qa") == "qa" else "expand"


def _route_from_expansion(state: AgentState) -> Literal["blog", "academic"]:
    return state.get("route", "blog")  # type: ignore[return-value]


def _build_graph_definition() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("input_guard", input_guard_node)
    g.add_node("router", router_node)
    g.add_node("query_expansion", query_expansion_node)
    g.add_node("qa", qa_node)
    g.add_node("blog", blog_node)
    g.add_node("academic", build_academic_subgraph())
    g.add_node("output_guard", output_guard_node)

    g.add_edge(START, "input_guard")
    g.add_conditional_edges(
        "input_guard",
        _route_from_input_guard,
        {"router": "router", "end": END},
    )
    g.add_conditional_edges(
        "router",
        _route_from_router,
        {"qa": "qa", "expand": "query_expansion"},
    )
    g.add_conditional_edges(
        "query_expansion",
        _route_from_expansion,
        {"blog": "blog", "academic": "academic"},
    )
    g.add_edge("qa", "output_guard")
    g.add_edge("blog", "output_guard")
    g.add_edge("academic", "output_guard")
    g.add_edge("output_guard", END)

    return g


_compiled_graph = None
_checkpointer_cm = None


def get_graph():
    """Compile and cache the graph for the lifetime of the process.

    The SqliteSaver context manager is entered once and intentionally never
    exited — the connection lives as long as the host process (Gradio app).
    """
    global _compiled_graph, _checkpointer_cm
    if _compiled_graph is None:
        _checkpointer_cm = SqliteSaver.from_conn_string(str(SETTINGS.checkpoint_db))
        checkpointer = _checkpointer_cm.__enter__()
        builder = _build_graph_definition()
        _compiled_graph = builder.compile(checkpointer=checkpointer)
    return _compiled_graph

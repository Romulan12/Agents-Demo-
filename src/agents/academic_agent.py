"""Academic Writer Agent — two-phase outline-then-paper subgraph.

Flow:
  research_node → outline_node → [interrupt: user approves outline]
    → if approved: paper_node → done
    → if rejected: outline_node again with feedback

The interrupt pauses the graph; the host (Gradio app) resumes via
`graph.invoke(Command(resume={"approved": True}), config={...})`.
"""
from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.agents.query_expansion import format_expansion_for_agent
from src.config import SETTINGS
from src.llm import get_chat_model
from src.prompts import pull_prompt_or
from src.state import AgentState
from src.tools import ACADEMIC_TOOLS

RESEARCH_PROMPT = """You are gathering research for a critical analysis paper on the given trend.

Use the available tools (vector_search, summarize_paper, arxiv_search) to collect:
- Key technical approaches in the field
- Current limitations and open problems
- Recent developments

Return a 300-400 word research brief with inline `[source: ...]` citations.
Do not write the paper itself yet."""


class Outline(BaseModel):
    thesis: str = Field(description="A specific, debatable thesis statement.")
    introduction_stakes: list[str] = Field(description="3-5 bullets establishing why the trend matters.")
    technical_landscape: list[str] = Field(description="3-5 bullets describing approaches and state of the art.")
    barriers_technical: list[str]
    barriers_empirical: list[str]
    barriers_theoretical: list[str]
    barriers_practical: list[str]
    original_contribution: list[str] = Field(description="3-5 bullets of unique perspective / framing.")
    conclusion: list[str] = Field(description="2-4 bullets summarizing implications.")


def _get_research_agent():
    return create_react_agent(
        model=get_chat_model(temperature=0.0),
        tools=ACADEMIC_TOOLS,
        prompt=pull_prompt_or("agentic-rag-academic_research", RESEARCH_PROMPT),
    )


def _last_user(state: AgentState) -> str:
    return next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )


def _research_node(state: AgentState) -> dict:
    trend = state.get("topic") or _last_user(state)
    agent = _get_research_agent()
    expansion = format_expansion_for_agent(state.get("expanded_query"))
    seed_text = f"Trend to analyze: {trend}"
    if expansion:
        seed_text += f"\n\nQuery analysis (use to guide research; do not change the trend):\n{expansion}"
    result = agent.invoke(
        {"messages": [HumanMessage(content=seed_text)]},
        config={"recursion_limit": SETTINGS.max_react_iterations * 2 + 5},
    )
    final = next(
        (m for m in reversed(result["messages"]) if isinstance(m, AIMessage) and m.content),
        None,
    )
    research_text = final.content if final else ""
    return {
        "topic": trend,
        "messages": [AIMessage(content=f"**Research brief:**\n\n{research_text}")],
        "outline": {"_research_brief": research_text},
    }


def _outline_node(state: AgentState) -> dict:
    trend = state["topic"]
    research_brief = state.get("outline", {}).get("_research_brief", "")
    feedback = state.get("outline", {}).get("_feedback", "")

    llm = get_chat_model(temperature=0.4).with_structured_output(Outline)
    user_msg = (
        f"Trend: {trend}\n\n"
        f"Research brief:\n{research_brief}\n\n"
        + (f"Reviewer feedback to address:\n{feedback}\n\n" if feedback else "")
        + "Generate a structured outline for a critical analysis paper. "
          "The thesis must be specific and debatable. Each barrier category needs 2-4 concrete points."
    )

    outline: Outline = llm.invoke(
        [
            ("system", "You are an expert academic writer specializing in critical analysis."),
            ("user", user_msg),
        ]
    )

    outline_dict = outline.model_dump()
    outline_dict["_research_brief"] = research_brief

    summary = _format_outline(trend, outline_dict)
    return {
        "outline": outline_dict,
        "messages": [AIMessage(content=summary)],
    }


def _format_outline(trend: str, outline: dict) -> str:
    lines = [
        f"# Outline: Critical Analysis of {trend}",
        "",
        f"**Thesis:** {outline['thesis']}",
        "",
        "## 1. Introduction & Stakes",
        *(f"- {p}" for p in outline["introduction_stakes"]),
        "",
        "## 2. Technical Landscape",
        *(f"- {p}" for p in outline["technical_landscape"]),
        "",
        "## 3. Barriers to Progress",
        "**Technical:**",
        *(f"  - {p}" for p in outline["barriers_technical"]),
        "**Empirical:**",
        *(f"  - {p}" for p in outline["barriers_empirical"]),
        "**Theoretical:**",
        *(f"  - {p}" for p in outline["barriers_theoretical"]),
        "**Practical / Institutional:**",
        *(f"  - {p}" for p in outline["barriers_practical"]),
        "",
        "## 4. Original Contribution",
        *(f"- {p}" for p in outline["original_contribution"]),
        "",
        "## 5. Conclusion",
        *(f"- {p}" for p in outline["conclusion"]),
    ]
    return "\n".join(lines)


def _approval_node(state: AgentState) -> dict:
    """Pause the graph and wait for the user to approve, reject, or revise."""
    decision = interrupt(
        {
            "kind": "outline_approval",
            "topic": state["topic"],
            "outline": state["outline"],
            "instructions": "Resume with {'approved': True} to write the paper, "
            "or {'approved': False, 'feedback': '...'} to revise the outline.",
        }
    )

    if isinstance(decision, dict) and decision.get("approved"):
        return {"outline_approved": True}

    feedback = (
        decision.get("feedback", "") if isinstance(decision, dict) else str(decision)
    )
    updated = dict(state["outline"])
    updated["_feedback"] = feedback
    return {"outline_approved": False, "outline": updated}


def _route_after_approval(state: AgentState) -> Literal["paper", "outline"]:
    return "paper" if state.get("outline_approved") else "outline"


def _paper_node(state: AgentState) -> dict:
    trend = state["topic"]
    outline = state["outline"]
    research_brief = outline.get("_research_brief", "")

    llm = get_chat_model(temperature=0.5)
    sections = []
    section_specs = [
        ("Introduction", outline["introduction_stakes"], 350),
        ("Technical Landscape", outline["technical_landscape"], 450),
        ("Technical Barriers", outline["barriers_technical"], 200),
        ("Empirical Barriers", outline["barriers_empirical"], 200),
        ("Theoretical Barriers", outline["barriers_theoretical"], 200),
        ("Practical / Institutional Barriers", outline["barriers_practical"], 200),
        ("Original Contribution", outline["original_contribution"], 350),
        ("Conclusion", outline["conclusion"], 250),
    ]

    for name, points, target_words in section_specs:
        bullet_text = "\n".join(f"- {p}" for p in points)
        prompt = (
            f"Write the **{name}** section of a critical analysis paper on '{trend}'.\n\n"
            f"Thesis: {outline['thesis']}\n\n"
            f"Points to cover:\n{bullet_text}\n\n"
            f"Research context:\n{research_brief[:1500]}\n\n"
            f"Requirements: formal academic tone, ~{target_words} words, complete paragraphs (no bullets), "
            f"cite sources from the research context inline as [source: ...]."
        )
        response = llm.invoke(
            [
                ("system", "You are an expert academic writer."),
                ("user", prompt),
            ]
        )
        sections.append(f"## {name}\n\n{response.content}")

    title = f"# Critical Analysis of {trend}"
    paper = "\n\n".join([title, *sections])
    word_count = len(paper.split())
    paper += f"\n\n---\n*Word count: {word_count}*"

    return {
        "messages": [AIMessage(content=paper)],
        "final_answer": paper,
    }


def build_academic_subgraph():
    """Compile the academic agent subgraph (with interrupt). Returned compiled
    graph is invoked from `academic_node` in the main graph."""
    g = StateGraph(AgentState)
    g.add_node("research", _research_node)
    g.add_node("outline", _outline_node)
    g.add_node("approval", _approval_node)
    g.add_node("paper", _paper_node)

    g.add_edge(START, "research")
    g.add_edge("research", "outline")
    g.add_edge("outline", "approval")
    g.add_conditional_edges(
        "approval", _route_after_approval, {"paper": "paper", "outline": "outline"}
    )
    g.add_edge("paper", END)

    return g.compile()



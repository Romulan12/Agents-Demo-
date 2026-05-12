"""Query expansion node — enriches a brief user query before routing.

Runs after input_guard, before router. Output is written to
`state["expanded_query"]` as a dict; downstream agents read it to inform
retrieval, topic extraction, and scope.

The original user message is preserved verbatim — expansion is additive
context, never a substitution.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.llm import get_fast_chat_model
from src.prompts import pull_prompt_or
from src.state import AgentState

QUERY_EXPANSION_SYSTEM = """You are a query rewriter for a multi-agent research \
assistant. Your job is to take a brief or vague user query and enrich it with \
context, terminology, and intent, **without changing what the user actually \
asked for**.

Hard rules:
- The `original_query` field MUST be the user's input verbatim. Do not paraphrase.
- `interpreted_intent` is your single best guess at what the user wants. Be \
  specific but humble — if the request is ambiguous, say so.
- `key_concepts` are the named entities, technical terms, or topics already \
  present (or strongly implied). 3–7 items.
- `search_terms` are query rewrites usable for vector search and arXiv. Include \
  synonyms, acronym expansions, related sub-fields. 4–10 items, no duplicates of \
  `key_concepts`.
- `scope_constraints` lists topics the answer should NOT drift into (e.g. if \
  asked about "transformer architectures", note "not electrical transformers"). \
  Empty list is fine if no risk of drift.

Never invent constraints the user didn't express. Never expand scope beyond a \
reasonable reading of the request. When in doubt, prefer fewer terms over more."""


class ExpandedQuery(BaseModel):
    original_query: str = Field(description="The user's input verbatim.")
    interpreted_intent: str = Field(description="One sentence describing what the user likely wants.")
    key_concepts: list[str] = Field(description="3-7 named entities or technical terms in the query.")
    search_terms: list[str] = Field(description="4-10 query rewrites for retrieval (synonyms, acronyms, related sub-fields).")
    scope_constraints: list[str] = Field(default_factory=list, description="Topics the answer should NOT drift into. Empty if no risk.")


def _last_user_text(state: AgentState) -> str:
    return next(
        (m.content for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)),
        "",
    )


def query_expansion_node(state: AgentState) -> dict:
    user_text = _last_user_text(state)
    if not user_text:
        return {"expanded_query": {}}

    llm = get_fast_chat_model(temperature=0.1).with_structured_output(ExpandedQuery)
    expanded: ExpandedQuery = llm.invoke(
        [
            ("system", pull_prompt_or("agentic-rag-query_expansion", QUERY_EXPANSION_SYSTEM)),
            ("user", user_text),
        ]
    )
    return {"expanded_query": expanded.model_dump()}


def format_expansion_for_agent(expanded: dict | None) -> str:
    """Render the expansion as a compact context block agents can paste into prompts."""
    if not expanded:
        return ""
    parts = [
        f"Interpreted intent: {expanded.get('interpreted_intent', '')}",
    ]
    if expanded.get("key_concepts"):
        parts.append("Key concepts: " + ", ".join(expanded["key_concepts"]))
    if expanded.get("search_terms"):
        parts.append("Suggested search terms: " + ", ".join(expanded["search_terms"]))
    if expanded.get("scope_constraints"):
        parts.append("Out of scope: " + "; ".join(expanded["scope_constraints"]))
    return "\n".join(parts)

"""Output guardrail node — runs after the agent, before responding.

Two checks:
  1. Grounding — is the final answer supported by retrieved tool output?
  2. Citation — for QA-routed answers, does the answer cite a source?

If a check fails, we annotate the answer rather than wholesale rewriting it,
so the user still sees something useful and the failure is visible in the trace.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field

from src.llm import get_fast_chat_model
from src.state import AgentState

SYSTEM = """You are an output safety reviewer for a research-paper RAG system.

You will receive:
- The final answer the assistant intends to send.
- The tool-call observations gathered during reasoning (may be empty for chitchat).

Decide:
1. grounded: Is the answer supported by the observations? For chitchat / greetings
   with no tool calls, return grounded=true (no grounding required).
2. cited: Does the answer reference at least one source from the observations
   (paper name, URL, or [source: ...] tag)? For chitchat, return cited=true.

Be lenient on chitchat. Be strict on factual claims about papers."""


class OutputVerdict(BaseModel):
    grounded: bool = Field(description="Answer is supported by observations.")
    cited: bool = Field(description="Answer cites at least one source when factual.")
    reason: str = Field(description="One-sentence justification.")


def _collect_observations(state: AgentState) -> str:
    parts = []
    for msg in state.get("messages", []):
        if isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            parts.append(f"[{msg.name}] {content[:1500]}")
    return "\n\n".join(parts) if parts else "(no tool calls were made)"


def _final_answer_text(state: AgentState) -> str:
    if state.get("final_answer"):
        return state["final_answer"]
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def output_guard_node(state: AgentState) -> dict:
    answer = _final_answer_text(state)
    if not answer.strip():
        return {
            "output_guard_passed": False,
            "output_guard_reason": "No final answer produced.",
            "final_answer": "I wasn't able to produce an answer. Please try rephrasing.",
        }

    observations = _collect_observations(state)

    llm = get_fast_chat_model(temperature=0.0).with_structured_output(OutputVerdict)
    verdict: OutputVerdict = llm.invoke(
        [
            ("system", SYSTEM),
            (
                "user",
                f"FINAL ANSWER:\n{answer}\n\nTOOL OBSERVATIONS:\n{observations}",
            ),
        ]
    )

    passed = verdict.grounded and verdict.cited
    update: dict = {
        "output_guard_passed": passed,
        "output_guard_reason": verdict.reason,
        "final_answer": answer,
    }

    if not passed:
        warnings = []
        if not verdict.grounded:
            warnings.append("⚠️ Some claims could not be verified against retrieved sources.")
        if not verdict.cited:
            warnings.append("⚠️ Answer is missing source citations.")
        annotated = f"{answer}\n\n---\n" + "\n".join(warnings) + f"\n_{verdict.reason}_"
        update["final_answer"] = annotated
        update["messages"] = [AIMessage(content=annotated)]

    return update

"""LangSmith Hub pull helper.

Each agent owns its own prompt as a module-level constant — the source of truth
lives next to the agent, not here. This module only adds the *option* to override
the local constant from LangSmith Hub at runtime, by name.

Usage from an agent:
    LOCAL_PROMPT = "...my system prompt..."
    prompt = pull_prompt_or("agentic-rag-blog_system", LOCAL_PROMPT)

Behavior:
- If LANGSMITH_PROMPTS=true and LANGSMITH_API_KEY is set, try the hub.
- On any failure (hub down, prompt missing, parse error), return the local fallback.
- Result is cached per (name, revision) for the process lifetime.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate

log = logging.getLogger(__name__)


def _revision() -> str:
    return os.getenv("PROMPT_REVISION", "latest")


def _hub_enabled() -> bool:
    return (
        bool(os.getenv("LANGSMITH_API_KEY"))
        and os.getenv("LANGSMITH_PROMPTS", "true").lower() == "true"
    )


def pull_prompt_or(name: str, fallback: str) -> str:
    """Pull `name` from LangSmith Hub if enabled; else return `fallback`.

    `name` is the bare prompt handle in the API key's workspace
    (e.g. "agentic-rag-blog_system"). LangSmith resolves it to your tenant.
    """
    return _cached_pull(name, _revision()) or fallback


@lru_cache(maxsize=None)
def _cached_pull(name: str, revision: str) -> str:
    if not _hub_enabled():
        return ""
    try:
        from langsmith import Client

        client = Client()
        ref = name if revision in ("", "latest") else f"{name}:{revision}"
        obj = client.pull_prompt(ref)
        text = _extract_system_text(obj)
        if text:
            log.info("Loaded prompt %r from LangSmith Hub (%s)", name, ref)
        return text
    except Exception as e:  # noqa: BLE001 — degrade gracefully on any hub error
        log.warning("Hub pull for %r failed (%s); using fallback.", name, e)
        return ""


def _extract_system_text(obj) -> str:
    """Best-effort extraction of system-prompt text from whatever the hub returns."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, ChatPromptTemplate):
        for msg in obj.messages:
            template = getattr(msg, "prompt", None)
            template_text = getattr(template, "template", None) if template else None
            role = getattr(msg, "role", "") or type(msg).__name__.lower()
            if template_text and "system" in role:
                return template_text
        if obj.messages:
            tmpl = getattr(obj.messages[0], "prompt", None)
            return getattr(tmpl, "template", "") if tmpl else ""
    return getattr(obj, "template", "") or ""

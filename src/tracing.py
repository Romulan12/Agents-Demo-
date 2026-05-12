"""Tracing setup using LangSmith.

LangChain auto-emits traces to LangSmith when `LANGSMITH_TRACING=true` and
`LANGSMITH_API_KEY` are set in the environment. There's nothing to instrument
manually — this module just validates the config and logs status.

Disable by setting LANGSMITH_TRACING=false in .env.
"""
from __future__ import annotations

import logging
import os

from src.config import SETTINGS

logger = logging.getLogger(__name__)

_initialized = False


def init_tracing() -> None:
    """Idempotent — safe to call multiple times."""
    global _initialized
    if _initialized:
        return

    if not SETTINGS.langsmith_tracing:
        logger.info("LangSmith tracing disabled (LANGSMITH_TRACING=false)")
        _initialized = True
        return

    if not SETTINGS.langsmith_api_key:
        logger.warning(
            "LANGSMITH_TRACING=true but LANGSMITH_API_KEY is unset — traces will not be sent."
        )
        _initialized = True
        return

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = SETTINGS.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = SETTINGS.langsmith_project
    if SETTINGS.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = SETTINGS.langsmith_endpoint

    _initialized = True
    logger.info(
        "LangSmith tracing initialized (project=%s)", SETTINGS.langsmith_project
    )

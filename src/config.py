"""Centralized configuration. Loads .env once and exposes typed settings."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_base_url: str
    llm_model: str
    fast_llm_model: str
    embedding_model: str
    datasets_dir: Path
    chroma_dir: Path
    checkpoint_db: Path
    max_react_iterations: int
    langsmith_tracing: bool
    langsmith_api_key: str
    langsmith_project: str
    langsmith_endpoint: str


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(f"Missing required env var: {key}")
    return val


def load_settings() -> Settings:
    settings = Settings(
        openai_api_key=_require("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", ""),
        llm_model=_require("LLM_MODEL"),
        fast_llm_model=os.getenv("FAST_LLM_MODEL", ""),
        embedding_model=_require("EMBEDDING_MODEL"),
        datasets_dir=Path(os.getenv("DATASETS_DIR", "./datasets")).resolve(),
        chroma_dir=Path(os.getenv("CHROMA_DIR", "./data/chroma")).resolve(),
        checkpoint_db=Path(os.getenv("CHECKPOINT_DB", "./data/checkpoints.sqlite")).resolve(),
        max_react_iterations=int(os.getenv("MAX_REACT_ITERATIONS", "4")),
        langsmith_tracing=os.getenv("LANGSMITH_TRACING", "true").lower() == "true",
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY", ""),
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "agentic-rag"),
        langsmith_endpoint=os.getenv("LANGSMITH_ENDPOINT", ""),
    )

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    settings.datasets_dir.mkdir(parents=True, exist_ok=True)

    return settings


SETTINGS = load_settings()

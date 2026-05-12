"""Backward-compatible re-exports. The real factory lives in src.llm_client."""
from src.llm_client import get_chat_model, get_embeddings, get_fast_chat_model

__all__ = ["get_chat_model", "get_embeddings", "get_fast_chat_model"]

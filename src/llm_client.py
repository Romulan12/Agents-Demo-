"""Provider-agnostic LLM client factory.

Reads config from .env via SETTINGS but everything is overridable per-call.
Works with any OpenAI-compatible endpoint: real OpenAI, LiteLLM gateways,
Azure proxies, local vLLM, etc. — set OPENAI_BASE_URL.

LLM_MODEL format: "<provider>:<model>" e.g. "openai:claude-sonnet-4-6".
The provider prefix only routes which SDK we use; the model name is whatever
the configured endpoint recognizes.

Run as a script for a one-shot smoke test:
    python -m src.llm_client
"""
from __future__ import annotations

import os
import ssl
from dataclasses import dataclass

import certifi

# Set BEFORE any httpx import so every httpx.Client in the process — ours or
# the ones langchain/openai create internally — uses certifi's CA bundle.
# Without this, SDK-created clients inherit Python's default trust store,
# which on macOS+anaconda doesn't validate corporate gateway chains.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

import httpx  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: E402

from src.config import SETTINGS  # noqa: E402


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    api_key: str
    base_url: str

    @classmethod
    def from_settings(cls) -> "ProviderConfig":
        provider, sep, model = SETTINGS.llm_model.partition(":")
        if not sep or not model:
            raise RuntimeError(
                f"LLM_MODEL must be 'provider:model' (e.g. 'openai:claude-sonnet-4-6'), "
                f"got: {SETTINGS.llm_model!r}"
            )
        return cls(
            provider=provider,
            model=model,
            api_key=SETTINGS.openai_api_key,
            base_url=SETTINGS.openai_base_url,
        )


def _ssl_clients() -> tuple[httpx.Client, httpx.AsyncClient]:
    ctx = ssl.create_default_context(cafile=certifi.where())
    return httpx.Client(verify=ctx), httpx.AsyncClient(verify=ctx)


def _build_chat(model_id: str, temperature: float, overrides: dict) -> BaseChatModel:
    provider, sep, model = model_id.partition(":")
    if not sep or not model:
        raise RuntimeError(f"Model id must be 'provider:model', got {model_id!r}")
    if provider != "openai":
        raise NotImplementedError(
            f"Provider {provider!r} not wired. Only OpenAI-compatible endpoints supported."
        )
    cfg = ProviderConfig.from_settings()
    sync_client, async_client = _ssl_clients()
    kwargs: dict = dict(
        model=model,
        api_key=cfg.api_key,
        temperature=temperature,
        http_client=sync_client,
        http_async_client=async_client,
    )
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    kwargs.update(overrides)
    return ChatOpenAI(**kwargs)


def get_chat_model(temperature: float = 0.0, **overrides) -> BaseChatModel:
    """Main chat model — used for the heavy agent work (qa, blog, academic paper writing)."""
    return _build_chat(SETTINGS.llm_model, temperature, overrides)


def get_fast_chat_model(temperature: float = 0.0, **overrides) -> BaseChatModel:
    """Fast/cheap chat model — used for lightweight nodes (guards, router, query
    expansion). Falls back to the main model if FAST_LLM_MODEL is unset."""
    return _build_chat(SETTINGS.fast_llm_model or SETTINGS.llm_model, temperature, overrides)


def get_embeddings() -> OpenAIEmbeddings:
    cfg = ProviderConfig.from_settings()
    sync_client, async_client = _ssl_clients()
    kwargs: dict = dict(
        model=SETTINGS.embedding_model,
        api_key=cfg.api_key,
        http_client=sync_client,
        http_async_client=async_client,
        # Skip tiktoken pre-chunking. OpenAI accepts token-id arrays, but
        # non-OpenAI backends behind LiteLLM (Titan/Bedrock, Cohere, Gemini)
        # only accept raw strings — they 400 on token arrays.
        check_embedding_ctx_length=False,
    )
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    return OpenAIEmbeddings(**kwargs)


def describe() -> dict:
    cfg = ProviderConfig.from_settings()
    return {
        "provider": cfg.provider,
        "model": cfg.model,
        "base_url": cfg.base_url or "(SDK default — api.openai.com)",
        "api_key_prefix": (cfg.api_key[:6] + "...") if cfg.api_key else "(missing)",
        "embedding_model": SETTINGS.embedding_model,
    }


if __name__ == "__main__":
    print("Loaded provider config:")
    for k, v in describe().items():
        print(f"  {k}: {v}")
    print("\nInvoking chat model...")
    print(get_chat_model().invoke("Say hello in one sentence.").content)

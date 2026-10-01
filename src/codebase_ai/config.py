"""Settings (pydantic-settings): provider selection, model ids, keys, retrieval and budget limits."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration, read from environment variables and an optional `.env` file.

    Variable names are the upper-case field names (``LLM_PROVIDER`` -> ``llm_provider``).
    Empty values (``ANTHROPIC_API_KEY=``) are ignored so defaults apply.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    # --- LLM provider ---
    llm_provider: Literal["anthropic", "openai", "ollama"] = "anthropic"
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-opus-5-5"
    # low | medium | high | xhigh | max. None = the model's own default. Only sent when set (older models reject it).
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    # Ask the API to re-run a request on another model when a safety classifier wrongly declines it (Claude API only).
    anthropic_refusal_fallback: bool = True
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:7b"
    # Ollama's default context window is far smaller than a retrieved prompt and would silently truncate it.
    ollama_num_ctx: int = Field(default=16_384, gt=0)

    # --- Answer generation ---
    answer_max_tokens: int = Field(default=16_000, gt=0)  # includes the model's thinking, so keep it generous
    answer_temperature: float | None = Field(default=None, ge=0, le=2)  # None = provider default (Claude rejects it)
    llm_timeout_seconds: float = Field(default=120.0, gt=0)

    # --- Embedding provider ---
    embedding_provider: Literal["local", "openai", "ollama"] = "local"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_embedding_model: str = "text-embedding-3-small"
    ollama_embedding_model: str = "nomic-embed-text"

    # --- Ingestion / retrieval limits ---
    max_file_bytes: int = Field(default=1_000_000, gt=0)
    max_config_bytes: int = Field(default=200_000, gt=0)
    chunk_max_lines: int = Field(default=120, gt=0)
    window_lines: int = Field(default=60, gt=0)
    window_overlap: int = Field(default=10, ge=0)
    embed_batch_size: int = Field(default=64, gt=0)
    context_token_budget: int = Field(default=12_000, gt=0)
    retrieve_top_k: int = Field(default=30, gt=0)
    # Chosen from the M2 evaluation (docs/RETRIEVAL_EVAL.md): vector search matched or beat hybrid fusion, and
    # demoting test files was a significant gain that never cost a question.
    retrieval_mode: Literal["vector", "keyword", "hybrid"] = "vector"
    keyword_weight: float = Field(default=1.0, ge=0)  # weight of keyword results vs vector results in hybrid fusion
    test_penalty: float = Field(default=0.5, gt=0, le=1)  # score multiplier for test files; 1 turns demotion off

    # --- Reranking (M4) ---
    # A cross-encoder model name (e.g. from Hugging Face) to re-order the top candidates; empty = off. Off by default:
    # it must earn its latency in the evaluation first (docs/QUALITY_EVAL.md).
    reranker_model: str | None = None
    rerank_top_n: int = Field(default=30, gt=0)  # how many of the best candidates it re-orders

    # --- Question handling (M4) ---
    use_repo_map: bool = True  # questions about the whole repository also get a map of it (see retrieval/repo_map.py)
    repo_map_tokens: int = Field(default=1500, gt=0)
    condense_followups: bool = True  # one short model call turns a follow-up into a standalone question
    history_turns: int = Field(default=3, ge=0)  # earlier exchanges used for that (0 turns it off)

    # --- Storage ---
    index_dir: Path = Field(default_factory=lambda: Path.home() / ".codebase_ai")

    def embedding_model_id(self) -> str:
        """Identifier stored in the index manifest; vectors from different ids are not comparable."""
        model = {
            "local": self.local_embedding_model,
            "openai": self.openai_embedding_model,
            "ollama": self.ollama_embedding_model,
        }[self.embedding_provider]
        return f"{self.embedding_provider}:{model}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, loaded once. Tests can call ``get_settings.cache_clear()``."""
    return Settings()

"""
Configuration module for the Agentic AI RAG system.

Loads environment variables (e.g. GROQ_API_KEY) and centralises all
tunable settings so the rest of the codebase depends on a single,
typed settings object. Uses pydantic-settings for validation.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]
NEWS_DIR = BASE_DIR / "news"
CHROMA_DIR = BASE_DIR / "chroma_db"
SQLITE_PATH = BASE_DIR / "newpaper_search.db"


class Settings(BaseSettings):
    """Application settings loaded from .env / environment."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ------------------------------------------------------------------
    # Groq (LLM provider)
    # ------------------------------------------------------------------
    groq_api_key: str = Field(..., description="Groq API key (required for all LLM calls)")
    groq_base_url: str = Field(
        "https://api.groq.com",
        description="Groq OpenAI-compatible base URL (host only, no version path)",
    )
    # Best available Groq model for complex agentic reasoning (supervisor / synthesis).
    # NOTE: model availability depends on your Groq account/key. This key provides
    # gpt-oss-120b / gpt-oss-20b. Adjust via GROQ_MODEL_REASONING env var if needed.
    groq_model_reasoning: str = Field(
        "openai/gpt-oss-120b",
        description="Groq model used for reasoning-heavy agents",
    )
    # Fast Groq model for simple classification / guardrail tasks
    groq_model_fast: str = Field(
        "openai/gpt-oss-20b",
        description="Groq model used for fast, low-latency tasks",
    )
    groq_temperature: float = Field(0.2, description="Default sampling temperature")
    groq_max_tokens: int = Field(2048, description="Default max output tokens")

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    # NOTE: Groq does NOT provide an embeddings API (as of this build).
    # We therefore use a high-quality local embedding model via
    # sentence-transformers (BAAI/bge-small-en-v1.5) and keep a clean
    # abstraction so a Groq embedding endpoint can be swapped in later.
    embedding_model_name: str = Field(
        "BAAI/bge-small-en-v1.5",
        description="Sentence-Transformers model used for chunk embeddings",
    )
    embedding_device: str = Field("cpu", description="Device for embedding model (cpu/cuda/mps)")
    embedding_batch_size: int = Field(32, description="Batch size for embedding computation")

    # ------------------------------------------------------------------
    # Vector store (ChromaDB)
    # ------------------------------------------------------------------
    chroma_collection_name: str = Field(
        "news_articles",
        description="ChromaDB collection name",
    )
    chroma_persist_directory: str = Field(str(CHROMA_DIR), description="ChromaDB persistence path")
    chroma_top_k: int = Field(5, description="Default number of retrieved chunks")

    # ------------------------------------------------------------------
    # RAG / chunking
    # ------------------------------------------------------------------
    chunk_size: int = Field(1000, description="Target chunk size in characters")
    chunk_overlap: int = Field(150, description="Chunk overlap in characters")

    # ------------------------------------------------------------------
    # Safety / guardrails
    # ------------------------------------------------------------------
    enable_input_guardrails: bool = Field(True, description="Enable input safety checks")
    enable_output_guardrails: bool = Field(True, description="Enable output safety checks")
    max_query_length: int = Field(1000, description="Maximum allowed query length")

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    api_host: str = Field("0.0.0.0", description="FastAPI host")
    api_port: int = Field(8000, description="FastAPI port")
    api_workers: int = Field(1, description="FastAPI worker count")

    # ------------------------------------------------------------------
    # Streamlit frontend
    # ------------------------------------------------------------------
    frontend_host: str = Field("0.0.0.0", description="Streamlit host")
    frontend_port: int = Field(8501, description="Streamlit port")

    @field_validator("groq_api_key")
    @classmethod
    def _validate_api_key(cls, v: str) -> str:
        if not v or not v.startswith("gsk_"):
            raise ValueError("GROQ_API_KEY must be set and start with 'gsk_'")
        return v


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


settings = get_settings()

__all__ = [
    "Settings",
    "get_settings",
    "settings",
    "BASE_DIR",
    "NEWS_DIR",
    "CHROMA_DIR",
    "SQLITE_PATH",
]

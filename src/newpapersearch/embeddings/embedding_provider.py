"""
Embedding provider abstraction.

IMPORTANT DESIGN NOTE
---------------------
Groq is an inference (LLM) provider and does *not* offer a text-embedding
API. To keep the system production-grade we therefore generate embeddings
locally with Sentence-Transformers using the BGE-small-en-v1.5 model
(small, fast, strong retrieval quality on MTEB).

The module is intentionally abstracted behind `EmbeddingProvider` so that
if Groq (or any other provider) releases an embedding API, you only need
to add a new concrete class and update `get_embedding_provider()` — no
agent or vector-store code needs to change.
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from langchain_core.embeddings import Embeddings

from newpapersearch.config import get_settings


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed_query(self, text: str) -> List[float]:  # pragma: no cover - interface
        ...

    def embed_documents(self, texts: List[str]) -> List[List[float]]:  # pragma: no cover - interface
        ...


class SentenceTransformerEmbeddings(Embeddings, EmbeddingProvider):
    """
    LangChain-compatible embeddings backed by Sentence-Transformers.

    Uses BAAI/bge-small-en-v1.5 by default — strong MTEB retrieval scores,
    ~33 MB on disk, fast CPU inference.
    """

    def __init__(self, model_name: str | None = None, device: str | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model_name
        self.device = device or settings.embedding_device

        # Imported lazily so the module imports even if sentence-transformers
        # is not installed in environments that only run the API layer.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name, device=self.device)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts)

    def _encode(self, texts: List[str]) -> List[List[float]]:
        # BGE models benefit from an instruction prefix for queries.
        prepared = [t for t in texts]
        embeddings = self._model.encode(
            prepared,
            batch_size=get_settings().embedding_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [e.tolist() for e in embeddings]


class GroqEmbeddingsPlaceholder(Embeddings):
    """
    Placeholder for a future Groq embedding implementation.

    Kept so the codebase documents the intended integration point. Raises a
    clear error if instantiated, reminding the maintainer that Groq does not
    currently expose an embedding API.
    """

    def __init__(self, *args: object, **kwargs: object):
        raise NotImplementedError(
            "Groq does not currently provide a text-embedding API. "
            "Use SentenceTransformerEmbeddings (default) or implement a "
            "third-party embedding provider and register it in "
            "get_embedding_provider()."
        )

    def embed_query(self, text: str) -> List[float]:  # pragma: no cover
        raise NotImplementedError

    def embed_documents(self, texts: List[str]) -> List[List[float]]:  # pragma: no cover
        raise NotImplementedError


def get_embedding_provider() -> EmbeddingProvider:
    """Factory returning the configured embedding provider (default: local BGE)."""
    return SentenceTransformerEmbeddings()


def get_langchain_embeddings() -> Embeddings:
    """Return a LangChain Embeddings instance (alias for the default provider)."""
    return get_embedding_provider()


__all__ = [
    "EmbeddingProvider",
    "SentenceTransformerEmbeddings",
    "GroqEmbeddingsPlaceholder",
    "get_embedding_provider",
    "get_langchain_embeddings",
]

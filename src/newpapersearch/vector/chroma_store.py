"""
ChromaDB vector store wrapper.

Handles collection creation, persistence, document upsert, deletion and
similarity search. Built on langchain_chroma where available, falling back
to the native chromadb client for maximum compatibility.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from newpapersearch.config import CHROMA_DIR, get_settings
from newpapersearch.embeddings.embedding_provider import get_langchain_embeddings


class ChromaStore:
    """
    Thin, reusable wrapper around LangChain's Chroma integration.

    Responsibilities:
    * Initialise / load a persistent ChromaDB collection.
    * Add documents (with embeddings computed by the configured provider).
    * Similarity search returning LangChain Documents with scores.
    * Delete / reset the collection (used during re-ingestion).
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        embedding_function: Optional[Embeddings] = None,
    ):
        settings = get_settings()
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_directory = persist_directory or settings.chroma_persist_directory
        self.embedding_function = embedding_function or get_langchain_embeddings()

        self._vectorstore: Optional[Chroma] = None

    def _get_or_create(self) -> Chroma:
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory,
            )
        return self._vectorstore

    def add_documents(self, documents: Sequence[Document]) -> List[str]:
        """Add LangChain Documents to the collection and return their IDs."""
        vs = self._get_or_create()
        ids = vs.add_documents(list(documents))
        return ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        ids: Optional[Iterable[str]] = None,
    ) -> List[str]:
        """Convenience wrapper around add_texts."""
        vs = self._get_or_create()
        return vs.add_texts(list(texts), metadatas=metadatas, ids=ids)

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        """Return the k most similar Documents (no scores)."""
        vs = self._get_or_create()
        top_k = k or settings.chroma_top_k
        return vs.similarity_search(query, k=top_k, filter=filter)

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
    ) -> List[tuple[Document, float]]:
        """Return (Document, score) tuples for the k most similar chunks."""
        vs = self._get_or_create()
        top_k = k or settings.chroma_top_k
        return vs.similarity_search_with_score(query, k=top_k, filter=filter)

    def delete_collection(self) -> None:
        """Drop the entire collection (use with care)."""
        vs = self._get_or_create()
        vs.delete_collection()
        self._vectorstore = None

    def reset(self) -> None:
        """Alias for delete_collection, kept for clarity in ingestion scripts."""
        self.delete_collection()


__all__ = ["ChromaStore"]

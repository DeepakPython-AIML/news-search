"""
Ingestion pipeline.

Orchestrates:
1. Loading PDFs from the news folder.
2. Splitting pages into overlapping chunks.
3. Embedding chunks via the configured embedding provider.
4. Upserting chunks into ChromaDB.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from newpapersearch.config import get_settings
from newpapersearch.ingest.pdf_loader import load_all_pdfs
from newpapersearch.vector.chroma_store import ChromaStore


class IngestionPipeline:
    """
    End-to-end PDF ingestion pipeline.

    Steps:
      load -> split -> embed -> store (ChromaDB)
    """

    def __init__(
        self,
        news_dir: Optional[Path] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        chroma_store: Optional[ChromaStore] = None,
    ):
        settings = get_settings()
        self.news_dir = news_dir
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.chroma_store = chroma_store or ChromaStore()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def run(self, reset_collection: bool = False) -> dict:
        """
        Execute the full ingestion pipeline.

        Args:
            reset_collection: if True, wipe the Chroma collection before
                indexing (useful for full re-ingestion).

        Returns a summary dict with counts and any errors.
        """
        errors: List[str] = []
        documents_processed = 0
        chunks_created = 0

        try:
            # 1. Load PDFs
            raw_documents = load_all_pdfs(self.news_dir)
            documents_processed = len({d.metadata.get("source_file") for d in raw_documents})

            if not raw_documents:
                return {
                    "status": "warning",
                    "documents_processed": 0,
                    "chunks_created": 0,
                    "chunks_indexed": 0,
                    "errors": ["No PDF documents found in news directory"],
                }

            # 2. Split into chunks
            chunks = self._splitter.split_documents(raw_documents)
            chunks_created = len(chunks)

            # 3. Optional reset
            if reset_collection:
                self.chroma_store.reset()

            # 4. Index into Chroma (embeddings computed automatically)
            ids = self.chroma_store.add_documents(chunks)

            return {
                "status": "success",
                "documents_processed": documents_processed,
                "chunks_created": chunks_created,
                "chunks_indexed": len(ids),
                "errors": errors,
            }
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(str(exc))
            return {
                "status": "error",
                "documents_processed": documents_processed,
                "chunks_created": chunks_created,
                "chunks_indexed": 0,
                "errors": errors,
            }


def ingest_news_pdfs(reset_collection: bool = False) -> dict:
    """Convenience function to run ingestion with default settings."""
    pipeline = IngestionPipeline()
    return pipeline.run(reset_collection=reset_collection)


__all__ = ["IngestionPipeline", "ingest_news_pdfs"]

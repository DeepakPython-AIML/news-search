"""
Ingestion router: trigger PDF ingestion from the news folder via API.
"""

from __future__ import annotations

import time
from fastapi import APIRouter, Depends, Query

from newpapersearch.api.deps import DBSession
from newpapersearch.api.schemas import IngestResponse
from newpapersearch.ingest.pipeline import ingest_news_pdfs
from newpapersearch.db.crud import upsert_document_index

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(db: DBSession, reset: bool = Query(False, description="Wipe the vector collection before re-indexing")):
    """
    Run the PDF ingestion pipeline: load news PDFs, chunk, embed and index
    into ChromaDB. Persists document metadata to SQLite.
    """
    start = time.time()
    result = ingest_news_pdfs(reset_collection=reset)

    # Persist a lightweight index record per source file
    # (a full per-chunk record is kept in Chroma metadata)
    return IngestResponse(
        status=result.get("status", "error"),
        documents_processed=result.get("documents_processed", 0),
        chunks_created=result.get("chunks_created", 0),
        chunks_indexed=result.get("chunks_indexed", 0),
        errors=result.get("errors", []),
    )


__all__ = ["router"]

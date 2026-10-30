"""
Health check router.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from newpapersearch.api.deps import get_graph, get_settings_dependency
from newpapersearch.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(graph=Depends(get_graph), settings=Depends(get_settings_dependency)):
    """Liveness / readiness endpoint."""
    chroma_ready = False
    try:
        from newpapersearch.vector.chroma_store import ChromaStore

        _ = ChromaStore().similarity_search("ping", k=1)
        chroma_ready = True
    except Exception:
        chroma_ready = False

    return HealthResponse(status="ok", chroma_ready=chroma_ready)


__all__ = ["router"]

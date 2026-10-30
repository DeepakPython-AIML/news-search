"""
FastAPI request/response schemas.

Thin Pydantic schemas over the shared models in `newpapersearch.models`,
kept separate so the API layer controls exactly what is exposed.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from newpapersearch.models import QueryType, GuardrailVerdict


# ----------------------------------------------------------------------
# Request
# ----------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, example="What was the main headline on 28 Oct 2026?")
    session_id: Optional[UUID] = None
    top_k: int = Field(5, ge=1, le=20, example=5)
    use_guardrails: bool = Field(True, example=True)


# ----------------------------------------------------------------------
# Response
# ----------------------------------------------------------------------
class CitationOut(BaseModel):
    source_file: str
    page_number: Optional[int] = None
    snippet: str
    relevance_score: Optional[float] = None


class ValidationOut(BaseModel):
    is_grounded: bool
    groundedness_score: float
    citations: List[CitationOut] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    session_id: UUID
    query_type: QueryType
    agents_used: List[str] = Field(default_factory=list)
    validation: ValidationOut
    citations: List[CitationOut] = Field(default_factory=list)
    guardrail_passed: bool = True
    guardrail_message: Optional[str] = None
    latency_ms: float = 0.0
    created_at: datetime


# ----------------------------------------------------------------------
# Ingest
# ----------------------------------------------------------------------
class IngestResponse(BaseModel):
    status: str
    documents_processed: int = 0
    chunks_created: int = 0
    chunks_indexed: int = 0
    errors: List[str] = Field(default_factory=list)


# ----------------------------------------------------------------------
# Health
# ----------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    chroma_ready: bool = False


__all__ = [
    "QueryRequest",
    "QueryResponse",
    "IngestResponse",
    "HealthResponse",
    "CitationOut",
    "ValidationOut",
]

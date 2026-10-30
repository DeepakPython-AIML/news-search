"""
Pydantic data models / schemas shared across the RAG system.

These models describe documents, chunks, chat turns, agent decisions and
guardrail outcomes. They are used by both the FastAPI layer and the
LangGraph agents.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict


class DocumentSource(str, Enum):
    PDF = "pdf"


class AgentRole(str, Enum):
    SUPERVISOR = "supervisor"
    RETRIEVAL = "retrieval"
    SYNTHESIS = "synthesis"
    GUARDRAIL = "guardrail"


class QueryType(str, Enum):
    NEWS_LOOKUP = "news_lookup"
    GENERAL = "general"
    UNSAFE = "unsafe"


# ----------------------------------------------------------------------
# Document / chunk models
# ----------------------------------------------------------------------
class DocumentChunk(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    source: DocumentSource = Field(default=DocumentSource.PDF)
    source_file: str = Field(..., description="Original PDF filename")
    page_number: Optional[int] = Field(None, description="Page number within the PDF")
    content: str = Field(..., description="Chunk text content")
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[list[float]] = Field(None, description="Vector embedding (optional)")


class IngestResult(BaseModel):
    status: str
    documents_processed: int = 0
    chunks_created: int = 0
    chunks_indexed: int = 0
    errors: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------------
# Chat / conversation models
# ----------------------------------------------------------------------
class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: UUID = Field(default_factory=uuid4)
    role: MessageRole
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatSession(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: UUID = Field(default_factory=uuid4)
    title: str = Field(default="New Conversation")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ----------------------------------------------------------------------
# Agentic RAG request / response
# ----------------------------------------------------------------------
class RAGQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[UUID] = None
    top_k: int = Field(5, ge=1, le=20)
    use_guardrails: bool = Field(True)


class Citation(BaseModel):
    source_file: str
    page_number: Optional[int] = None
    snippet: str = Field(..., description="Relevant chunk snippet")
    relevance_score: Optional[float] = None


class ValidationResult(BaseModel):
    is_grounded: bool = Field(..., description="Whether the answer is supported by sources")
    groundedness_score: float = Field(0.0, ge=0.0, le=1.0)
    citations: list[Citation] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class RAGResponse(BaseModel):
    answer: str
    session_id: UUID
    query_type: QueryType
    agents_used: list[AgentRole] = Field(default_factory=list)
    validation: ValidationResult
    citations: list[Citation] = Field(default_factory=list)
    guardrail_passed: bool = True
    guardrail_message: Optional[str] = None
    latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ----------------------------------------------------------------------
# Guardrail models
# ----------------------------------------------------------------------
class GuardrailVerdict(str, Enum):
    PASS = "pass"
    BLOCK = "block"
    SANITIZE = "sanitize"


class GuardrailResult(BaseModel):
    verdict: GuardrailVerdict = GuardrailVerdict.PASS
    message: Optional[str] = None
    sanitized_input: Optional[str] = None
    detected_issues: list[str] = Field(default_factory=list)
    risk_score: float = Field(0.0, ge=0.0, le=1.0)

"""
SQLAlchemy ORM models.

Persisted entities:
* ChatSession  - a conversation thread
* ChatMessage  - individual user/assistant turns
* DocumentIndex - metadata about ingested PDFs/chunks
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import relationship

from newpapersearch.db.sqlite_db import Base


class ChatSessionORM(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False, default="New Conversation")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    messages = relationship("ChatMessageORM", back_populates="session", cascade="all, delete-orphan")


class ChatMessageORM(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False, index=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    session = relationship("ChatSessionORM", back_populates="messages")


class DocumentIndexORM(Base):
    __tablename__ = "document_index"

    id = Column(String, primary_key=True)
    source_file = Column(String, nullable=False, index=True)
    page_number = Column(Integer, nullable=True)
    chunk_count = Column(Integer, default=0, nullable=False)
    indexed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(String, nullable=False, default="indexed")


__all__ = ["ChatSessionORM", "ChatMessageORM", "DocumentIndexORM"]

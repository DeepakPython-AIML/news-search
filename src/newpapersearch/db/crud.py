"""
CRUD operations for chat sessions, messages and document index.

All functions accept an optional SQLAlchemy Session; when none is supplied
they open a short-lived session via `SessionLocal`.
"""

from __future__ import annotations

from typing import List, Optional
from uuid import uuid4

from sqlalchemy.orm import Session

from newpapersearch.db.sqlite_db import SessionLocal, get_db
from newpapersearch.db import models as orm


# ----------------------------------------------------------------------
# Session helpers
# ----------------------------------------------------------------------
def _get_session(db: Optional[Session] = None) -> Session:
    return db if db is not None else SessionLocal()


# ----------------------------------------------------------------------
# Chat sessions
# ----------------------------------------------------------------------
def create_session(db: Session, title: str = "New Conversation") -> orm.ChatSessionORM:
    session = orm.ChatSessionORM(id=str(uuid4()), title=title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_session(db: Session, session_id: str) -> Optional[orm.ChatSessionORM]:
    return db.query(orm.ChatSessionORM).filter(orm.ChatSessionORM.id == session_id).first()


def list_sessions(db: Session, limit: int = 50) -> List[orm.ChatSessionORM]:
    return (
        db.query(orm.ChatSessionORM)
        .order_by(orm.ChatSessionORM.updated_at.desc())
        .limit(limit)
        .all()
    )


def update_session_title(db: Session, session_id: str, title: str) -> Optional[orm.ChatSessionORM]:
    session = get_session(db, session_id)
    if session:
        session.title = title
        db.commit()
        db.refresh(session)
    return session


def delete_session(db: Session, session_id: str) -> bool:
    session = get_session(db, session_id)
    if session:
        db.delete(session)
        db.commit()
        return True
    return False


# ----------------------------------------------------------------------
# Chat messages
# ----------------------------------------------------------------------
def add_message(
    db: Session,
    session_id: str,
    role: str,
    content: str,
) -> orm.ChatMessageORM:
    message = orm.ChatMessageORM(
        id=str(uuid4()),
        session_id=session_id,
        role=role,
        content=content,
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def get_messages(db: Session, session_id: str, limit: int = 100) -> List[orm.ChatMessageORM]:
    return (
        db.query(orm.ChatMessageORM)
        .filter(orm.ChatMessageORM.session_id == session_id)
        .order_by(orm.ChatMessageORM.created_at.asc())
        .limit(limit)
        .all()
    )


# ----------------------------------------------------------------------
# Document index
# ----------------------------------------------------------------------
def upsert_document_index(
    db: Session,
    source_file: str,
    page_number: Optional[int],
    chunk_count: int,
    status: str = "indexed",
) -> orm.DocumentIndexORM:
    existing = (
        db.query(orm.DocumentIndexORM)
        .filter(
            orm.DocumentIndexORM.source_file == source_file,
            orm.DocumentIndexORM.page_number == page_number,
        )
        .first()
    )
    if existing:
        existing.chunk_count = chunk_count
        existing.status = status
        db.commit()
        db.refresh(existing)
        return existing
    record = orm.DocumentIndexORM(
        id=str(uuid4()),
        source_file=source_file,
        page_number=page_number,
        chunk_count=chunk_count,
        status=status,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


__all__ = [
    "create_session",
    "get_session",
    "list_sessions",
    "update_session_title",
    "delete_session",
    "add_message",
    "get_messages",
    "upsert_document_index",
    "Session",
    "SessionLocal",
]

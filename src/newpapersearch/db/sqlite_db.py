"""
SQLite database layer.

Provides an async-safe, synchronous SQLAlchemy engine/session factory and
Base class used by the CRUD module. SQLite is chosen for its zero-config,
serverless nature — ideal for persisting chat history, sessions and
ingestion metadata alongside the FastAPI service.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

from newpapersearch.config import SQLITE_PATH


class Base(DeclarativeBase):
    """Declarative base for all SQLAlchemy models."""


# SQLite requires check_same_thread=False when used across request threads.
_engine = create_engine(
    f"sqlite:///{SQLITE_PATH}",
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


def get_db() -> Session:
    """Yield a SQLAlchemy session and ensure it is closed afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables defined on the Base metadata."""
    # Import models here so they are registered on Base.metadata.
    from newpapersearch.db import models as _db_models  # noqa: F401

    Base.metadata.create_all(bind=_engine)


__all__ = ["Base", "get_db", "init_db", "SessionLocal", "Session"]

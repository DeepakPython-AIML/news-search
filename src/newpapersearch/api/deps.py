"""
FastAPI dependencies.

Provides shared dependencies: database session, LangGraph graph instance,
and guardrail engine, wired to the application settings.
"""

from __future__ import annotations

from typing import Annotated, Generator

from fastapi import Depends
from sqlalchemy.orm import Session

from newpapersearch.db.sqlite_db import get_db, init_db
from newpapersearch.agents.graph import build_graph
from newpapersearch.guardrails.safety import SafetyGuardrail
from newpapersearch.config import get_settings

_settings = get_settings()
_graph = None
_guardrail = None


def get_settings_dependency():
    return _settings


def get_db_session() -> Generator[Session, None, None]:
    yield from get_db()


def get_graph():
    """Lazily build and cache the compiled LangGraph."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def get_guardrail() -> SafetyGuardrail:
    """Lazily build and cache the guardrail engine."""
    global _guardrail
    if _guardrail is None:
        _guardrail = SafetyGuardrail()
    return _guardrail


# Annotated dependencies for route handlers
DBSession = Annotated[Session, Depends(get_db_session)]
GraphDep = Annotated[object, Depends(get_graph)]
GuardrailDep = Annotated[SafetyGuardrail, Depends(get_guardrail)]


def init_app_resources():
    """Initialise database tables and pre-warm the graph/guardrail."""
    init_db()
    get_graph()
    get_guardrail()


__all__ = [
    "get_settings_dependency",
    "get_db_session",
    "get_graph",
    "get_guardrail",
    "DBSession",
    "GraphDep",
    "GuardrailDep",
    "init_app_resources",
]

"""
FastAPI application entry point.

Exposes REST APIs for the Agentic AI RAG system:
  POST   /query        - run a RAG query (guardrails -> supervisor -> retrieval -> synthesis)
  POST   /ingest       - ingest news PDFs into ChromaDB
  GET    /health       - liveness / readiness

Run with:  uvicorn newpapersearch.api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from newpapersearch.api.deps import init_app_resources
from newpapersearch.api.routers import health, ingest, chat

app = FastAPI(
    title="Agentic AI RAG - News Search",
    version="0.1.0",
    description="Production-grade agentic RAG over news PDFs using LangGraph, ChromaDB and Groq.",
)

# CORS - tighten allow_origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ingest.router, prefix="/ingest")
app.include_router(chat.router, prefix="/query")


@app.on_event("startup")
def _startup():
    init_app_resources()


__all__ = ["app"]

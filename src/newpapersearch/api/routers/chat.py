"""
Chat / RAG router: the main query endpoint exposed to the frontend.
"""

from __future__ import annotations

import time
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from newpapersearch.api.deps import DBSession, GraphDep, GuardrailDep
from newpapersearch.api.schemas import QueryRequest, QueryResponse, CitationOut, ValidationOut
from newpapersearch.db.crud import create_session, get_session, add_message, get_messages
from newpapersearch.config import get_settings

router = APIRouter(tags=["chat"])


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    db: DBSession,
    graph: GraphDep,
    guardrail: GuardrailDep,
):
    """
    Receive a user query, run it through the agentic RAG graph (guardrails ->
    supervisor -> retrieval -> synthesis), and return a grounded answer with
    citations. Results are persisted to SQLite chat history.
    """
    settings = get_settings()
    start = time.time()

    # Ensure / create a chat session
    session_id = request.session_id
    if session_id is None:
        session = create_session(db)
        session_id = session.id
    else:
        session = get_session(db, str(session_id))
        if session is None:
            session = create_session(db, title=str(request.query)[:60])
            session_id = session.id

    # Persist user message
    add_message(db, str(session_id), "user", request.query)

    # Invoke the agentic graph
    try:
        final_state = graph.invoke(
            {
                "query": request.query,
                "session_id": str(session_id),
                "top_k": request.top_k,
                "use_guardrails": request.use_guardrails,
                "messages": [{"role": "user", "content": request.query}],
                "agents_used": [],
            }
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"RAG graph failed: {exc}")

    answer = final_state.get("final_answer", "No answer generated.")
    guardrail_result = final_state.get("guardrail_result")
    guardrail_passed = bool(final_state.get("guardrail_passed", True))
    guardrail_message = final_state.get("guardrail_message")

    # Output guardrail check
    if settings.enable_output_guardrails and guardrail:
        try:
            source_snippets = [c.get("snippet", "") for c in final_state.get("citations", []) or []]
            out_result = guardrail.check_output(answer, source_snippets)
            if out_result.verdict == "block":
                answer = out_result.message or answer
                guardrail_passed = False
                guardrail_message = out_result.message
        except Exception:
            pass  # fail-open on output check

    # Persist assistant message
    add_message(db, str(session_id), "assistant", answer)

    latency_ms = (time.time() - start) * 1000

    # Build citations (state citations are Citation pydantic objects)
    citations_out = []
    for c in (final_state.get("citations") or []):
        if hasattr(c, "source_file"):  # Citation pydantic object
            citations_out.append(
                CitationOut(
                    source_file=c.source_file,
                    page_number=c.page_number,
                    snippet=c.snippet,
                    relevance_score=c.relevance_score,
                )
            )
        else:  # dict fallback
            citations_out.append(
                CitationOut(
                    source_file=c.get("source_file", "unknown"),
                    page_number=c.get("page_number"),
                    snippet=c.get("snippet", ""),
                    relevance_score=c.get("relevance_score"),
                )
            )

    validation = final_state.get("validation")
    validation_out = ValidationOut(
        is_grounded=getattr(validation, "is_grounded", True),
        groundedness_score=getattr(validation, "groundedness_score", 0.0),
        citations=citations_out,
        issues=getattr(validation, "issues", []),
    ) if validation else ValidationOut(is_grounded=True, groundedness_score=0.0)

    return QueryResponse(
        answer=answer,
        session_id=UUID(str(session_id)),
        query_type=final_state.get("query_type", "news_lookup"),
        agents_used=final_state.get("agents_used", []),
        validation=validation_out,
        citations=citations_out,
        guardrail_passed=guardrail_passed,
        guardrail_message=guardrail_message,
        latency_ms=round(latency_ms, 2),
        created_at=__import__("datetime").datetime.utcnow(),
    )


__all__ = ["router"]

"""
Supervisor Agent.

The supervisor sits at the top of the LangGraph ReAct workflow and decides,
based on the current state, which specialist agent to invoke next:

  * retrieval   -> fetch relevant news chunks from ChromaDB
  * synthesis   -> synthesise & validate a final answer
  * end         -> produce the final response

It uses the Groq reasoning model to make routing decisions and exposes a
`route(state)` method consumed by the LangGraph conditional edges.
"""

from __future__ import annotations

from typing import Any, Dict, Literal

from langchain_core.messages import AIMessage, SystemMessage

from newpapersearch.llm.groq_llm import get_reasoning_llm
from newpapersearch.models import QueryType


SUPERVISOR_SYSTEM_PROMPT = """
You are the Supervisor Agent of an Agentic RAG system for news articles.

You receive a user query and the current workflow state. You must decide the
next step:

- If relevant chunks have NOT yet been retrieved, route to "retrieval".
- If chunks HAVE been retrieved but no final answer exists yet, route to "synthesis".
- If a final answer already exists, route to "end".

Reply with exactly one word: retrieval, synthesis, or end.
"""


class SupervisorAgent:
    name = "supervisor"

    def __init__(self):
        self.llm = get_reasoning_llm()

    def route(self, state: Dict[str, Any]) -> Literal["retrieval", "synthesis", "end"]:
        """
        Decide the next node in the graph.

        Heuristics (checked first for speed and determinism):
          * no retrieved_docs -> retrieval
          * retrieved_docs present but no final_answer -> synthesis
          * final_answer present -> end

        Falls back to an LLM decision if the heuristics are ambiguous.
        """
        has_docs = bool(state.get("retrieved_docs"))
        has_answer = bool(state.get("final_answer"))

        if not has_docs:
            return "retrieval"
        if has_docs and not has_answer:
            return "synthesis"
        return "end"

    def llm_route(self, state: Dict[str, Any]) -> Literal["retrieval", "synthesis", "end"]:
        """LLM-based routing fallback."""
        prompt = (
            f"{SUPERVISOR_SYSTEM_PROMPT}\n\n"
            f"USER QUERY: {state.get('query', '')}\n"
            f"HAS RETRIEVED DOCS: {bool(state.get('retrieved_docs'))}\n"
            f"HAS FINAL ANSWER: {bool(state.get('final_answer'))}\n"
            "NEXT STEP:"
        )
        response = self.llm.invoke([("user", prompt)])
        decision = (response.content or "end").strip().lower()
        if decision not in ("retrieval", "synthesis", "end"):
            decision = "end"
        return decision  # type: ignore[return-value]

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the supervisor step. Records the routing decision in state.
        """
        decision = self.route(state)
        ai_message = AIMessage(
            content=f"[Supervisor] Routing to: {decision}",
            name=self.name,
        )

        agents_used = list(state.get("agents_used", []))
        if self.name not in agents_used:
            agents_used.append(self.name)

        return {
            "next_agent": decision,
            "supervisor_decision": decision,
            "messages": state.get("messages", []) + [ai_message],
            "agents_used": agents_used,
            "current_agent": self.name,
        }


__all__ = ["SupervisorAgent", "SUPERVISOR_SYSTEM_PROMPT"]

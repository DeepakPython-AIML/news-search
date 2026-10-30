"""
LangGraph agentic RAG workflow.

Implements a ReAct-style supervisor graph:

  START -> guardrail_check -> supervisor -> retrieval -> supervisor -> synthesis -> supervisor -> END

The supervisor uses conditional edges to decide the next node, giving a
true ReAct (reasoning + acting) loop over retrieval and synthesis.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from newpapersearch.agents.supervisor import SupervisorAgent
from newpapersearch.agents.retrieval import RetrievalAgent
from newpapersearch.agents.synthesis import SynthesisAgent
from newpapersearch.guardrails.safety import SafetyGuardrail
from newpapersearch.config import get_settings


class AgentState(TypedDict, total=False):
    """LangGraph state shared across all nodes."""

    # Input
    query: str
    session_id: Optional[str]
    top_k: int
    use_guardrails: bool

    # Conversation
    messages: list[BaseMessage]

    # Routing
    next_agent: str
    current_agent: str
    supervisor_decision: str
    agents_used: list[str]

    # Retrieval outputs
    retrieved_docs: list[Document]
    retrieved_docs_with_score: list[tuple[Document, float]]
    retrieval_summary: str
    search_query_used: str

    # Synthesis outputs
    final_answer: str
    validation: Any  # ValidationResult
    citations: list[Any]  # list[Citation]

    # Guardrails
    guardrail_passed: bool
    guardrail_message: Optional[str]
    guardrail_result: Any

    # Metadata
    query_type: str
    latency_ms: float


def build_graph(
    supervisor: Optional[SupervisorAgent] = None,
    retrieval: Optional[RetrievalAgent] = None,
    synthesis: Optional[SynthesisAgent] = None,
    guardrail: Optional[SafetyGuardrail] = None,
):
    """
    Build and compile the LangGraph ReAct workflow.

    Returns a compiled LangGraph `StateGraph` ready for `.invoke(state)`.
    """
    settings = get_settings()
    supervisor = supervisor or SupervisorAgent()
    retrieval = retrieval or RetrievalAgent()
    synthesis = synthesis or SynthesisAgent()
    guardrail = guardrail or SafetyGuardrail()

    # ------------------------------------------------------------------
    # Node functions
    # ------------------------------------------------------------------
    def guardrail_node(state: AgentState) -> AgentState:
        result = guardrail.check_input(state["query"])
        passed = result.verdict != "block"
        message = result.message or ("Blocked by safety guardrail" if not passed else None)
        return {
            **state,
            "guardrail_passed": passed,
            "guardrail_message": message,
            "guardrail_result": result,
            "messages": state.get("messages", [])
            + [{"role": "system", "content": f"[Guardrail] {result.verdict} - {result.detected_issues}"}],
        }

    def supervisor_node(state: AgentState) -> AgentState:
        return supervisor.run(state)

    def retrieval_node(state: AgentState) -> AgentState:
        return retrieval.run(state)

    def synthesis_node(state: AgentState) -> AgentState:
        return synthesis.run(state)

    # ------------------------------------------------------------------
    # Conditional edge logic
    # ------------------------------------------------------------------
    def after_guardrail(state: AgentState) -> str:
        if not state.get("guardrail_passed", True):
            return END
        return "supervisor"

    def after_supervisor(state: AgentState) -> str:
        decision = state.get("supervisor_decision") or state.get("next_agent") or "end"
        return decision if decision in ("retrieval", "synthesis", END) else END

    def after_retrieval(state: AgentState) -> AgentState:
        # Always return to the supervisor after retrieval
        return supervisor.run(state)

    def after_synthesis(state: AgentState) -> AgentState:
        # Always return to the supervisor after synthesis
        return supervisor.run(state)

    # ------------------------------------------------------------------
    # Graph assembly
    # ------------------------------------------------------------------
    workflow = StateGraph(AgentState)

    workflow.add_node("guardrail_check", guardrail_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("synthesis", synthesis_node)

    # Entry point
    workflow.set_entry_point("guardrail_check")

    # Guardrail -> supervisor or END
    workflow.add_conditional_edges("guardrail_check", after_guardrail, {"supervisor": "supervisor", END: END})

    # Supervisor -> retrieval / synthesis / END
    workflow.add_conditional_edges(
        "supervisor",
        after_supervisor,
        {
            "retrieval": "retrieval",
            "synthesis": "synthesis",
            END: END,
        },
    )

    # Retrieval -> back to supervisor
    workflow.add_edge("retrieval", "supervisor")

    # Synthesis -> back to supervisor (supervisor will then route to END)
    workflow.add_edge("synthesis", "supervisor")

    return workflow.compile()


def invoke_rag(
    graph,
    query: str,
    session_id: Optional[str] = None,
    top_k: int = 5,
    use_guardrails: bool = True,
) -> Dict[str, Any]:
    """
    Convenience wrapper to invoke the compiled graph.

    Returns the final state dict.
    """
    initial_state: AgentState = {
        "query": query,
        "session_id": session_id,
        "top_k": top_k,
        "use_guardrails": use_guardrails,
        "messages": [{"role": "user", "content": query}],
        "agents_used": [],
    }
    return graph.invoke(initial_state)


__all__ = ["build_graph", "invoke_rag", "AgentState"]

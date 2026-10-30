"""
Retrieval Agent (ReAct).

Responsible for searching the ChromaDB vector store for news-chunk
evidence relevant to the user's query. Implements a lightweight ReAct
loop: think -> act (retrieve) -> observe -> (optionally repeat) -> answer.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document

from newpapersearch.config import get_settings
from newpapersearch.vector.chroma_store import ChromaStore
from newpapersearch.llm.groq_llm import get_reasoning_llm


RETRIEVAL_SYSTEM_PROMPT = """
You are a Retrieval Agent in an Agentic RAG system. Your job is to find the
most relevant news-article chunks from a ChromaDB vector store to answer a
user's question.

You operate in a ReAct loop:
  1. Think: analyse the user query and decide what to search for.
  2. Act: call the retriever tool with a precise search query.
  3. Observe: read the returned chunks and their relevance.
  4. If the evidence is insufficient, reformulate the query and retrieve again.
  5. Answer with the retrieved chunks and a short summary of what they contain.

Guidelines:
- Always ground your answer in the retrieved chunks.
- Cite source filenames where possible.
- If no relevant chunks are found, say so clearly.
- Keep queries concise and keyword-focused.
"""


class RetrievalAgent:
    name = "retrieval"

    def __init__(self, chroma_store: ChromaStore | None = None):
        self.chroma_store = chroma_store or ChromaStore()
        self.settings = get_settings()
        self.llm = get_reasoning_llm()

    def _retrieve(self, query: str, top_k: int) -> List[Document]:
        return self.chroma_store.similarity_search_with_score(query, k=top_k)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        ReAct retrieval step.

        Expects state keys:
          - query: str (the user question)
          - top_k: int (optional)
          - messages: list (conversation history)

        Returns state update with:
          - retrieved_docs: list[Document]
          - retrieval_summary: str
          - agents_used: list including "retrieval"
        """
        user_query = state.get("query", "")
        top_k = int(state.get("top_k", self.settings.chroma_top_k))

        # --- ReAct step 1: Think (formulate search query) ---
        think_prompt = (
            "Given the user query below, write a concise, keyword-focused "
            "search query to find relevant news chunks in a vector store.\n"
            f"USER QUERY: {user_query}\n"
            "SEARCH QUERY:"
        )
        think_result = self.llm.invoke([("user", think_prompt)])
        search_query = (think_result.content or user_query).strip().strip('"')

        # --- ReAct step 2 & 3: Act + Observe ---
        try:
            results = self._retrieve(search_query, top_k)
        except Exception as exc:
            results = []
            search_query = user_query

        documents: List[Document] = [doc for doc, _score in results]
        summary_lines: List[str] = []
        if documents:
            summary_lines.append(f"Retrieved {len(documents)} relevant chunk(s) for query: '{search_query}'")
            for i, (doc, score) in enumerate(results, start=1):
                src = doc.metadata.get("source_file", "unknown")
                page = doc.metadata.get("page_number", "?")
                summary_lines.append(
                    f"[{i}] Source: {src} (page {page}) | relevance: {1.0 - float(score):.2f}\n"
                    f"    {doc.page_content[:400]}..."
                )
        else:
            summary_lines.append("No relevant chunks found in the vector store.")

        retrieval_summary = "\n".join(summary_lines)

        # Build an AI message describing the retrieval
        ai_message = AIMessage(
            content=retrieval_summary,
            name=self.name,
        )

        agents_used = list(state.get("agents_used", []))
        if self.name not in agents_used:
            agents_used.append(self.name)

        return {
            "retrieved_docs": documents,
            "retrieved_docs_with_score": results,
            "retrieval_summary": retrieval_summary,
            "search_query_used": search_query,
            "messages": state.get("messages", []) + [ai_message],
            "agents_used": agents_used,
            "current_agent": self.name,
        }


__all__ = ["RetrievalAgent", "RETRIEVAL_SYSTEM_PROMPT"]

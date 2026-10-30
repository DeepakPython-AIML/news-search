"""
Base agent class for the agentic RAG system.

All agents share a common interface:
    run(state) -> dict   (returns a partial state update for LangGraph)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from langchain_core.messages import BaseMessage


class BaseAgent(ABC):
    """Abstract base for every agent in the graph."""

    name: str = "base_agent"

    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent logic and return a state update."""
        ...

    def _messages(self, state: Dict[str, Any]) -> List[BaseMessage]:
        return state.get("messages", [])

    def _append_message(self, state: Dict[str, Any], message: BaseMessage) -> List[BaseMessage]:
        msgs = self._messages(state)
        msgs = list(msgs) + [message]
        return msgs


__all__ = ["BaseAgent"]

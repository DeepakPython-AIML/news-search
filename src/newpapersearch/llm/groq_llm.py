"""
Groq LLM wrapper.

Provides a small, typed abstraction over langchain_groq.ChatGroq so the
agentic graph can request either a "reasoning" model (complex planning /
synthesis) or a "fast" model (guardrails / simple classification) with a
single call site.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_groq import ChatGroq

from newpapersearch.config import get_settings


class GroqLLM(BaseChatModel):
    """
    Thin wrapper around ChatGroq that exposes two preset models:

    * reasoning  -> llama-3.3-70b-versatile  (supervisor, synthesis, validation)
    * fast       -> llama-3.1-8b-instant     (guardrails, classification)
    """

    _llm: ChatGroq = None  # type: ignore[assignment]

    def __init__(self, mode: str = "reasoning", **kwargs: Any):
        super().__init__()
        settings = get_settings()
        model_name = (
            settings.groq_model_reasoning if mode == "reasoning" else settings.groq_model_fast
        )
        self._llm = ChatGroq(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
            model_name=model_name,
            temperature=settings.groq_temperature,
            max_tokens=settings.groq_max_tokens,
            **kwargs,
        )
        self._mode = mode

    @property
    def _llm_type(self) -> str:
        return f"groq-{self._mode}"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._llm._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await self._llm._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

    @property
    def model_name(self) -> str:
        return self._llm.model_name


def get_reasoning_llm() -> GroqLLM:
    """Return an LLM configured for complex reasoning (supervisor / synthesis)."""
    return GroqLLM(mode="reasoning")


def get_fast_llm() -> GroqLLM:
    """Return an LLM configured for fast, low-latency tasks (guardrails)."""
    return GroqLLM(mode="fast")


__all__ = ["GroqLLM", "get_reasoning_llm", "get_fast_llm"]

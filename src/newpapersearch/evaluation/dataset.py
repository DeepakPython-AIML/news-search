"""
Default evaluation dataset for the LLM-as-Judge layer.

A small, curated set of news-search queries used to exercise the RAG system
during evaluation. Each case is an :class:`EvalCase` with a query, an optional
reference (gold-standard) answer, and optional expected source filenames.

Reference answers are intentionally kept optional: the groundedness, answer
relevance, context relevance, and citation accuracy judges can all score an
answer without a gold reference, so the dataset remains useful even before
gold answers are written.

Extend this list (or load your own from JSON/CSV) to build a regression suite
for the news-search RAG system.
"""

from __future__ import annotations

from newpapersearch.evaluation.metrics import EvalCase

DEFAULT_EVAL_CASES: list[EvalCase] = [
    EvalCase(
        case_id="news-001",
        query="What were the main headlines and editorials this week?",
        top_k=5,
    ),
    EvalCase(
        case_id="news-002",
        query="Summarize the key opinions expressed in the October 2026 editorials.",
        top_k=5,
    ),
    EvalCase(
        case_id="news-003",
        query="What did the editorials say about the economy or financial markets?",
        top_k=5,
    ),
    EvalCase(
        case_id="news-004",
        query="Were there any international affairs covered in the editorials?",
        top_k=5,
    ),
    EvalCase(
        case_id="news-005",
        query="What is the editorial stance on technology and artificial intelligence?",
        top_k=5,
    ),
    EvalCase(
        case_id="news-006",
        query="Which sources discuss climate change or the environment?",
        top_k=5,
    ),
    EvalCase(
        case_id="news-007",
        query="Give me a brief summary of all editorials published between 25 and 30 October 2026.",
        top_k=5,
    ),
    EvalCase(
        case_id="news-008",
        query="Tell me about the top news story of the week.",
        top_k=5,
    ),
]


def load_default_cases() -> list[EvalCase]:
    """Return the default list of evaluation cases."""
    return list(DEFAULT_EVAL_CASES)


__all__ = ["DEFAULT_EVAL_CASES", "load_default_cases"]

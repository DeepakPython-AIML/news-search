"""
LLM-as-Judge evaluation panel.

Each judge is an LLM prompt that scores one dimension of a RAG answer
(groundedness, answer relevance, context relevance, citation accuracy) on a
0-10 scale, returning a ``JudgeScore`` plus a short justification.

Design principles
-----------------
* **Separation of concerns:** each judge answers exactly one question.
* **Fail-open:** if the judge LLM errors or returns unparseable output, the
  judge degrades to a deterministic heuristic so the evaluation never crashes.
* **Structured output with fallback:** the judge is asked for JSON; a regex
  fallback extracts a score when the model is not strict.
* **Reuses the existing Groq reasoning LLM** (``get_reasoning_llm``) so no new
  provider or API key is required.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

from newpapersearch.evaluation.metrics import (
    JudgeScore,
    MetricName,
    score_to_label,
)
from newpapersearch.llm.groq_llm import get_reasoning_llm


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
_SCORE_RE = re.compile(r"(?:\"score\"\\?\\s*:\\?\\s*(\d+(?:\.\d+)?))", re.IGNORECASE)
_FALLBACK_NUMBER_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:/10|out of 10)?\b")


class _JudgeOutput:
    """In-memory shape expected from a judge LLM (score 0-10 + reasoning)."""

    def __init__(self, score: float, reasoning: str):
        self.score = score
        self.reasoning = reasoning


def _parse_judge_output(raw: str) -> Optional[_JudgeOutput]:
    """Parse a judge's text response into a score + reasoning.

    Tries JSON first, then falls back to a regex score extraction. Returns
    None when nothing usable can be extracted.
    """
    text = (raw or "").strip()

    # 1) Try to extract a JSON object (handles markdown fences and extra text).
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(text[start : end + 1])
            score = float(obj.get("score", obj.get("Score", -1)))
            reasoning = str(obj.get("reasoning", obj.get("Reasoning", "")))
            if 0.0 <= score <= 10.0:
                return _JudgeOutput(score=score, reasoning=reasoning)
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
        pass

    # 2) Fallback: regex for a number /10.
    match = _SCORE_RE.search(text) or _FALLBACK_NUMBER_RE.search(text)
    if match:
        try:
            score = float(match.group(1))
            if 0.0 <= score <= 10.0:
                reasoning = text[:200].strip()
                return _JudgeOutput(score=score, reasoning=reasoning)
        except ValueError:
            pass

    return None


def _failover_score(reason: str) -> _JudgeOutput:
    """Neutral middle-of-the-road score used when a judge cannot run."""
    return _JudgeOutput(
        score=5.0,
        reasoning=f"Judge unavailable ({reason}); assigned neutral score.",
    )


# ---------------------------------------------------------------------------
# Base judge
# ---------------------------------------------------------------------------
class BaseJudge(ABC):
    """Abstract LLM-as-judge. Subclasses implement ``build_prompt``."""

    name: str
    metric: MetricName

    def __init__(self, llm=None):
        self.llm = llm or get_reasoning_llm()

    def judge(
        self,
        *,
        query: str,
        answer: str,
        context: str,
        citations: Optional[list[dict]] = None,
        reference_answer: Optional[str] = None,
    ) -> JudgeScore:
        """Run the judge and return a :class:`JudgeScore`.

        Fail-open: any LLM/parsing error yields a neutral score so the
        evaluation runner keeps going.
        """
        citations = citations or []
        try:
            prompt = self.build_prompt(
                query=query,
                answer=answer,
                context=context,
                citations=citations,
                reference_answer=reference_answer,
            )
            response = self.llm.invoke([("user", prompt)])
            raw = response.content or ""
            parsed = _parse_judge_output(raw)
            if parsed is None:
                return JudgeScore.from_raw(
                    self.metric,
                    5.0,
                    f"Unparseable judge output; neutral score. Raw: {raw[:120]}",
                )
            return JudgeScore.from_raw(
                self.metric,
                parsed.score,
                parsed.reasoning or "No reasoning provided.",
            )
        except Exception as exc:  # pragma: no cover - defensive fail-open
            return JudgeScore.from_raw(
                self.metric, 5.0, f"Judge error: {exc}; neutral score."
            )

    @abstractmethod
    def build_prompt(
        self,
        *,
        query: str,
        answer: str,
        context: str,
        citations: list[dict],
        reference_answer: Optional[str],
    ) -> str:
        """Build the prompt sent to the judge LLM."""


# ---------------------------------------------------------------------------
# 1. Groundedness (faithfulness) judge
# ---------------------------------------------------------------------------
GROUNDING_PROMPT_TEMPLATE = """You are a strict fact-checking judge evaluating a RAG (retrieval-augmented
generation) answer for a news-search system.

TASK: Score the GROUNDEDNESS (faithfulness) of the ANSWER against the RETRIEVED CONTEXT.
- 10 = every claim in the answer is directly supported by the context.
- 5  = some claims are supported, others are not verifiable from the context.
- 0  = the answer is largely unsupported or contradicts the context.
Do not reward fluency; reward whether the answer is backed by the sources.

USER QUERY:
{query}

RETRIEVED CONTEXT (source chunks):
{context}

ANSWER TO JUDGE:
{answer}

CITATIONS (if any):
{citations}

Reply with exactly one JSON object and nothing else:
{{
  "score": <number 0-10>,
  "reasoning": "<one or two sentences justifying the score>"
}}"""


class GroundednessJudge(BaseJudge):
    name = "groundedness_judge"
    metric = MetricName.GROUNDEDNESS

    def build_prompt(
        self,
        *,
        query: str,
        answer: str,
        context: str,
        citations: list[dict],
        reference_answer: Optional[str] = None,
    ) -> str:
        return GROUNDING_PROMPT_TEMPLATE.format(
            query=query,
            answer=answer,
            context=context or "(no context retrieved)",
            citations=self._format_citations(citations),
        )

    @staticmethod
    def _format_citations(citations: list[dict]) -> str:
        if not citations:
            return "none"
        lines = []
        for c in citations[:6]:
            lines.append(
                f"- {c.get('source_file', 'unknown')}"
                + (f" (page {c.get('page_number')})" if c.get("page_number") else "")
                + f": {c.get('snippet', '')[:150]}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Answer relevance judge
# ---------------------------------------------------------------------------
ANSWER_RELEVANCE_PROMPT_TEMPLATE = """You are a judge evaluating a RAG answer for a news-search system.

TASK: Score how well the ANSWER addresses the USER QUERY.
- 10 = directly and fully answers the question, no irrelevant content.
- 5  = partially answers or includes tangents.
- 0  = does not answer the question at all.
Do not consider whether the answer is grounded in sources (that is a separate
judge); judge relevance to the user's question only.

USER QUERY:
{query}

ANSWER TO JUDGE:
{answer}

Reply with exactly one JSON object and nothing else:
{{
  "score": <number 0-10>,
  "reasoning": "<one or two sentences justifying the score>"
}}"""


class AnswerRelevanceJudge(BaseJudge):
    name = "answer_relevance_judge"
    metric = MetricName.ANSWER_RELEVANCE

    def build_prompt(
        self,
        *,
        query: str,
        answer: str,
        context: str,
        citations: list[dict],
        reference_answer: Optional[str] = None,
    ) -> str:
        return ANSWER_RELEVANCE_PROMPT_TEMPLATE.format(query=query, answer=answer)


# ---------------------------------------------------------------------------
# 3. Context relevance judge
# ---------------------------------------------------------------------------
CONTEXT_RELEVANCE_PROMPT_TEMPLATE = """You are a judge evaluating retrieval quality for a news-search RAG system.

TASK: Score how RETRIEVED CONTEXT is relevant and useful for answering the USER QUERY.
- 10 = context contains all key information needed to answer the question.
- 5  = context is partially relevant / contains some useful but incomplete info.
- 0  = context is irrelevant or does not help answer the question at all.

USER QUERY:
{query}

RETRIEVED CONTEXT:
{context}

Reply with exactly one JSON object and nothing else:
{{
  "score": <number 0-10>,
  "reasoning": "<one or two sentences justifying the score>"
}}"""


class ContextRelevanceJudge(BaseJudge):
    name = "context_relevance_judge"
    metric = MetricName.CONTEXT_RELEVANCE

    def build_prompt(
        self,
        *,
        query: str,
        answer: str,
        context: str,
        citations: list[dict],
        reference_answer: Optional[str] = None,
    ) -> str:
        return CONTEXT_RELEVANCE_PROMPT_TEMPLATE.format(
            query=query, context=context or "(no context retrieved)"
        )


# ---------------------------------------------------------------------------
# 4. Citation accuracy judge
# ---------------------------------------------------------------------------
CITATION_ACCURACY_PROMPT_TEMPLATE = """You are a judge evaluating citation quality for a news-search RAG system.

TASK: Score whether the CITATIONS attached to the ANSWER are accurate and traceable
to the RETRIEVED CONTEXT.
- 10 = every cited claim can be traced to a cited source chunk; citations match
       source file/page correctly.
- 5  = citations mostly match sources but a few claims are uncited or loosely cited.
- 0  = citations are missing, fabricated, or do not match the sources at all.

USER QUERY:
{query}

ANSWER TO JUDGE:
{answer}

RETRIEVED CONTEXT:
{context}

CITATIONS:
{citations}

Reply with exactly one JSON object and nothing else:
{{
  "score": <number 0-10>,
  "reasoning": "<one or two sentences justifying the score>"
}}"""


class CitationAccuracyJudge(BaseJudge):
    name = "citation_accuracy_judge"
    metric = MetricName.CITATION_ACCURACY

    def build_prompt(
        self,
        *,
        query: str,
        answer: str,
        context: str,
        citations: list[dict],
        reference_answer: Optional[str] = None,
    ) -> str:
        return CITATION_ACCURACY_PROMPT_TEMPLATE.format(
            query=query,
            answer=answer,
            context=context or "(no context retrieved)",
            citations=self._format_citations(citations),
        )

    @staticmethod
    def _format_citations(citations: list[dict]) -> str:
        if not citations:
            return "none provided"
        lines = []
        for c in citations[:8]:
            lines.append(
                f"- {c.get('source_file', 'unknown')}"
                + (f" (page {c.get('page_number')})" if c.get("page_number") else "")
                + f": {c.get('snippet', '')[:150]}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Judge panel
# ---------------------------------------------------------------------------
class LLMJudgePanel:
    """A panel of LLM-as-judge scorers run against one RAG case."""

    def __init__(self, llm=None):
        self.llm = llm or get_reasoning_llm()
        self.judges: list[BaseJudge] = [
            GroundednessJudge(llm),
            AnswerRelevanceJudge(llm),
            ContextRelevanceJudge(llm),
            CitationAccuracyJudge(llm),
        ]

    def evaluate_all(
        self,
        *,
        query: str,
        answer: str,
        context: str,
        citations: Optional[list[dict]] = None,
        reference_answer: Optional[str] = None,
    ) -> dict[MetricName, JudgeScore]:
        """Run every judge and return {metric: JudgeScore}."""
        citations = citations or []
        scores: dict[MetricName, JudgeScore] = {}
        for judge in self.judges:
            score = judge.judge(
                query=query,
                answer=answer,
                context=context,
                citations=citations,
                reference_answer=reference_answer,
            )
            scores[score.metric] = score
        return scores

    @property
    def metric_names(self) -> list[MetricName]:
        return [j.metric for j in self.judges]


__all__ = [
    "BaseJudge",
    "GroundednessJudge",
    "AnswerRelevanceJudge",
    "ContextRelevanceJudge",
    "CitationAccuracyJudge",
    "LLMJudgePanel",
    "_parse_judge_output",
    "_failover_score",
]

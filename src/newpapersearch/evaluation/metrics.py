"""
Evaluation metrics and result models for the LLM-as-Judge layer.

Defines the shared data contracts used by the judges, the evaluation runner,
and the report renderer:

* ``JudgeScore``   - a single judge's verdict (score 0-10, label, reasoning)
* ``EvalCase``     - one test case (query + optional reference answer)
* ``EvalResult``   - per-case result with all judge scores + an overall score
* ``EvalReport``   - aggregated report across many cases

Scores are on a 0-10 scale mapped to labels:
    9-10 excellent | 7-8 good | 5-6 fair | 0-4 poor
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


class MetricName(str, Enum):
    """Names of the LLM-as-judge metrics."""

    GROUNDEDNESS = "groundedness"  # answer supported by retrieved context (faithfulness)
    ANSWER_RELEVANCE = "answer_relevance"  # answer addresses the user query
    CONTEXT_RELEVANCE = "context_relevance"  # retrieved context is useful for the query
    CITATION_ACCURACY = "citation_accuracy"  # citations trace back to the context
    OVERALL = "overall"  # weighted aggregate of the above


# Weights used to compute the overall score from the four judge metrics.
# Groundedness is weighted most heavily because correctness/safety matter most
# for a news-search RAG system.
METRIC_WEIGHTS: dict[MetricName, float] = {
    MetricName.GROUNDEDNESS: 0.40,
    MetricName.ANSWER_RELEVANCE: 0.30,
    MetricName.CONTEXT_RELEVANCE: 0.20,
    MetricName.CITATION_ACCURACY: 0.10,
}


def score_to_label(score: float) -> str:
    """Map a 0-10 score to a human-readable label."""
    if score >= 9.0:
        return "excellent"
    if score >= 7.0:
        return "good"
    if score >= 5.0:
        return "fair"
    return "poor"


class JudgeScore(BaseModel):
    """A single LLM-as-judge verdict."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metric: MetricName
    score: float = Field(..., ge=0.0, le=10.0, description="Judge score on a 0-10 scale")
    label: str = Field(..., description="excellent / good / fair / poor")
    reasoning: str = Field(..., description="Short justification from the judge LLM")

    @classmethod
    def from_raw(cls, metric: MetricName, score: float, reasoning: str) -> "JudgeScore":
        score = max(0.0, min(10.0, float(score)))
        return cls(
            metric=metric,
            score=score,
            label=score_to_label(score),
            reasoning=reasoning,
        )


class EvalCase(BaseModel):
    """One evaluation test case.

    ``reference_answer`` is optional: groundedness, answer relevance,
    context relevance and citation accuracy can all be judged without a
    reference, so cases remain useful even when gold answers are unknown.
    """

    case_id: str = Field(default_factory=lambda: str(uuid4()))
    query: str = Field(..., min_length=1, description="User question to evaluate")
    reference_answer: Optional[str] = Field(
        None, description="Optional gold-standard answer (used by reference-based checks)"
    )
    expected_sources: list[str] = Field(
        default_factory=list, description="Optional expected source filenames"
    )
    top_k: int = Field(5, ge=1, le=20, description="Retrieval top_k for this case")


class EvalResult(BaseModel):
    """Result of evaluating a single test case."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    case_id: str
    query: str
    answer: str
    context: str = Field("", description="Concatenated retrieved context used by the judge")
    citations: list[dict[str, Any]] = Field(default_factory=list)
    judges: dict[MetricName, JudgeScore] = Field(default_factory=dict)
    overall_score: float = Field(0.0, ge=0.0, le=10.0)
    overall_label: str = ""
    latency_ms: float = 0.0
    agents_used: list[str] = Field(default_factory=list)
    guardrail_passed: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def judge_score(self, metric: MetricName) -> Optional[JudgeScore]:
        return self.judges.get(metric)


class EvalReport(BaseModel):
    """Aggregated evaluation report across many test cases."""

    report_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    total_cases: int = 0
    average_scores: dict[MetricName, float] = Field(default_factory=dict)
    average_overall: float = 0.0
    results: list[EvalResult] = Field(default_factory=list)
    pass_rate: float = Field(
        0.0, description="Fraction of cases with overall score >= 7.0 (good+)"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


__all__ = [
    "MetricName",
    "METRIC_WEIGHTS",
    "score_to_label",
    "JudgeScore",
    "EvalCase",
    "EvalResult",
    "EvalReport",
]

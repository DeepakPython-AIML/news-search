"""
Tests for the LLM-as-Judge evaluation layer.

These tests avoid live Groq calls by mocking the LLM and the RAG graph, so
they run offline and deterministically. A live Groq-backed test is included
but skipped unless ``GROQ_API_KEY`` is set.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest

from langchain_core.documents import Document

from newpapersearch.evaluation.dataset import load_default_cases
from newpapersearch.evaluation.judges import LLMJudgePanel, _parse_judge_output
from newpapersearch.evaluation.metrics import (
    EvalCase,
    EvalReport,
    EvalResult,
    JudgeScore,
    MetricName,
    METRIC_WEIGHTS,
    score_to_label,
)
from newpapersearch.evaluation.runner import EvaluationRunner


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def test_score_to_label_bounds():
    assert score_to_label(10.0) == "excellent"
    assert score_to_label(9.0) == "excellent"
    assert score_to_label(7.5) == "good"
    assert score_to_label(7.0) == "good"
    assert score_to_label(5.5) == "fair"
    assert score_to_label(5.0) == "fair"
    assert score_to_label(2.0) == "poor"
    assert score_to_label(0.0) == "poor"


def test_metric_weights_sum_to_one():
    assert abs(sum(METRIC_WEIGHTS.values()) - 1.0) < 1e-9


def test_judge_score_from_raw_clamps():
    s = JudgeScore.from_raw(MetricName.GROUNDEDNESS, 12.0, "too high")
    assert s.score == 10.0
    s = JudgeScore.from_raw(MetricName.GROUNDEDNESS, -3.0, "too low")
    assert s.score == 0.0
    assert s.metric == MetricName.GROUNDEDNESS


# --------------------------------------------------------------------------- #
# Judge output parsing
# --------------------------------------------------------------------------- #
def test_parse_judge_output_json():
    out = _parse_judge_output('{"score": 8, "reasoning": "well supported"}')
    assert out is not None
    assert out.score == 8.0
    assert "well supported" in out.reasoning


def test_parse_judge_output_markdown_fence():
    raw = "```json\n{\"score\": 6.5, \"reasoning\": \"partial\"}\n```"
    out = _parse_judge_output(raw)
    assert out is not None
    assert out.score == 6.5


def test_parse_judge_output_fallback_number():
    out = _parse_judge_output("I would give this a 7 / 10.")
    assert out is not None
    assert out.score == 7.0


def test_parse_judge_output_garbage():
    assert _parse_judge_output("nothing useful here") is None
    assert _parse_judge_output("") is None


# --------------------------------------------------------------------------- #
# Judge fail-open with a fake LLM
# --------------------------------------------------------------------------- #
class FakeLLM:
    """Stand-in for GroqLLM that returns canned responses."""

    def __init__(self, responses: dict[str, str]):
        self.responses = responses
        self.calls: list[str] = []

    def invoke(self, messages, *args, **kwargs):
        prompt = messages[0][1] if messages else ""
        self.calls.append(prompt)
        for key, resp in self.responses.items():
            if key.lower() in prompt.lower():
                return SimpleNamespace(content=resp)
        return SimpleNamespace(content="5")
def test_groundedness_judge_scores_supported_answer():
    llm = FakeLLM({"groundedness": '{"score": 9, "reasoning": "fully supported"}'})
    from newpapersearch.evaluation.judges import GroundednessJudge

    judge = GroundednessJudge(llm=llm)
    score = judge.judge(
        query="What happened?",
        answer="The market closed up.",
        context="The market closed up on Friday.",
        citations=[{"source_file": "a.pdf", "page_number": 1, "snippet": "..."}],
    )
    assert score.metric == MetricName.GROUNDEDNESS
    assert score.score == 9.0
    assert score.label == "excellent"


def test_judge_fail_open_on_llm_error():
    class BrokenLLM:
        def invoke(self, messages, *args, **kwargs):
            raise RuntimeError("groq down")

    from newpapersearch.evaluation.judges import AnswerRelevanceJudge

    judge = AnswerRelevanceJudge(llm=BrokenLLM())
    score = judge.judge(query="q", answer="a", context="c")
    assert score.score == 5.0
    assert "error" in score.reasoning.lower() or "neutral" in score.reasoning.lower()


def test_panel_evaluate_all_returns_all_metrics():
    llm = FakeLLM({"score": '{"score": 7, "reasoning": "ok"}'})
    panel = LLMJudgePanel(llm=llm)
    scores = panel.evaluate_all(
        query="q", answer="a", context="c", citations=[], reference_answer=None
    )
    assert set(scores.keys()) == {
        MetricName.GROUNDEDNESS,
        MetricName.ANSWER_RELEVANCE,
        MetricName.CONTEXT_RELEVANCE,
        MetricName.CITATION_ACCURACY,
    }
    assert all(isinstance(s, JudgeScore) for s in scores.values())


# --------------------------------------------------------------------------- #
# Runner with mocked graph + panel (offline)
# --------------------------------------------------------------------------- #
def _fake_state() -> dict[str, Any]:
    docs = [
        Document(
            page_content="The market closed up on Friday amid strong earnings.",
            metadata={"source_file": "editorial_29.pdf", "page_number": 4},
        )
    ]
    return {
        "final_answer": "The market closed up on Friday amid strong earnings.",
        "retrieved_docs_with_score": [(docs[0], 0.2)],
        "citations": [
            SimpleNamespace(
                source_file="editorial_29.pdf",
                page_number=4,
                snippet="The market closed up on Friday amid strong earnings.",
                relevance_score=0.8,
            )
        ],
        "agents_used": ["retrieval", "synthesis"],
        "guardrail_passed": True,
    }
def test_runner_run_case_offline(monkeypatch):
    """run_case should not call Groq when graph and panel are mocked."""
    from newpapersearch.evaluation import runner as runner_mod

    monkeypatch.setattr(runner_mod, "invoke_rag", lambda graph, **kw: _fake_state())

    fake_panel = SimpleNamespace(
        evaluate_all=lambda **kw: {
            MetricName.GROUNDEDNESS: JudgeScore.from_raw(
                MetricName.GROUNDEDNESS, 8.0, "supported"
            ),
            MetricName.ANSWER_RELEVANCE: JudgeScore.from_raw(
                MetricName.ANSWER_RELEVANCE, 9.0, "relevant"
            ),
            MetricName.CONTEXT_RELEVANCE: JudgeScore.from_raw(
                MetricName.CONTEXT_RELEVANCE, 7.0, "useful"
            ),
            MetricName.CITATION_ACCURACY: JudgeScore.from_raw(
                MetricName.CITATION_ACCURACY, 6.0, "ok"
            ),
        }
    )

    runner = EvaluationRunner(graph=None, panel=fake_panel, top_k=5)
    case = EvalCase(case_id="t-1", query="What happened on Friday?")
    result = runner.run_case(case)

    assert result.case_id == "t-1"
    assert result.query == "What happened on Friday?"
    expected = 0.4 * 8 + 0.3 * 9 + 0.2 * 7 + 0.1 * 6
    assert result.overall_score == pytest.approx(expected)
    assert result.overall_label == "good"
    assert result.guardrail_passed is True
    assert len(result.citations) == 1


def test_runner_run_case_graph_failure_is_recorded(monkeypatch):
    """A graph failure yields a zero-score result rather than crashing."""
    from newpapersearch.evaluation import runner as runner_mod

    def _boom(graph, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(runner_mod, "invoke_rag", _boom)

    runner = EvaluationRunner(graph=None, panel=LLMJudgePanel(), top_k=5)
    case = EvalCase(case_id="t-2", query="q")
    result = runner.run_case(case)
    assert result.overall_score == 0.0
    assert result.overall_label == "poor"
    assert "EVALUATION ERROR" in result.answer


def test_runner_run_aggregates_report(monkeypatch):
    fake_panel = SimpleNamespace(
        evaluate_all=lambda **kw: {
            MetricName.GROUNDEDNESS: JudgeScore.from_raw(MetricName.GROUNDEDNESS, 8.0, "x"),
            MetricName.ANSWER_RELEVANCE: JudgeScore.from_raw(
                MetricName.ANSWER_RELEVANCE, 8.0, "x"
            ),
            MetricName.CONTEXT_RELEVANCE: JudgeScore.from_raw(
                MetricName.CONTEXT_RELEVANCE, 8.0, "x"
            ),
            MetricName.CITATION_ACCURACY: JudgeScore.from_raw(
                MetricName.CITATION_ACCURACY, 8.0, "x"
            ),
        }
    )
    from newpapersearch.evaluation import runner as runner_mod

    monkeypatch.setattr(runner_mod, "invoke_rag", lambda graph, **kw: _fake_state())

    runner = EvaluationRunner(graph=None, panel=fake_panel, top_k=5)
    cases = [EvalCase(case_id=f"c{i}", query=f"q{i}") for i in range(3)]
    report = runner.run(cases)

    assert isinstance(report, EvalReport)
    assert report.total_cases == 3
    assert report.average_overall == pytest.approx(8.0)
    assert report.pass_rate == 1.0


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def test_default_dataset_non_empty():
    cases = load_default_cases()
    assert len(cases) > 0
    assert all(c.query for c in cases)


# --------------------------------------------------------------------------- #
# Live Groq-backed test (skipped without API key)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    "GROQ_API_KEY" not in os.environ,
    reason="Requires GROQ_API_KEY for a live judge call",
)
def test_live_judge_returns_valid_score():
    from newpapersearch.evaluation.judges import GroundednessJudge
    from newpapersearch.llm.groq_llm import get_reasoning_llm

    judge = GroundednessJudge(llm=get_reasoning_llm())
    score = judge.judge(
        query="What was the headline?",
        answer="The market closed up on Friday.",
        context="The market closed up on Friday amid strong earnings.",
        citations=[],
    )
    assert 0.0 <= score.score <= 10.0
    assert score.reasoning



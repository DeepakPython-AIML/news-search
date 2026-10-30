"""
Evaluation runner.

The runner executes the RAG graph for each evaluation case, collects the
answer, retrieved context, citations and latency, runs the LLM-as-judge panel,
computes a weighted overall score, and aggregates everything into an
:class:`EvalReport`.

Usage (programmatic):
    from newpapersearch.agents.graph import build_graph
    from newpapersearch.evaluation.dataset import load_default_cases
    from newpapersearch.evaluation.runner import EvaluationRunner

    graph = build_graph()
    runner = EvaluationRunner(graph=graph)
    report = runner.run()
    print(report.average_overall)

Or via the CLI:  uv run newpapersearch evaluate
"""

from __future__ import annotations

import time
from typing import Any, Optional

from newpapersearch.agents.graph import invoke_rag
from newpapersearch.config import get_settings
from newpapersearch.evaluation.dataset import load_default_cases
from newpapersearch.evaluation.judges import LLMJudgePanel
from newpapersearch.evaluation.metrics import (
    EvalCase,
    EvalReport,
    EvalResult,
    MetricName,
    METRIC_WEIGHTS,
    score_to_label,
)

# Number of top chunks to concatenate into the judge context (keeps judge
# prompts within context limits while still representing the evidence).
JUDGE_CONTEXT_CHUNKS = 6


class EvaluationRunner:
    """
    Runs the full evaluation pipeline: RAG inference -> LLM judging -> report.

    Parameters
    ----------
    graph:
        Compiled LangGraph (optional). If None, a fresh graph is built via
        ``build_graph()``.
    panel:
        LLM-as-judge panel (optional). If None, a default panel using the
        Groq reasoning LLM is created.
    top_k:
        Default retrieval top_k applied when a case does not specify one.
    """

    def __init__(
        self,
        graph: Optional[Any] = None,
        panel: Optional[LLMJudgePanel] = None,
        top_k: int = 5,
    ):
        self.graph = graph
        self.settings = get_settings()
        self.panel = panel or LLMJudgePanel()
        self.default_top_k = top_k

    # ------------------------------------------------------------------ #
    # Single-case evaluation
    # ------------------------------------------------------------------ #
    def run_case(self, case: EvalCase) -> EvalResult:
        """Evaluate one test case end to end.

        Runs the RAG graph, builds the judge context from retrieved chunks,
        runs the judge panel, and computes the weighted overall score.
        """
        top_k = case.top_k or self.default_top_k
        settings = get_settings()

        start = time.time()
        try:
            state = invoke_rag(
                self.graph or _build_graph(),
                query=case.query,
                session_id=None,
                top_k=top_k,
                use_guardrails=settings.enable_input_guardrails,
            )
        except Exception as exc:  # pragma: no cover - defensive
            # Record a failed case rather than crashing the whole run.
            return EvalResult(
                case_id=case.case_id,
                query=case.query,
                answer=f"[EVALUATION ERROR] RAG invocation failed: {exc}",
                judges={},
                overall_score=0.0,
                overall_label="poor",
                latency_ms=(time.time() - start) * 1000,
            )
        latency_ms = (time.time() - start) * 1000

        answer = state.get("final_answer", "") or ""
        retrieved_docs_with_score = state.get("retrieved_docs_with_score") or []

        # Build judge context from the top chunks only (bounded size).
        context_parts: list[str] = []
        for doc, _score in retrieved_docs_with_score[:JUDGE_CONTEXT_CHUNKS]:
            src = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page_number")
            tag = f"{src}" + (f" (page {page})" if page is not None else "")
            context_parts.append(f"[{tag}]\n{doc.page_content}")
        context = "\n\n".join(context_parts) if context_parts else "(no context retrieved)"

        # Citations as plain dicts (JSON-serializable) for the judges + report.
        citations: list[dict[str, Any]] = []
        for c in state.get("citations") or []:
            if hasattr(c, "source_file"):
                citations.append(
                    {
                        "source_file": c.source_file,
                        "page_number": c.page_number,
                        "snippet": c.snippet,
                        "relevance_score": c.relevance_score,
                    }
                )
            else:
                citations.append(c)

        # Run the LLM-as-judge panel.
        judge_scores = self.panel.evaluate_all(
            query=case.query,
            answer=answer,
            context=context,
            citations=citations,
            reference_answer=case.reference_answer,
        )

        # Weighted overall score (skip missing judges defensively).
        total_weight = 0.0
        weighted_sum = 0.0
        for metric, score in judge_scores.items():
            weight = METRIC_WEIGHTS.get(metric, 0.0)
            weighted_sum += score.score * weight
            total_weight += weight
        overall_score = weighted_sum / total_weight if total_weight else 0.0

        result = EvalResult(
            case_id=case.case_id,
            query=case.query,
            answer=answer,
            context=context,
            citations=citations,
            judges=judge_scores,
            overall_score=overall_score,
            overall_label=score_to_label(overall_score),
            latency_ms=round(latency_ms, 2),
            agents_used=list(state.get("agents_used") or []),
            guardrail_passed=bool(state.get("guardrail_passed", True)),
        )
        return result

    # ------------------------------------------------------------------ #
    # Full-suite evaluation
    # ------------------------------------------------------------------ #
    def run(self, cases: Optional[list[EvalCase]] = None) -> EvalReport:
        """Evaluate a list of cases (defaults to the built-in dataset)."""
        cases = cases or load_default_cases()
        results = [self.run_case(case) for case in cases]

        report = EvalReport(total_cases=len(results), results=results)
        if results:
            # Average per metric across cases (only metrics that were scored).
            metric_sums: dict[MetricName, float] = {}
            metric_counts: dict[MetricName, int] = {}
            for result in results:
                for metric, score in result.judges.items():
                    metric_sums[metric] = metric_sums.get(metric, 0.0) + score.score
                    metric_counts[metric] = metric_counts.get(metric, 0) + 1
            report.average_scores = {
                metric: metric_sums[metric] / metric_counts[metric]
                for metric in metric_sums
            }
            report.average_overall = (
                sum(r.overall_score for r in results) / len(results)
            )
            good = sum(1 for r in results if r.overall_score >= 7.0)
            report.pass_rate = good / len(results)
        return report


def _build_graph():
    """Lazy import to build a graph when none is supplied (avoids import cycles)."""
    from newpapersearch.agents.graph import build_graph

    return build_graph()


__all__ = ["EvaluationRunner", "JUDGE_CONTEXT_CHUNKS"]

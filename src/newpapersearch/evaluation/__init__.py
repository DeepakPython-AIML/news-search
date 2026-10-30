"""
LLM-as-Judge evaluation package for the Agentic AI RAG system.

Provides:
* ``metrics``  - data models (EvalCase, EvalResult, EvalReport, JudgeScore)
* ``judges``   - LLM-as-judge panel (groundedness, relevance, context, citations)
* ``dataset``  - default evaluation dataset
* ``runner``   - EvaluationRunner: RAG inference -> judging -> report
* ``reporter`` - console / JSON / CSV report rendering
"""

from newpapersearch.evaluation.dataset import load_default_cases
from newpapersearch.evaluation.judges import LLMJudgePanel
from newpapersearch.evaluation.metrics import (
    EvalCase,
    EvalReport,
    EvalResult,
    JudgeScore,
    MetricName,
)
from newpapersearch.evaluation.reporter import print_report, write_json_report, write_csv_report
from newpapersearch.evaluation.runner import EvaluationRunner

__all__ = [
    "EvalCase",
    "EvalReport",
    "EvalResult",
    "JudgeScore",
    "MetricName",
    "LLMJudgePanel",
    "EvaluationRunner",
    "load_default_cases",
    "print_report",
    "write_json_report",
    "write_csv_report",
]

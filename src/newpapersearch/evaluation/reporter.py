"""
Evaluation report rendering.

Renders an :class:`EvalReport` to the console (human-readable table) and to
JSON/CSV files. No third-party dependencies are used so the evaluation output
is easy to capture in CI logs or dashboards.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from newpapersearch.evaluation.metrics import EvalReport, MetricName

# Order in which metrics are displayed.
DISPLAY_ORDER = [
    MetricName.GROUNDEDNESS,
    MetricName.ANSWER_RELEVANCE,
    MetricName.CONTEXT_RELEVANCE,
    MetricName.CITATION_ACCURACY,
    MetricName.OVERALL,
]


def _metric_label(metric: MetricName) -> str:
    return {
        MetricName.GROUNDEDNESS: "Groundedness",
        MetricName.ANSWER_RELEVANCE: "Answer Relevance",
        MetricName.CONTEXT_RELEVANCE: "Context Relevance",
        MetricName.CITATION_ACCURACY: "Citation Accuracy",
        MetricName.OVERALL: "Overall",
    }.get(metric, metric.value)


def _bar(score: float, width: int = 20) -> str:
    """Simple ASCII bar for a 0-10 score."""
    filled = int(round(score / 10.0 * width))
    filled = max(0, min(width, filled))
    return "#" * filled + "-" * (width - filled)


def print_report(report: EvalReport) -> None:
    """Print a human-readable evaluation report to stdout."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(" LLM-as-Judge Evaluation Report")
    lines.append(f" Report ID : {report.report_id}")
    lines.append(f" Created   : {report.created_at.isoformat(timespec='seconds')}")
    lines.append(f" Cases     : {report.total_cases}")
    lines.append("=" * 72)

    # Header
    header = f" {'Metric':<22} | {'Avg':>5} | {'Min':>5} | {'Max':>5} | Progress"
    lines.append(header)
    lines.append("-" * len(header))

    for metric in DISPLAY_ORDER:
        avg = report.average_scores.get(metric)
        if avg is None:
            continue
        scored = [r for r in report.results if metric in r.judges]
        scores = [r.judges[metric].score for r in scored]
        mn = min(scores) if scores else 0.0
        mx = max(scores) if scores else 0.0
        label = _metric_label(metric)
        if metric == MetricName.OVERALL:
            avg = report.average_overall
            mn = min((r.overall_score for r in report.results), default=0.0)
            mx = max((r.overall_score for r in report.results), default=0.0)
        lines.append(
            f" {label:<22} | {avg:>5.2f} | {mn:>5.2f} | {mx:>5.2f} | {_bar(avg)}"
        )

    lines.append("-" * len(header))
    lines.append(f" Pass rate (overall >= 7.0): {report.pass_rate * 100:.1f}%")
    lines.append(f" Average overall score      : {report.average_overall:.2f} / 10")
    lines.append("=" * 72)

    # Per-case detail
    lines.append("\nPer-case results:")
    lines.append(f" {'Case':<8} | {'Query':<40} | {'Overall':>8} | {'Label':<10} | {'Latency':>8}")
    lines.append("-" * 82)
    for r in report.results:
        query = r.query[:37] + "..." if len(r.query) > 40 else r.query
        lines.append(
            f" {r.case_id[:8]:<8} | {query:<40} | {r.overall_score:>8.2f} | "
            f"{r.overall_label:<10} | {r.latency_ms:>7.0f} ms"
        )
    lines.append("=" * 72)
    print("\n".join(lines))


def write_json_report(report: EvalReport, path: str | Path) -> None:
    """Write the full report as JSON to ``path``."""
    data = report.to_dict()
    # Flatten judge scores for easier downstream consumption.
    for case in data.get("results", []):
        judges = case.pop("judges", {})
        flat = {}
        for metric_name, score in judges.items():
            flat[metric_name] = {
                "score": score["score"],
                "label": score["label"],
                "reasoning": score["reasoning"],
            }
        case["judge_scores"] = flat
        case["overall_score"] = round(case.get("overall_score", 0.0), 2)
    Path(path).write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv_report(report: EvalReport, path: str | Path) -> None:
    """Write per-case results as CSV to ``path``."""
    rows: list[dict[str, Any]] = []
    for r in report.results:
        row: dict[str, Any] = {
            "case_id": r.case_id,
            "query": r.query,
            "overall_score": round(r.overall_score, 2),
            "overall_label": r.overall_label,
            "latency_ms": r.latency_ms,
            "guardrail_passed": r.guardrail_passed,
        }
        for metric in DISPLAY_ORDER:
            score = r.judges.get(metric)
            if score is not None:
                row[f"{metric.value}_score"] = round(score.score, 2)
                row[f"{metric.value}_label"] = score.label
        rows.append(row)

    fieldnames = [
        "case_id", "query", "overall_score", "overall_label",
        "groundedness_score", "groundedness_label",
        "answer_relevance_score", "answer_relevance_label",
        "context_relevance_score", "context_relevance_label",
        "citation_accuracy_score", "citation_accuracy_label",
        "latency_ms", "guardrail_passed",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


__all__ = ["print_report", "write_json_report", "write_csv_report", "DISPLAY_ORDER"]

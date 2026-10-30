"""
CLI entry point.

`uv run newpapersearch`  -> prints a short help banner.
"""

from __future__ import annotations

import argparse

from newpapersearch.ingest.pipeline import ingest_news_pdfs
from newpapersearch.config import get_settings


def main():
    parser = argparse.ArgumentParser(description="Agentic AI RAG - News Search")
    sub = parser.add_subparsers(dest="command")

    p_ingest = sub.add_parser("ingest", help="Ingest news PDFs into ChromaDB")
    p_ingest.add_argument("--reset", action="store_true", help="Wipe collection before indexing")

    p_eval = sub.add_parser(
        "evaluate", help="Run LLM-as-judge evaluation over the RAG system"
    )
    p_eval.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N cases from the default dataset",
    )
    p_eval.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Retrieval top_k used for every evaluation case",
    )
    p_eval.add_argument(
        "--output",
        default=None,
        help="Write the full JSON report to this file (e.g. eval_report.json)",
    )
    p_eval.add_argument(
        "--csv",
        default=None,
        help="Also write per-case results as CSV to this file",
    )

    args = parser.parse_args()

    if args.command == "ingest":
        result = ingest_news_pdfs(reset_collection=args.reset)
        print(result)
    elif args.command == "evaluate":
        _run_evaluate(args)
    else:
        parser.print_help()
        settings = get_settings()
        print(f"\nGroq model (reasoning): {settings.groq_model_reasoning}")
        print(f"Embedding model: {settings.embedding_model_name}")


def _run_evaluate(args) -> None:
    """Run the LLM-as-judge evaluation and print/write the report."""
    from newpapersearch.agents.graph import build_graph
    from newpapersearch.evaluation.dataset import load_default_cases
    from newpapersearch.evaluation.reporter import print_report, write_csv_report, write_json_report
    from newpapersearch.evaluation.runner import EvaluationRunner

    cases = load_default_cases()
    if args.limit is not None:
        cases = cases[: args.limit]

    graph = build_graph()
    runner = EvaluationRunner(graph=graph, top_k=args.top_k)
    report = runner.run(cases)

    print_report(report)

    if args.output:
        write_json_report(report, args.output)
        print(f"\nJSON report written to: {args.output}")
    if args.csv:
        write_csv_report(report, args.csv)
        print(f"CSV report written to: {args.csv}")


if __name__ == "__main__":
    main()

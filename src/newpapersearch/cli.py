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

    args = parser.parse_args()

    if args.command == "ingest":
        result = ingest_news_pdfs(reset_collection=args.reset)
        print(result)
    else:
        parser.print_help()
        settings = get_settings()
        print(f"\nGroq model (reasoning): {settings.groq_model_reasoning}")
        print(f"Embedding model: {settings.embedding_model_name}")


if __name__ == "__main__":
    main()

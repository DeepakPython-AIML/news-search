"""
Standalone ingestion script.

Loads all PDFs from the news folder, chunks them, embeds with the configured
embedding provider and indexes them into ChromaDB.

Run with:  uv run python scripts/ingest_pdfs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script from the repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from newpapersearch.ingest.pipeline import ingest_news_pdfs  # noqa: E402


def main():
    print("📥 Starting PDF ingestion pipeline...")
    result = ingest_news_pdfs(reset_collection=True)

    print(f"Status:      {result.get('status')}")
    print(f"Documents:   {result.get('documents_processed')}")
    print(f"Chunks:      {result.get('chunks_created')} created, {result.get('chunks_indexed')} indexed")
    if result.get("errors"):
        print("Errors:")
        for err in result["errors"]:
            print(f"  - {err}")

    if result["status"] == "success":
        print("✅ Ingestion complete.")
    else:
        print("❌ Ingestion finished with errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()

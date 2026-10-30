"""
PDF loading and text extraction.

Uses pypdf to extract text page-by-page from every PDF in the news folder.
Each page is returned as a LangChain Document with source metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from pypdf import PdfReader

from newpapersearch.config import NEWS_DIR


def load_pdf_pages(pdf_path: Path) -> List[Document]:
    """
    Extract text from a single PDF file.

    Returns one Document per page. Metadata includes source_file,
    page_number (1-indexed) and source type.
    """
    reader = PdfReader(str(pdf_path))
    documents: List[Document] = []
    source_file = pdf_path.name

    for page_index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # pragma: no cover - defensive
            text = ""
        if not text.strip():
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source_file": source_file,
                    "page_number": page_index,
                    "source": "pdf",
                    "pdf_path": str(pdf_path),
                },
            )
        )
    return documents


def load_all_pdfs(news_dir: Path | None = None) -> List[Document]:
    """
    Load every PDF in the news directory.

    Returns a flat list of page Documents across all PDFs.
    """
    directory = news_dir or NEWS_DIR
    pdf_files = sorted(directory.glob("*.pdf"))
    all_documents: List[Document] = []
    for pdf_path in pdf_files:
        all_documents.extend(load_pdf_pages(pdf_path))
    return all_documents


__all__ = ["load_pdf_pages", "load_all_pdfs"]

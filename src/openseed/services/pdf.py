"""PDF text extraction."""

from __future__ import annotations

from PyPDF2 import PdfReader


def extract_text(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_pages(pdf_path: str) -> list[dict[str, object]]:
    """Extract text page-by-page with page numbers."""
    reader = PdfReader(pdf_path)
    return [
        {"page": i + 1, "text": page.extract_text() or ""}
        for i, page in enumerate(reader.pages)
    ]

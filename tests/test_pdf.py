"""Tests for openseed.services.pdf — extraction and markdown conversion."""

from __future__ import annotations

from pathlib import Path

import pytest

from openseed.services.pdf import extract_text, extract_text_pages, pdf_to_markdown, save_markdown


@pytest.fixture
def minimal_pdf(tmp_path: Path) -> str:
    """Create a minimal one-page PDF with text content using fitz."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "Abstract")
    page.insert_text((72, 130), "This paper introduces a new method for testing.")
    page.insert_text((72, 200), "1. Introduction")
    page.insert_text((72, 230), "We present our approach here.")
    pdf_path = str(tmp_path / "test.pdf")
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def blank_pdf(tmp_path: Path) -> str:
    """Create a PDF with one blank page (no text content)."""
    import fitz

    doc = fitz.open()
    doc.new_page(width=612, height=792)
    pdf_path = str(tmp_path / "blank.pdf")
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def corrupt_pdf(tmp_path: Path) -> str:
    """Write random bytes that are not a valid PDF."""
    corrupt_path = tmp_path / "corrupt.pdf"
    corrupt_path.write_bytes(b"\x00\x01\x02garbage bytes not a pdf at all!!!")
    return str(corrupt_path)


class TestExtractText:
    def test_extracts_text_from_valid_pdf(self, minimal_pdf: str) -> None:
        text = extract_text(minimal_pdf)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        missing = str(tmp_path / "nonexistent.pdf")
        with pytest.raises(Exception):
            extract_text(missing)

    def test_corrupt_pdf_raises(self, corrupt_pdf: str) -> None:
        with pytest.raises(Exception):
            extract_text(corrupt_pdf)

    def test_blank_page_pdf_returns_string(self, blank_pdf: str) -> None:
        result = extract_text(blank_pdf)
        assert isinstance(result, str)


class TestExtractTextPages:
    def test_returns_list_of_page_dicts(self, minimal_pdf: str) -> None:
        pages = extract_text_pages(minimal_pdf)
        assert isinstance(pages, list)
        assert len(pages) >= 1
        assert "page" in pages[0]
        assert "text" in pages[0]

    def test_page_numbers_start_at_one(self, minimal_pdf: str) -> None:
        pages = extract_text_pages(minimal_pdf)
        assert pages[0]["page"] == 1

    def test_blank_page_pdf_returns_list_with_entry(self, blank_pdf: str) -> None:
        pages = extract_text_pages(blank_pdf)
        assert isinstance(pages, list)
        assert len(pages) >= 1

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        missing = str(tmp_path / "ghost.pdf")
        with pytest.raises(Exception):
            extract_text_pages(missing)


class TestPdfToMarkdown:
    def test_returns_string_for_valid_pdf(self, minimal_pdf: str) -> None:
        result = pdf_to_markdown(minimal_pdf)
        assert isinstance(result, str)

    def test_blank_page_pdf_returns_string(self, blank_pdf: str) -> None:
        result = pdf_to_markdown(blank_pdf)
        assert isinstance(result, str)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        missing = str(tmp_path / "missing.pdf")
        with pytest.raises(Exception):
            pdf_to_markdown(missing)

    def test_corrupt_pdf_raises(self, corrupt_pdf: str) -> None:
        with pytest.raises(Exception):
            pdf_to_markdown(corrupt_pdf)


class TestSaveMarkdown:
    def test_saves_file_alongside_pdf(self, minimal_pdf: str) -> None:
        md_content = "# Title\n\nSome content."
        md_path = save_markdown(minimal_pdf, md_content)
        assert md_path.endswith(".md")
        assert Path(md_path).exists()
        assert Path(md_path).read_text(encoding="utf-8") == md_content

    def test_replaces_pdf_extension(self, minimal_pdf: str) -> None:
        md_path = save_markdown(minimal_pdf, "content")
        assert not md_path.endswith(".pdf")
        assert md_path.endswith(".md")

    def test_returns_path_string(self, minimal_pdf: str) -> None:
        result = save_markdown(minimal_pdf, "test")
        assert isinstance(result, str)

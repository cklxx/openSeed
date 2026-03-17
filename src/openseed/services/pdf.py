"""PDF text extraction and Markdown conversion."""

from __future__ import annotations

import re
from pathlib import Path


def extract_text(pdf_path: str) -> str:
    """Extract all text from a PDF file using pymupdf for better quality."""
    import fitz  # pymupdf

    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n\n".join(pages)


def extract_text_pages(pdf_path: str) -> list[dict[str, object]]:
    """Extract text page-by-page with page numbers."""
    import fitz  # pymupdf

    doc = fitz.open(pdf_path)
    result = []
    for i, page in enumerate(doc):
        result.append({"page": i + 1, "text": page.get_text("text") or ""})
    doc.close()
    return result


def _get_blocks_with_fonts(pdf_path: str) -> list[dict]:
    """Return text blocks with font size info from all pages."""
    import fitz  # pymupdf

    doc = fitz.open(pdf_path)
    all_blocks = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") != 0:  # 0 = text block
                continue
            lines = block.get("lines", [])
            for line in lines:
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                max_size = max(s["size"] for s in spans)
                all_blocks.append(
                    {
                        "text": text,
                        "size": max_size,
                        "page": page_num + 1,
                        "bbox": block["bbox"],
                    }
                )
    doc.close()
    return all_blocks


def _is_page_number(text: str) -> bool:
    """Return True if this line looks like a page number or running header/footer."""
    stripped = text.strip()
    # Pure digit(s)
    if re.fullmatch(r"\d+", stripped):
        return True
    # "Page N" or "- N -"
    if re.fullmatch(r"[-–—]?\s*\d+\s*[-–—]?", stripped):
        return True
    return False


def pdf_to_markdown(pdf_path: str) -> str:
    """Convert a PDF to structured Markdown.

    Produces:
    - Title as ``# Title``
    - Section headings as ``## Section``
    - Abstract wrapped as ``**Abstract:** ...``
    - Paragraph breaks (double newlines)
    - Stripped page numbers and running headers/footers

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Clean Markdown string.
    """
    blocks = _get_blocks_with_fonts(pdf_path)
    if not blocks:
        return ""

    # Determine median font size to classify headings
    sizes = sorted(b["size"] for b in blocks)
    median_size = sizes[len(sizes) // 2]

    # Title: first block on page 1 with font size clearly above body text (>= 1.4× median).
    # We use the FIRST such block (not the largest), to avoid arXiv stamp overlays, watermarks,
    # or other decorative large-font elements that may appear later in the block stream.
    title_threshold = median_size * 1.40
    title_idx = next(
        (i for i, b in enumerate(blocks) if b["page"] == 1 and b["size"] >= title_threshold),
        0,
    )
    title_size = blocks[title_idx]["size"] if blocks else median_size

    # Heading threshold: must be clearly above body text AND above a minimum absolute gap
    # Use 20% above median to reduce false positives from slightly-larger body text
    heading_threshold = max(median_size * 1.20, median_size + 1.5)

    md_lines: list[str] = []
    title_written = False
    in_abstract = False
    abstract_lines: list[str] = []
    prev_text = ""

    # Process only from the title block onward (skip preamble/watermarks)
    for block in blocks[title_idx:]:
        text = block["text"].strip()
        size = block["size"]

        if not text or _is_page_number(text):
            continue

        # Skip arXiv stamp / submission identifiers
        if re.match(r"^arXiv:\d{4}\.\d+", text):
            continue

        # Title: largest font, first occurrence
        if not title_written and size >= title_size * 0.95:
            md_lines.append(f"# {text}")
            title_written = True
            prev_text = text
            continue

        # Abstract keyword
        if re.match(r"^abstract\b", text, re.IGNORECASE):
            in_abstract = True
            # Abstract may be on the same line: "Abstract This paper..."
            after = re.sub(r"^abstract\s*[:.]?\s*", "", text, flags=re.IGNORECASE).strip()
            if after:
                abstract_lines.append(after)
            prev_text = text
            continue

        # Section heading detection:
        # 1) Font significantly larger than median (20%+ above)
        # 2) Line is ALL CAPS and short (≤60 chars)
        # 3) Looks like "1. Introduction" or "2.3 Related Work"
        is_large_font = size >= heading_threshold
        is_all_caps = text.isupper() and len(text) <= 60
        is_numbered_section = bool(re.match(r"^\d+(\.\d+)*\.?\s+[A-Z]", text) and len(text) <= 80)

        if is_large_font or is_all_caps or is_numbered_section:
            # Flush abstract if we were collecting it
            if in_abstract and abstract_lines:
                abstract_body = " ".join(abstract_lines).strip()
                md_lines.append(f"\n**Abstract:** {abstract_body}\n")
                abstract_lines = []
                in_abstract = False

            md_lines.append(f"\n## {text}\n")
            prev_text = text
            continue

        # If collecting abstract body
        if in_abstract:
            abstract_lines.append(text)
            prev_text = text
            continue

        # Regular paragraph text — add spacing when switching paragraphs
        if prev_text and not prev_text.endswith(("-", "–")):
            if prev_text.endswith(".") or prev_text.endswith(":"):
                md_lines.append("")
        md_lines.append(text)
        prev_text = text

    # Flush any remaining abstract
    if in_abstract and abstract_lines:
        abstract_body = " ".join(abstract_lines).strip()
        md_lines.append(f"\n**Abstract:** {abstract_body}\n")

    return "\n".join(md_lines)


def save_markdown(pdf_path: str, md_content: str) -> str:
    """Save Markdown content alongside the PDF (replaces .pdf extension with .md).

    Args:
        pdf_path: Path to the source PDF file.
        md_content: Markdown string to save.

    Returns:
        Path to the saved .md file.
    """
    md_path = Path(pdf_path).with_suffix(".md")
    md_path.write_text(md_content, encoding="utf-8")
    return str(md_path)

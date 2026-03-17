"""ArXiv API integration."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import httpx

from openseed.models.paper import Author, Paper

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?$")
_ARXIV_API = "http://export.arxiv.org/api/query"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"


def parse_arxiv_id(url: str) -> str | None:
    """Extract ArXiv ID from a URL or raw ID string."""
    match = _ARXIV_ID_RE.search(url)
    return match.group(1) if match else None


async def fetch_paper_metadata(arxiv_id: str) -> Paper:
    """Fetch paper metadata from the ArXiv API."""
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(_ARXIV_API, params={"id_list": arxiv_id})
        resp.raise_for_status()

    root = ET.fromstring(resp.text)
    entry = root.find(f"{_ATOM_NS}entry")
    if entry is None:
        raise ValueError(f"No entry found for {arxiv_id}")

    title = (entry.findtext(f"{_ATOM_NS}title") or "").strip()
    abstract = (entry.findtext(f"{_ATOM_NS}summary") or "").strip()

    authors = []
    for author_el in entry.findall(f"{_ATOM_NS}author"):
        name = (author_el.findtext(f"{_ATOM_NS}name") or "").strip()
        if name:
            authors.append(Author(name=name))

    return Paper(
        title=title,
        authors=authors,
        abstract=abstract,
        arxiv_id=arxiv_id,
        url=f"https://arxiv.org/abs/{arxiv_id}",
    )


def search_papers(query: str, max_results: int = 10) -> list[Paper]:
    """Search ArXiv for papers matching a keyword query.

    Args:
        query: Search terms to query against all fields.
        max_results: Maximum number of results to return.

    Returns:
        List of Paper objects with metadata populated.
    """
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        resp = client.get(
            _ARXIV_API,
            params={"search_query": f"all:{query}", "max_results": max_results},
        )
        resp.raise_for_status()

    root = ET.fromstring(resp.text)
    papers = []
    for entry in root.findall(f"{_ATOM_NS}entry"):
        entry_id = (entry.findtext(f"{_ATOM_NS}id") or "").strip()
        arxiv_id = parse_arxiv_id(entry_id)
        if not arxiv_id:
            continue

        title = (entry.findtext(f"{_ATOM_NS}title") or "").strip().replace("\n", " ")
        abstract = (entry.findtext(f"{_ATOM_NS}summary") or "").strip()

        authors = []
        for author_el in entry.findall(f"{_ATOM_NS}author"):
            name = (author_el.findtext(f"{_ATOM_NS}name") or "").strip()
            if name:
                authors.append(Author(name=name))

        papers.append(
            Paper(
                title=title,
                authors=authors,
                abstract=abstract,
                arxiv_id=arxiv_id,
                url=f"https://arxiv.org/abs/{arxiv_id}",
            )
        )
    return papers


async def download_pdf(arxiv_id: str, dest: str) -> str:
    """Download the PDF for an ArXiv paper."""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            f.write(resp.content)
    return dest

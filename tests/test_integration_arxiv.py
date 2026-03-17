"""Integration tests: real ArXiv API. Run with: pytest -m integration"""

from __future__ import annotations

import asyncio

import pytest

from openseed.services.arxiv import fetch_paper_metadata, parse_arxiv_id, search_papers

pytestmark = pytest.mark.integration

_TRANSFORMER = "1706.03762"
_BERT = "1810.04805"


def test_fetch_transformer_title_and_authors():
    paper = asyncio.run(fetch_paper_metadata(_TRANSFORMER))
    assert paper.arxiv_id == _TRANSFORMER
    assert "attention" in paper.title.lower()
    assert len(paper.authors) >= 5
    assert paper.url == f"https://arxiv.org/abs/{_TRANSFORMER}"


def test_fetch_abstract_length():
    paper = asyncio.run(fetch_paper_metadata(_TRANSFORMER))
    assert len(paper.abstract) > 200


def test_fetch_bert():
    paper = asyncio.run(fetch_paper_metadata(_BERT))
    assert paper.arxiv_id == _BERT
    assert len(paper.title) > 0
    assert len(paper.authors) >= 2


def test_fetch_invalid_id_raises():
    with pytest.raises(Exception):
        asyncio.run(fetch_paper_metadata("9999.99999"))


def test_search_returns_results():
    papers = search_papers("transformer self-attention", max_results=5)
    assert len(papers) >= 1
    assert all(p.arxiv_id for p in papers)
    assert all(p.title for p in papers)
    assert all(p.url for p in papers)


def test_search_relevance():
    papers = search_papers("variational autoencoder latent", max_results=5)
    titles_and_abstracts = " ".join(p.title.lower() + " " + p.abstract.lower() for p in papers)
    assert (
        "autoencoder" in titles_and_abstracts
        or "variational" in titles_and_abstracts
        or "latent" in titles_and_abstracts
    )


def test_url_parse_roundtrip():
    paper = asyncio.run(fetch_paper_metadata(_TRANSFORMER))
    assert parse_arxiv_id(paper.url) == _TRANSFORMER


def test_pdf_url_parse():
    assert parse_arxiv_id(f"https://arxiv.org/pdf/{_TRANSFORMER}.pdf") == _TRANSFORMER


def test_fetch_concurrent_two_papers():
    """asyncio.gather on two papers completes without error."""
    import asyncio as _asyncio

    async def _both():
        return await _asyncio.gather(
            fetch_paper_metadata(_TRANSFORMER),
            fetch_paper_metadata(_BERT),
        )

    p1, p2 = asyncio.run(_both())
    assert p1.arxiv_id == _TRANSFORMER
    assert p2.arxiv_id == _BERT

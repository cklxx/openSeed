"""Integration tests: Semantic Scholar citation API. Run with: pytest -m integration"""

from __future__ import annotations

import pytest

from openseed.agent.reader import _fetch_citations, enrich_citations

pytestmark = pytest.mark.integration

_TRANSFORMER = "1706.03762"
_BERT = "1810.04805"


def test_fetch_transformer_citations():
    result = _fetch_citations([_TRANSFORMER])
    assert _TRANSFORMER in result
    assert result[_TRANSFORMER] > 10_000  # "Attention Is All You Need" has 100k+


def test_fetch_bert_citations():
    result = _fetch_citations([_BERT])
    assert _BERT in result
    assert result[_BERT] > 10_000  # BERT has many thousands of citations


def test_fetch_batch():
    result = _fetch_citations([_TRANSFORMER, _BERT])
    assert len(result) == 2
    assert all(v > 0 for v in result.values())


def test_fetch_empty_input():
    assert _fetch_citations([]) == {}


def test_fetch_unknown_id_does_not_raise():
    result = _fetch_citations(["0001.00001"])
    assert isinstance(result, dict)


def test_enrich_adds_year_and_score():
    papers = [
        {"arxiv_id": _TRANSFORMER, "citations": 0, "title": "Transformer"},
        {"arxiv_id": _BERT, "citations": 0, "title": "BERT"},
    ]
    enriched = enrich_citations(papers)
    assert len(enriched) == 2
    assert all("year" in p for p in enriched)
    assert all("score" in p for p in enriched)
    assert all(p["citations"] > 0 for p in enriched)


def test_enrich_sorted_descending():
    papers = [
        {"arxiv_id": _TRANSFORMER, "citations": 0, "title": "Transformer"},
        {"arxiv_id": _BERT, "citations": 0, "title": "BERT"},
    ]
    enriched = enrich_citations(papers)
    scores = [p["score"] for p in enriched]
    assert scores == sorted(scores, reverse=True)


def test_real_citations_exceed_estimated():
    """Real Semantic Scholar counts should replace the 0-estimate we seeded."""
    papers = [{"arxiv_id": _TRANSFORMER, "citations": 0, "title": "T"}]
    enriched = enrich_citations(papers)
    assert enriched[0]["citations"] > 0

"""Tests for the research strategy module."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from openseed.agent.strategy import (
    ReadingRecommendation,
    ResearchStrategy,
    _build_cluster_summaries,
    _group_by_tags,
    _parse_gap_json,
)
from openseed.models.paper import Paper, Tag
from openseed.storage.library import PaperLibrary


@pytest.fixture()
def library(tmp_path):
    lib = PaperLibrary(tmp_path / "lib")
    yield lib
    lib.close()


def _make_paper(title: str, tags: list[str] | None = None, pid: str | None = None) -> Paper:
    tag_objs = [Tag(name=t) for t in (tags or [])]
    kwargs = {"title": title, "abstract": f"Abstract of {title}", "tags": tag_objs}
    if pid:
        kwargs["id"] = pid
    return Paper(**kwargs)


class TestGroupByTags:
    def test_groups_by_first_tag(self):
        papers = [
            _make_paper("P1", ["ml", "nlp"]),
            _make_paper("P2", ["ml"]),
            _make_paper("P3", ["cv"]),
        ]
        groups = _group_by_tags(papers)
        assert set(groups.keys()) == {"ml", "cv"}
        assert len(groups["ml"]) == 2
        assert len(groups["cv"]) == 1

    def test_untagged_papers(self):
        papers = [_make_paper("P1"), _make_paper("P2", ["ml"])]
        groups = _group_by_tags(papers)
        assert "untagged" in groups
        assert len(groups["untagged"]) == 1

    def test_empty_list(self):
        assert _group_by_tags([]) == {}


class TestClusterSummaries:
    def test_returns_titles_per_tag(self):
        groups = _group_by_tags(
            [
                _make_paper("Attention", ["transformers"]),
                _make_paper("BERT", ["transformers"]),
            ]
        )
        summaries = _build_cluster_summaries(groups)
        assert summaries == {"transformers": ["Attention", "BERT"]}


class TestParseGapJson:
    def test_valid_json(self):
        raw = json.dumps([{"cluster_name": "ml", "confidence": 0.8}])
        assert len(_parse_gap_json(raw)) == 1

    def test_with_markdown_fences(self):
        raw = '```json\n[{"cluster_name": "ml"}]\n```'
        assert len(_parse_gap_json(raw)) == 1

    def test_invalid_json(self):
        assert _parse_gap_json("not json") == []

    def test_non_array_json(self):
        assert _parse_gap_json('{"key": "value"}') == []


class TestAnalyzeGapsEmpty:
    def test_empty_library(self, library):
        strategy = ResearchStrategy(library)
        assert strategy.analyze_gaps() == []


class TestAnalyzeGapsFewPapers:
    def test_returns_simple_gaps_under_threshold(self, library):
        library.add_paper(_make_paper("P1", ["ml"]))
        library.add_paper(_make_paper("P2", ["cv"]))
        strategy = ResearchStrategy(library)
        gaps = strategy.analyze_gaps()
        assert len(gaps) == 2
        assert all(g.confidence == 0.3 for g in gaps)
        names = {g.cluster_name for g in gaps}
        assert names == {"ml", "cv"}


class TestAnalyzeGapsWithAI:
    @patch("openseed.agent.strategy._ask")
    def test_returns_ai_gaps(self, mock_ask, library):
        for i, tag in enumerate(["ml", "ml", "nlp"]):
            library.add_paper(_make_paper(f"Paper {i}", [tag]))
        ai_response = json.dumps(
            [
                {
                    "cluster_name": "ml",
                    "gap_description": "No papers on reinforcement learning",
                    "suggested_queries": ["reinforcement learning survey"],
                    "confidence": 0.9,
                },
                {
                    "cluster_name": "nlp",
                    "gap_description": "Missing multilingual models",
                    "suggested_queries": ["multilingual NLP"],
                    "confidence": 0.7,
                },
            ]
        )
        mock_ask.return_value = ai_response
        strategy = ResearchStrategy(library)
        gaps = strategy.analyze_gaps()
        assert len(gaps) == 2
        assert gaps[0].confidence >= gaps[1].confidence
        assert gaps[0].cluster_name == "ml"
        mock_ask.assert_called_once()

    @patch("openseed.agent.strategy._ask", side_effect=Exception("API timeout"))
    def test_ai_failure_falls_back(self, mock_ask, library):
        for i in range(3):
            library.add_paper(_make_paper(f"Paper {i}", ["ml"]))
        strategy = ResearchStrategy(library)
        gaps = strategy.analyze_gaps()
        assert len(gaps) >= 1
        assert all(g.confidence == 0.3 for g in gaps)


class TestSuggestReadingOrder:
    def test_empty_results(self, library):
        strategy = ResearchStrategy(library)
        assert strategy.suggest_reading_order("quantum") == []

    def test_no_match(self, library):
        library.add_paper(_make_paper("Deep Learning Basics", ["ml"]))
        strategy = ResearchStrategy(library)
        assert strategy.suggest_reading_order("quantum physics") == []

    def test_simple_order_few_papers(self, library):
        p1 = _make_paper("Transformer Basics", ["transformers"], pid="p1")
        p2 = _make_paper("Transformer Applications", ["transformers"], pid="p2")
        library.add_paper(p1)
        library.add_paper(p2)
        strategy = ResearchStrategy(library)
        recs = strategy.suggest_reading_order("Transformer")
        assert len(recs) == 2
        assert recs[0].priority == 1
        assert recs[1].priority == 2

    def test_citation_based_order(self, library):
        p1 = _make_paper("Foundational Attention", ["attention"], pid="p1")
        p2 = _make_paper("Attention Extension", ["attention"], pid="p2")
        library.add_paper(p1)
        library.add_paper(p2)
        library.add_edge("p2", "p1", "cites")
        strategy = ResearchStrategy(library)
        recs = strategy.suggest_reading_order("Attention")
        assert recs[0].paper.id == "p1"
        assert "citation" in recs[0].reason.lower()

    @patch("openseed.agent.strategy._ask")
    def test_ai_reading_order(self, mock_ask, library):
        papers = []
        for i in range(3):
            p = _make_paper(f"Attention Paper {i}", ["attention"], pid=f"att{i}")
            library.add_paper(p)
            papers.append(p)
        ai_response = json.dumps(
            [
                {"id": "att2", "reason": "Most recent advances"},
                {"id": "att0", "reason": "Foundational concepts"},
                {"id": "att1", "reason": "Builds on att0"},
            ]
        )
        mock_ask.return_value = ai_response
        strategy = ResearchStrategy(library)
        recs = strategy.suggest_reading_order("Attention")
        assert len(recs) == 3
        assert recs[0].paper.id == "att2"
        assert recs[0].priority == 1

    @patch("openseed.agent.strategy._ask", side_effect=Exception("fail"))
    def test_ai_failure_reading_order(self, mock_ask, library):
        for i in range(3):
            library.add_paper(_make_paper(f"ML Paper {i}", ["ml"], pid=f"ml{i}"))
        strategy = ResearchStrategy(library)
        recs = strategy.suggest_reading_order("ML")
        assert len(recs) == 3
        assert all(isinstance(r, ReadingRecommendation) for r in recs)

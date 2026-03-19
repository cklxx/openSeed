"""Tests for openseed.agent.autoresearch — multi-round discovery pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openseed.agent.autoresearch import AutoResearcher
from openseed.models.paper import Author, Paper
from openseed.models.research import ResearchSession
from openseed.storage.library import PaperLibrary


@pytest.fixture
def lib(tmp_path: Path) -> PaperLibrary:
    return PaperLibrary(tmp_path / "library")


@pytest.fixture
def researcher(lib: PaperLibrary) -> AutoResearcher:
    return AutoResearcher(model="claude-3-5-haiku-20241022", lib=lib)


@pytest.fixture
def sample_paper() -> Paper:
    return Paper(
        title="Test Paper on Attention",
        authors=[Author(name="Jane Doe")],
        abstract="A study about attention mechanisms.",
        arxiv_id="2401.12345",
        summary="Summary of attention paper.",
    )


@pytest.fixture
def raw_paper_dict() -> dict:
    return {
        "arxiv_id": "2401.12345",
        "score": 10,
        "citations": 5,
        "title": "Test Paper on Attention",
    }


class TestQueryVariants:
    def test_returns_parsed_lines(self, researcher: AutoResearcher) -> None:
        with patch("openseed.agent.autoresearch._ask", return_value="query one\nquery two\n"):
            variants = researcher._query_variants("attention", depth=2)
        assert variants == ["query one", "query two"]

    def test_falls_back_to_topic_on_empty_response(self, researcher: AutoResearcher) -> None:
        with patch("openseed.agent.autoresearch._ask", return_value="   \n  \n"):
            variants = researcher._query_variants("attention", depth=2)
        assert variants == ["attention"]

    def test_truncates_to_depth(self, researcher: AutoResearcher) -> None:
        with patch("openseed.agent.autoresearch._ask", return_value="a\nb\nc\nd"):
            variants = researcher._query_variants("topic", depth=2)
        assert len(variants) == 2


class TestDedupSorted:
    def test_deduplicates_by_arxiv_id(self, researcher: AutoResearcher) -> None:
        batch1 = [{"arxiv_id": "001", "score": 5}, {"arxiv_id": "002", "score": 3}]
        batch2 = [{"arxiv_id": "001", "score": 5}, {"arxiv_id": "003", "score": 1}]
        result = researcher._dedup_sorted([batch1, batch2], total_count=10)
        ids = [p["arxiv_id"] for p in result]
        assert ids.count("001") == 1
        assert len(result) == 3

    def test_sorts_by_score_descending(self, researcher: AutoResearcher) -> None:
        batch = [{"arxiv_id": "low", "score": 1}, {"arxiv_id": "high", "score": 9}]
        result = researcher._dedup_sorted([batch], total_count=10)
        assert result[0]["arxiv_id"] == "high"

    def test_respects_total_count_limit(self, researcher: AutoResearcher) -> None:
        batch = [{"arxiv_id": str(i), "score": i} for i in range(20)]
        result = researcher._dedup_sorted([batch], total_count=5)
        assert len(result) == 5

    def test_empty_batches_returns_empty(self, researcher: AutoResearcher) -> None:
        result = researcher._dedup_sorted([[], []], total_count=10)
        assert result == []


class TestRunEmptyDiscovery:
    def test_run_no_papers_found_returns_session(self, researcher: AutoResearcher) -> None:
        with (
            patch("openseed.agent.autoresearch._ask", return_value="query one"),
            patch("openseed.agent.autoresearch.discover_papers", return_value=[]),
            patch("openseed.agent.autoresearch.enrich_citations", return_value=[]),
            patch("openseed.agent.autoresearch.search_papers", return_value=[]),
        ):
            session = researcher.run("quantum computing", count=5, depth=1)
        assert isinstance(session, ResearchSession)
        assert session.topic == "quantum computing"
        assert session.paper_ids == []

    def test_run_no_papers_produces_empty_synthesis(self, researcher: AutoResearcher) -> None:
        with (
            patch("openseed.agent.autoresearch._ask", return_value="query one"),
            patch("openseed.agent.autoresearch.discover_papers", return_value=[]),
            patch("openseed.agent.autoresearch.enrich_citations", return_value=[]),
            patch("openseed.agent.autoresearch.search_papers", return_value=[]),
        ):
            session = researcher.run("quantum computing", count=5, depth=1)
        assert session.synthesis == ""
        assert session.report == ""


class TestRunSuccessful:
    def test_run_returns_session_with_paper_ids(
        self,
        researcher: AutoResearcher,
        sample_paper: Paper,
        raw_paper_dict: dict,
    ) -> None:
        mock_reader = MagicMock()
        mock_reader.summarize_paper.return_value = "Test summary"
        with (
            patch("openseed.agent.autoresearch._ask", side_effect=["query one", "the report text"]),
            patch("openseed.agent.autoresearch.discover_papers", return_value=[raw_paper_dict]),
            patch("openseed.agent.autoresearch.enrich_citations", return_value=[raw_paper_dict]),
            patch(
                "openseed.agent.autoresearch.fetch_paper_metadata",
                new=AsyncMock(return_value=sample_paper),
            ),
            patch("openseed.agent.autoresearch.synthesize_papers", return_value="synthesis"),
            patch("openseed.agent.autoresearch.auto_tag_paper", return_value=["ml"]),
            patch("openseed.agent.autoresearch.PaperReader", return_value=mock_reader),
            patch("time.sleep"),
        ):
            session = researcher.run("attention mechanisms", count=5, depth=1)
        assert len(session.paper_ids) == 1
        assert session.synthesis == "synthesis"
        assert session.report == "the report text"

    def test_on_step_callback_is_called(
        self,
        researcher: AutoResearcher,
        sample_paper: Paper,
        raw_paper_dict: dict,
    ) -> None:
        steps: list[str] = []
        mock_reader = MagicMock()
        mock_reader.summarize_paper.return_value = "Summary"
        with (
            patch("openseed.agent.autoresearch._ask", side_effect=["query one", "report"]),
            patch("openseed.agent.autoresearch.discover_papers", return_value=[raw_paper_dict]),
            patch("openseed.agent.autoresearch.enrich_citations", return_value=[raw_paper_dict]),
            patch(
                "openseed.agent.autoresearch.fetch_paper_metadata",
                new=AsyncMock(return_value=sample_paper),
            ),
            patch("openseed.agent.autoresearch.synthesize_papers", return_value="synth"),
            patch("openseed.agent.autoresearch.auto_tag_paper", return_value=[]),
            patch("openseed.agent.autoresearch.PaperReader", return_value=mock_reader),
            patch("time.sleep"),
        ):
            researcher.run("test", count=5, depth=1, on_step=steps.append)
        assert len(steps) > 0

    def test_cached_paper_skips_fetch(
        self,
        researcher: AutoResearcher,
        lib: PaperLibrary,
        sample_paper: Paper,
        raw_paper_dict: dict,
    ) -> None:
        lib.add_paper(sample_paper)
        with (
            patch("openseed.agent.autoresearch._ask", side_effect=["query one", "report"]),
            patch("openseed.agent.autoresearch.discover_papers", return_value=[raw_paper_dict]),
            patch("openseed.agent.autoresearch.enrich_citations", return_value=[raw_paper_dict]),
            patch("openseed.agent.autoresearch.fetch_paper_metadata") as mock_fetch,
            patch("openseed.agent.autoresearch.synthesize_papers", return_value="synth"),
            patch("openseed.agent.autoresearch.auto_tag_paper", return_value=[]),
            patch("openseed.agent.autoresearch.PaperReader"),
        ):
            session = researcher.run("attention", count=5, depth=1)
        mock_fetch.assert_not_called()
        assert len(session.paper_ids) == 1


class TestRunApiError:
    def test_api_error_skips_paper_gracefully(
        self, researcher: AutoResearcher, raw_paper_dict: dict
    ) -> None:
        with (
            patch("openseed.agent.autoresearch._ask", return_value="query one"),
            patch("openseed.agent.autoresearch.discover_papers", return_value=[raw_paper_dict]),
            patch("openseed.agent.autoresearch.enrich_citations", return_value=[raw_paper_dict]),
            patch(
                "openseed.agent.autoresearch.fetch_paper_metadata",
                new=AsyncMock(side_effect=RuntimeError("API down")),
            ),
            patch("time.sleep"),
        ):
            session = researcher.run("attention", count=5, depth=1)
        assert isinstance(session, ResearchSession)
        assert session.paper_ids == []

    def test_api_error_on_step_notified(
        self, researcher: AutoResearcher, raw_paper_dict: dict
    ) -> None:
        steps: list[str] = []
        with (
            patch("openseed.agent.autoresearch._ask", return_value="query one"),
            patch("openseed.agent.autoresearch.discover_papers", return_value=[raw_paper_dict]),
            patch("openseed.agent.autoresearch.enrich_citations", return_value=[raw_paper_dict]),
            patch(
                "openseed.agent.autoresearch.fetch_paper_metadata",
                new=AsyncMock(side_effect=RuntimeError("API down")),
            ),
            patch("time.sleep"),
        ):
            researcher.run("attention", count=5, depth=1, on_step=steps.append)
        assert any("Skipping" in s for s in steps)

"""Tests for the paper library storage."""

import pytest

from openseed.models.experiment import Experiment
from openseed.models.paper import Paper
from openseed.models.research import ResearchSession
from openseed.storage.library import PaperLibrary


class TestPaperLibrary:
    def test_add_and_list(self, tmp_library: PaperLibrary, sample_paper: Paper) -> None:
        tmp_library.add_paper(sample_paper)
        papers = tmp_library.list_papers()
        assert len(papers) == 1
        assert papers[0].title == sample_paper.title

    def test_get_paper(self, tmp_library: PaperLibrary, sample_paper: Paper) -> None:
        tmp_library.add_paper(sample_paper)
        found = tmp_library.get_paper(sample_paper.id)
        assert found is not None
        assert found.id == sample_paper.id

    def test_get_missing(self, tmp_library: PaperLibrary) -> None:
        assert tmp_library.get_paper("nonexistent") is None

    def test_remove_paper(self, tmp_library: PaperLibrary, sample_paper: Paper) -> None:
        tmp_library.add_paper(sample_paper)
        assert tmp_library.remove_paper(sample_paper.id) is True
        assert tmp_library.list_papers() == []

    def test_remove_missing(self, tmp_library: PaperLibrary) -> None:
        assert tmp_library.remove_paper("nonexistent") is False

    def test_update_paper(self, tmp_library: PaperLibrary, sample_paper: Paper) -> None:
        tmp_library.add_paper(sample_paper)
        sample_paper.status = "reading"
        tmp_library.update_paper(sample_paper)
        found = tmp_library.get_paper(sample_paper.id)
        assert found is not None
        assert found.status == "reading"

    def test_update_paper_not_found(self, tmp_library: PaperLibrary, sample_paper: Paper) -> None:
        with pytest.raises(KeyError, match=sample_paper.id):
            tmp_library.update_paper(sample_paper)

    def test_search(self, tmp_library: PaperLibrary, sample_paper: Paper) -> None:
        tmp_library.add_paper(sample_paper)
        results = tmp_library.search_papers("attention")
        assert len(results) == 1
        assert tmp_library.search_papers("nonexistent") == []

    def test_empty_library(self, tmp_library: PaperLibrary) -> None:
        assert tmp_library.list_papers() == []
        assert tmp_library.list_experiments() == []


class TestExperimentLibrary:
    def test_add_and_list(self, tmp_library: PaperLibrary, sample_experiment: Experiment) -> None:
        tmp_library.add_experiment(sample_experiment)
        experiments = tmp_library.list_experiments()
        assert len(experiments) == 1
        assert experiments[0].name == sample_experiment.name

    def test_get_experiment(self, tmp_library: PaperLibrary, sample_experiment: Experiment) -> None:
        tmp_library.add_experiment(sample_experiment)
        found = tmp_library.get_experiment(sample_experiment.id)
        assert found is not None
        assert found.id == sample_experiment.id

    def test_remove_experiment(
        self, tmp_library: PaperLibrary, sample_experiment: Experiment
    ) -> None:
        tmp_library.add_experiment(sample_experiment)
        assert tmp_library.remove_experiment(sample_experiment.id) is True
        assert tmp_library.list_experiments() == []


class TestResearchSessionLibrary:
    def test_add_and_list(self, tmp_library: PaperLibrary) -> None:
        s = ResearchSession(topic="attention mechanisms")
        tmp_library.add_research_session(s)
        sessions = tmp_library.list_research_sessions()
        assert len(sessions) == 1
        assert sessions[0].topic == "attention mechanisms"

    def test_get_session(self, tmp_library: PaperLibrary) -> None:
        s = ResearchSession(topic="diffusion")
        tmp_library.add_research_session(s)
        found = tmp_library.get_research_session(s.id)
        assert found is not None
        assert found.id == s.id

    def test_get_missing(self, tmp_library: PaperLibrary) -> None:
        assert tmp_library.get_research_session("nonexistent") is None

    def test_empty(self, tmp_library: PaperLibrary) -> None:
        assert tmp_library.list_research_sessions() == []

    def test_cache_consistency(self, tmp_library: PaperLibrary) -> None:
        s1 = ResearchSession(topic="topic a")
        s2 = ResearchSession(topic="topic b")
        tmp_library.add_research_session(s1)
        tmp_library.add_research_session(s2)
        assert len(tmp_library.list_research_sessions()) == 2


class TestSearchPapers:
    def test_multi_token(self, tmp_library: PaperLibrary, sample_paper: Paper) -> None:
        tmp_library.add_paper(sample_paper)
        # Both tokens present → match
        results = tmp_library.search_papers("attention transformer")
        assert len(results) == 1
        # One token absent → no match
        assert tmp_library.search_papers("attention python") == []

    def test_title_score_priority(self, tmp_library: PaperLibrary) -> None:
        # Paper with token in title should rank above paper with token only in abstract
        p_title = Paper(title="attention mechanism", abstract="some paper")
        p_abstract = Paper(title="neural network", abstract="uses attention for encoding")
        tmp_library.add_paper(p_title)
        tmp_library.add_paper(p_abstract)
        results = tmp_library.search_papers("attention")
        assert results[0].title == "attention mechanism"

    def test_empty_query(self, tmp_library: PaperLibrary, sample_paper: Paper) -> None:
        tmp_library.add_paper(sample_paper)
        assert tmp_library.search_papers("") == []

    def test_save_report(self, tmp_library: PaperLibrary) -> None:
        path = tmp_library.save_report("sess123", "transformers in NLP", "## Summary\nContent.")
        assert path.exists()
        assert "transformers_in_nlp" in path.name
        assert "sess123" in path.name
        content = path.read_text()
        assert "Research Report: transformers in NLP" in content

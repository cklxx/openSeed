"""Tests for data models."""

from openseed.models.experiment import Experiment, ExperimentRun
from openseed.models.paper import Annotation, Author, Paper, Tag
from openseed.models.research import ResearchSession


class TestPaperModels:
    def test_author_minimal(self) -> None:
        author = Author(name="Alice")
        assert author.name == "Alice"
        assert author.affiliation is None

    def test_author_full(self) -> None:
        author = Author(name="Bob", affiliation="MIT", email="bob@mit.edu")
        assert author.affiliation == "MIT"
        assert author.email == "bob@mit.edu"

    def test_tag_defaults(self) -> None:
        tag = Tag(name="ML")
        assert tag.color == "blue"

    def test_annotation_auto_id(self) -> None:
        a = Annotation(text="Important finding")
        assert a.id
        assert a.page is None
        assert a.note == ""

    def test_paper_defaults(self) -> None:
        paper = Paper(title="Test Paper")
        assert paper.id
        assert paper.status == "unread"
        assert paper.authors == []
        assert paper.tags == []
        assert paper.annotations == []

    def test_paper_roundtrip(self, sample_paper: Paper) -> None:
        data = sample_paper.model_dump(mode="json")
        restored = Paper.model_validate(data)
        assert restored.title == sample_paper.title
        assert len(restored.authors) == 2
        assert len(restored.tags) == 2


class TestExperimentModels:
    def test_experiment_run_defaults(self) -> None:
        run = ExperimentRun()
        assert run.status == "running"
        assert run.metrics == {}

    def test_experiment_minimal(self) -> None:
        exp = Experiment(name="Test", paper_id="p1")
        assert exp.id
        assert exp.runs == []

    def test_experiment_roundtrip(self, sample_experiment: Experiment) -> None:
        data = sample_experiment.model_dump(mode="json")
        restored = Experiment.model_validate(data)
        assert restored.name == sample_experiment.name
        assert restored.paper_id == sample_experiment.paper_id


class TestResearchSession:
    def test_defaults(self) -> None:
        s = ResearchSession(topic="transformers")
        assert s.topic == "transformers"
        assert s.paper_ids == []
        assert s.query_variants == []
        assert s.synthesis == ""
        assert s.report == ""
        assert s.id  # auto-generated

    def test_roundtrip(self) -> None:
        s = ResearchSession(topic="diffusion models", paper_ids=["abc", "def"])
        data = s.model_dump(mode="json")
        s2 = ResearchSession(**data)
        assert s2.topic == s.topic
        assert s2.paper_ids == s.paper_ids
        assert s2.id == s.id

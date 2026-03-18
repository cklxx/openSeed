"""Tests for SQLite storage, migration, and knowledge graph."""

import json
from pathlib import Path

import pytest

from openseed.models.paper import Author, Paper, Tag
from openseed.storage.library import PaperLibrary


@pytest.fixture
def lib(tmp_path: Path) -> PaperLibrary:
    return PaperLibrary(tmp_path / "library")


@pytest.fixture
def paper_a() -> Paper:
    return Paper(
        id="aaa111",
        title="Attention Is All You Need",
        authors=[Author(name="Vaswani")],
        abstract="We propose the Transformer architecture.",
        arxiv_id="1706.03762",
        tags=[Tag(name="transformers")],
    )


@pytest.fixture
def paper_b() -> Paper:
    return Paper(
        id="bbb222",
        title="BERT: Pre-training of Deep Bidirectional Transformers",
        authors=[Author(name="Devlin")],
        abstract="We introduce BERT for language understanding.",
        arxiv_id="1810.04805",
        tags=[Tag(name="nlp"), Tag(name="bert")],
    )


class TestSQLiteCRUD:
    def test_db_file_created(self, lib: PaperLibrary) -> None:
        assert (lib._dir / "library.db").exists()

    def test_add_duplicate_arxiv(self, lib: PaperLibrary, paper_a: Paper) -> None:
        assert lib.add_paper(paper_a) is True
        dup = Paper(id="other", title="Dup", arxiv_id="1706.03762")
        assert lib.add_paper(dup) is False

    def test_add_duplicate_url(self, lib: PaperLibrary) -> None:
        p1 = Paper(id="p1", title="P1", url="https://example.com/paper")
        p2 = Paper(id="p2", title="P2", url="https://example.com/paper")
        assert lib.add_paper(p1) is True
        assert lib.add_paper(p2) is False

    def test_get_paper_by_arxiv(self, lib: PaperLibrary, paper_a: Paper) -> None:
        lib.add_paper(paper_a)
        found = lib.get_paper_by_arxiv("1706.03762")
        assert found is not None
        assert found.id == "aaa111"

    def test_update_preserves_data(self, lib: PaperLibrary, paper_a: Paper) -> None:
        lib.add_paper(paper_a)
        paper_a.summary = "A great paper about transformers."
        paper_a.status = "read"
        lib.update_paper(paper_a)
        found = lib.get_paper("aaa111")
        assert found is not None
        assert found.summary == "A great paper about transformers."
        assert found.status == "read"
        assert found.authors[0].name == "Vaswani"


class TestFTSSearch:
    def test_search_by_title(self, lib: PaperLibrary, paper_a: Paper) -> None:
        lib.add_paper(paper_a)
        results = lib.search_papers("attention")
        assert len(results) == 1
        assert results[0].id == "aaa111"

    def test_search_by_abstract(self, lib: PaperLibrary, paper_a: Paper) -> None:
        lib.add_paper(paper_a)
        results = lib.search_papers("transformer")
        assert len(results) == 1

    def test_search_no_match(self, lib: PaperLibrary, paper_a: Paper) -> None:
        lib.add_paper(paper_a)
        assert lib.search_papers("quantum") == []

    def test_search_empty_query(self, lib: PaperLibrary) -> None:
        assert lib.search_papers("") == []

    def test_search_after_update(self, lib: PaperLibrary, paper_a: Paper) -> None:
        lib.add_paper(paper_a)
        paper_a.summary = "Novel quantum computing approach"
        lib.update_paper(paper_a)
        results = lib.search_papers("quantum")
        assert len(results) == 1

    def test_search_after_remove(self, lib: PaperLibrary, paper_a: Paper) -> None:
        lib.add_paper(paper_a)
        lib.remove_paper("aaa111")
        assert lib.search_papers("attention") == []

    def test_multi_token_search(self, lib: PaperLibrary, paper_a: Paper, paper_b: Paper) -> None:
        lib.add_paper(paper_a)
        lib.add_paper(paper_b)
        results = lib.search_papers("transformer")
        assert len(results) >= 1


class TestKnowledgeGraph:
    def test_add_edge(self, lib: PaperLibrary) -> None:
        assert lib.add_edge("aaa", "bbb", "cites") is True
        assert lib.edge_count() == 1

    def test_add_duplicate_edge(self, lib: PaperLibrary) -> None:
        lib.add_edge("aaa", "bbb", "cites")
        assert lib.add_edge("aaa", "bbb", "cites") is False
        assert lib.edge_count() == 1

    def test_different_edge_types(self, lib: PaperLibrary) -> None:
        lib.add_edge("aaa", "bbb", "cites")
        lib.add_edge("aaa", "bbb", "related")
        assert lib.edge_count() == 2

    def test_get_neighbors(self, lib: PaperLibrary) -> None:
        lib.add_edge("aaa", "bbb", "cites")
        lib.add_edge("ccc", "aaa", "cites")
        neighbors = lib.get_neighbors("aaa")
        ids = {n["paper_id"] for n in neighbors}
        assert ids == {"bbb", "ccc"}

    def test_get_neighbors_empty(self, lib: PaperLibrary) -> None:
        assert lib.get_neighbors("nonexistent") == []

    def test_get_edges_from(self, lib: PaperLibrary) -> None:
        lib.add_edge("aaa", "bbb", "cites")
        lib.add_edge("aaa", "ccc", "cites")
        edges = lib.get_edges_from("aaa")
        assert len(edges) == 2

    def test_get_edges_to(self, lib: PaperLibrary) -> None:
        lib.add_edge("bbb", "aaa", "cites")
        lib.add_edge("ccc", "aaa", "cites")
        edges = lib.get_edges_to("aaa")
        assert len(edges) == 2

    def test_clusters_single(self, lib: PaperLibrary) -> None:
        lib.add_edge("a", "b")
        lib.add_edge("b", "c")
        clusters = lib.get_clusters()
        assert len(clusters) == 1
        assert sorted(clusters[0]) == ["a", "b", "c"]

    def test_clusters_multiple(self, lib: PaperLibrary) -> None:
        lib.add_edge("a", "b")
        lib.add_edge("c", "d")
        clusters = lib.get_clusters()
        assert len(clusters) == 2

    def test_clusters_empty(self, lib: PaperLibrary) -> None:
        assert lib.get_clusters() == []

    def test_edge_with_metadata(self, lib: PaperLibrary) -> None:
        lib.add_edge("a", "b", metadata={"source": "semantic_scholar"})
        assert lib.edge_count() == 1


class TestJSONMigration:
    def _write_json(self, path: Path, data: list[dict]) -> None:
        path.write_text(json.dumps(data, default=str))

    def test_migrate_papers(self, tmp_path: Path) -> None:
        lib_dir = tmp_path / "library"
        lib_dir.mkdir()
        self._write_json(
            lib_dir / "papers.json",
            [
                {
                    "id": "p1",
                    "title": "Test Paper",
                    "arxiv_id": "2301.00001",
                    "status": "unread",
                    "added_at": "2025-01-01T00:00:00",
                    "abstract": "",
                    "authors": [],
                    "tags": [],
                    "annotations": [],
                    "note": "",
                    "url": None,
                    "pdf_path": None,
                    "summary": None,
                    "experiment_path": None,
                }
            ],
        )
        lib = PaperLibrary(lib_dir)
        papers = lib.list_papers()
        assert len(papers) == 1
        assert papers[0].title == "Test Paper"

    def test_migrate_preserves_backup(self, tmp_path: Path) -> None:
        lib_dir = tmp_path / "library"
        lib_dir.mkdir()
        self._write_json(lib_dir / "papers.json", [])
        PaperLibrary(lib_dir)
        assert (lib_dir / "json_backup").exists()

    def test_migrate_idempotent(self, tmp_path: Path) -> None:
        lib_dir = tmp_path / "library"
        lib_dir.mkdir()
        self._write_json(
            lib_dir / "papers.json",
            [
                {
                    "id": "p1",
                    "title": "Test",
                    "arxiv_id": None,
                    "status": "unread",
                    "added_at": "2025-01-01T00:00:00",
                    "abstract": "",
                    "authors": [],
                    "tags": [],
                    "annotations": [],
                    "note": "",
                    "url": None,
                    "pdf_path": None,
                    "summary": None,
                    "experiment_path": None,
                }
            ],
        )
        lib1 = PaperLibrary(lib_dir)
        assert len(lib1.list_papers()) == 1
        # Create a second instance — should not duplicate
        lib2 = PaperLibrary(lib_dir)
        assert len(lib2.list_papers()) == 1

    def test_no_json_no_migration(self, tmp_path: Path) -> None:
        lib_dir = tmp_path / "library"
        lib = PaperLibrary(lib_dir)
        assert lib.list_papers() == []
        assert not (lib_dir / "json_backup").exists()

    def test_migrate_watches(self, tmp_path: Path) -> None:
        lib_dir = tmp_path / "library"
        lib_dir.mkdir()
        self._write_json(
            lib_dir / "watches.json",
            [{"id": "w1", "query": "attention", "since_year": 2024, "last_run": None}],
        )
        lib = PaperLibrary(lib_dir)
        watches = lib.list_watches()
        assert len(watches) == 1
        assert watches[0].query == "attention"

    def test_migrate_experiments(self, tmp_path: Path) -> None:
        lib_dir = tmp_path / "library"
        lib_dir.mkdir()
        self._write_json(
            lib_dir / "experiments.json",
            [
                {
                    "id": "e1",
                    "name": "test exp",
                    "paper_id": "p1",
                    "repo_url": None,
                    "local_path": None,
                    "description": "",
                    "runs": [],
                    "created_at": "2025-01-01T00:00:00",
                    "tags": [],
                }
            ],
        )
        lib = PaperLibrary(lib_dir)
        assert len(lib.list_experiments()) == 1

    def test_migrate_sessions(self, tmp_path: Path) -> None:
        lib_dir = tmp_path / "library"
        lib_dir.mkdir()
        self._write_json(
            lib_dir / "research_sessions.json",
            [
                {
                    "id": "s1",
                    "topic": "transformers",
                    "query_variants": [],
                    "paper_ids": [],
                    "synthesis": "",
                    "report": "",
                    "created_at": "2025-01-01T00:00:00",
                }
            ],
        )
        lib = PaperLibrary(lib_dir)
        assert len(lib.list_research_sessions()) == 1

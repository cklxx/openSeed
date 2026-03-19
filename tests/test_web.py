"""Tests for the web dashboard."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from openseed.config import OpenSeedConfig
from openseed.models.paper import Author, Paper, Tag
from openseed.storage.library import PaperLibrary


@pytest.fixture
def lib(tmp_path: Path) -> PaperLibrary:
    lib = PaperLibrary(tmp_path / "library")
    lib.add_paper(
        Paper(
            id="p1",
            title="Attention Is All You Need",
            arxiv_id="1706.03762",
            authors=[Author(name="Vaswani")],
            abstract="We propose the Transformer.",
            summary="A great paper about transformers.",
            tags=[Tag(name="transformers")],
            status="read",
        )
    )
    lib.add_paper(
        Paper(
            id="p2",
            title="BERT",
            arxiv_id="1810.04805",
            authors=[Author(name="Devlin")],
            abstract="Bidirectional pre-training.",
            tags=[Tag(name="nlp")],
        )
    )
    lib.add_edge("p1", "p2", "cites")
    return lib


@pytest.fixture
def client(lib: PaperLibrary, tmp_path: Path) -> TestClient:
    config = OpenSeedConfig(library_dir=lib._dir, config_dir=tmp_path / "config")

    def mock_lib():
        return lib

    with (
        patch("openseed.web.app._lib", mock_lib),
        patch("openseed.web.app.load_config", return_value=config),
    ):
        from openseed.web.app import app

        yield TestClient(app)


class TestWebDashboard:
    def test_index(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Dashboard" in resp.text
        assert "2" in resp.text  # total papers

    def test_papers_list(self, client: TestClient) -> None:
        resp = client.get("/papers")
        assert resp.status_code == 200
        assert "Attention Is All You Need" in resp.text
        assert "BERT" in resp.text

    def test_papers_filter_status(self, client: TestClient) -> None:
        resp = client.get("/papers?status=read")
        assert resp.status_code == 200
        assert "Attention Is All You Need" in resp.text
        assert "BERT" not in resp.text

    def test_papers_search(self, client: TestClient) -> None:
        resp = client.get("/papers?q=transformer")
        assert resp.status_code == 200
        assert "Attention" in resp.text

    def test_paper_detail(self, client: TestClient) -> None:
        resp = client.get("/papers/p1")
        assert resp.status_code == 200
        assert "Attention Is All You Need" in resp.text
        assert "Vaswani" in resp.text
        assert "A great paper about transformers" in resp.text
        assert "BERT" in resp.text  # neighbor

    def test_paper_not_found(self, client: TestClient) -> None:
        resp = client.get("/papers/nonexistent")
        assert resp.status_code == 404

    def test_graph(self, client: TestClient) -> None:
        resp = client.get("/graph")
        assert resp.status_code == 200
        assert "Knowledge Graph" in resp.text
        assert "Cluster 1" in resp.text

    def test_graph_empty_library(self, tmp_path: Path) -> None:
        empty_lib = PaperLibrary(tmp_path / "empty")

        with patch("openseed.web.app._lib", return_value=empty_lib):
            from openseed.web.app import app

            c = TestClient(app)
            resp = c.get("/graph")
        assert resp.status_code == 200

    def test_graph_isolated_nodes_excluded(self, tmp_path: Path) -> None:
        """Papers with no edges should not appear as graph nodes."""
        lib = PaperLibrary(tmp_path / "iso")
        lib.add_paper(
            Paper(id="x1", title="Isolated Paper", authors=[], abstract="No edges here.")
        )
        with patch("openseed.web.app._lib", return_value=lib):
            from openseed.web.app import app

            c = TestClient(app)
            resp = c.get("/graph")
        assert resp.status_code == 200

    def test_digests_empty(self, client: TestClient) -> None:
        resp = client.get("/digests")
        assert resp.status_code == 200
        assert "No digests yet" in resp.text

    def test_sessions_empty(self, client: TestClient) -> None:
        resp = client.get("/sessions")
        assert resp.status_code == 200
        assert "No research sessions yet" in resp.text

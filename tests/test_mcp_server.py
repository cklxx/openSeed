"""Tests for MCP server tool functions."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from openseed.models.paper import Author, Paper, Tag
from openseed.storage.library import PaperLibrary


@pytest.fixture
def lib(tmp_path):
    library = PaperLibrary(tmp_path / "library")
    p = Paper(
        title="Attention Is All You Need",
        authors=[Author(name="Vaswani")],
        abstract="Transformer architecture. " * 50,
        summary="A detailed summary. " * 100,
        arxiv_id="1706.03762",
        tags=[Tag(name="transformers"), Tag(name="nlp")],
    )
    library.add_paper(p)
    return library


@pytest.fixture(autouse=True)
def patch_lib(lib):
    with patch("openseed.mcp.server._get_library", return_value=lib):
        yield


# ── library_stats ─────────────────────────────────────────────────────────────


def test_library_stats():
    from openseed.mcp.server import library_stats

    result = json.loads(library_stats())
    assert result["total_papers"] == 1
    assert "unread" in result["by_status"]
    assert "transformers" in result["top_tags"]


# ── search_papers ─────────────────────────────────────────────────────────────


def test_search_papers_returns_paginated():
    from openseed.mcp.server import search_papers

    result = json.loads(search_papers("attention"))
    assert "items" in result
    assert "total" in result
    assert "has_more" in result
    assert any("Attention" in p["title"] for p in result["items"])


def test_search_papers_empty():
    from openseed.mcp.server import search_papers

    result = json.loads(search_papers("nonexistent_xyz_query"))
    assert result["items"] == []
    assert result["total"] == 0


# ── get_paper ─────────────────────────────────────────────────────────────────


def test_get_paper_preview_truncates(lib):
    from openseed.mcp.server import get_paper

    pid = lib.list_papers()[0].id
    result = json.loads(get_paper(pid))
    assert result["title"] == "Attention Is All You Need"
    assert result.get("abstract_truncated") is True
    assert result.get("summary_truncated") is True


def test_get_paper_section_abstract(lib):
    from openseed.mcp.server import get_paper

    pid = lib.list_papers()[0].id
    result = json.loads(get_paper(pid, section="abstract"))
    assert "abstract_truncated" not in result
    assert "summary_truncated" in result  # summary still truncated


def test_get_paper_section_full(lib):
    from openseed.mcp.server import get_paper

    pid = lib.list_papers()[0].id
    result = json.loads(get_paper(pid, section="full"))
    assert "abstract_truncated" not in result
    assert "summary_truncated" not in result


def test_get_paper_not_found():
    from openseed.mcp.server import get_paper

    result = json.loads(get_paper("nonexistent"))
    assert "error" in result


# ── list_papers ───────────────────────────────────────────────────────────────


def test_list_papers_paginated():
    from openseed.mcp.server import list_papers

    result = json.loads(list_papers())
    assert result["total"] == 1
    assert len(result["items"]) == 1
    assert result["has_more"] is False


def test_list_papers_filter_status():
    from openseed.mcp.server import list_papers

    result = json.loads(list_papers(status="unread"))
    assert result["total"] == 1
    result = json.loads(list_papers(status="read"))
    assert result["total"] == 0


# ── get_graph ─────────────────────────────────────────────────────────────────


def test_get_graph():
    from openseed.mcp.server import get_graph

    result = json.loads(get_graph("nonexistent"))
    assert isinstance(result, list)


# ── search_memories ───────────────────────────────────────────────────────────


def test_search_memories_paginated():
    from openseed.mcp.server import search_memories

    result = json.loads(search_memories("attention"))
    assert "items" in result
    assert "total" in result


# ── ask_research ──────────────────────────────────────────────────────────────


@patch("openseed.agent.assistant._ask")
def test_ask_research(mock_ask):
    mock_ask.return_value = "Transformers use self-attention."
    from openseed.mcp.server import ask_research

    result = ask_research("What are transformers?")
    assert "attention" in result.lower()

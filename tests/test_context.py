"""Tests for the library-aware context engine."""

from __future__ import annotations

from dataclasses import dataclass

from openseed.agent.context import ContextBuilder, ContextResult
from openseed.models.paper import Author, Paper
from openseed.storage.library import PaperLibrary


def _make_paper(
    title: str = "Test Paper",
    abstract: str = "An abstract.",
    arxiv_id: str | None = "2401.12345",
    summary: str | None = None,
    paper_id: str | None = None,
) -> Paper:
    return Paper(
        id=paper_id or title.lower().replace(" ", "_")[:12],
        title=title,
        abstract=abstract,
        arxiv_id=arxiv_id,
        summary=summary,
        authors=[Author(name="Alice Smith")],
    )


def _make_library(tmp_path, papers: list[Paper] | None = None) -> PaperLibrary:
    lib = PaperLibrary(tmp_path / "lib")
    for p in papers or []:
        lib.add_paper(p)
    return lib


@dataclass
class FakeMemoryEntry:
    session: str = "sess1"
    timestamp: str = "2024-01-01"
    content: str = "Some memory content"


class FakeMemoryStore:
    def __init__(self, entries: list | None = None):
        self._entries = entries or []

    def search_memories(self, query: str) -> list:
        return self._entries


# ── Core build_context ─────────────────────────────────────


def test_build_context_with_papers(tmp_path):
    papers = [_make_paper(title="Attention Paper", abstract="Attention is all you need")]
    lib = _make_library(tmp_path, papers)
    builder = ContextBuilder(lib)
    result = builder.build_context("attention")
    assert isinstance(result, ContextResult)
    assert len(result.papers) == 1
    assert result.papers[0].title == "Attention Paper"
    assert result.total_tokens > 0
    assert "paper_content" in result.xml_context


def test_build_context_empty_library(tmp_path):
    lib = _make_library(tmp_path)
    builder = ContextBuilder(lib)
    result = builder.build_context("nonexistent query")
    assert result.papers == []
    assert result.memories == []
    assert result.xml_context == ""
    assert result.total_tokens == 0


def test_build_context_no_matching_papers(tmp_path):
    papers = [_make_paper(title="Transformers", abstract="Deep learning model")]
    lib = _make_library(tmp_path, papers)
    builder = ContextBuilder(lib)
    result = builder.build_context("quantum computing")
    assert result.papers == []
    assert result.xml_context == ""


# ── XML assembly ────────────────────────────────────────────


def test_xml_has_security_tags(tmp_path):
    papers = [_make_paper(title="Test", arxiv_id="2401.99999")]
    lib = _make_library(tmp_path, papers)
    builder = ContextBuilder(lib)
    result = builder.build_context("test")
    assert 'source="untrusted"' in result.xml_context
    assert 'paper_id="arxiv:2401.99999"' in result.xml_context


def test_xml_includes_abstract_and_summary(tmp_path):
    papers = [_make_paper(abstract="My abstract", summary="My summary")]
    lib = _make_library(tmp_path, papers)
    builder = ContextBuilder(lib)
    result = builder.build_context("test")
    assert "<abstract>My abstract</abstract>" in result.xml_context
    assert "<summary>My summary</summary>" in result.xml_context


def test_xml_skips_missing_abstract_and_summary(tmp_path):
    papers = [_make_paper(abstract="", summary=None)]
    lib = _make_library(tmp_path, papers)
    builder = ContextBuilder(lib)
    result = builder.build_context("test")
    assert "<abstract>" not in result.xml_context
    assert "<summary>" not in result.xml_context


def test_xml_escapes_special_characters(tmp_path):
    papers = [_make_paper(title='Paper <with> "special" & chars', abstract="A < B & C > D")]
    lib = _make_library(tmp_path, papers)
    builder = ContextBuilder(lib)
    result = builder.build_context("paper special chars")
    assert "&lt;with&gt;" in result.xml_context
    assert "&amp; chars" in result.xml_context
    assert "A &lt; B &amp; C &gt; D" in result.xml_context


def test_xml_memories_have_system_source(tmp_path):
    lib = _make_library(tmp_path, [_make_paper()])
    memory = FakeMemoryStore([FakeMemoryEntry(content="Remember this")])
    builder = ContextBuilder(lib, memory_store=memory)
    result = builder.build_context("test")
    assert 'source="system"' in result.xml_context
    assert "Remember this" in result.xml_context


# ── Token budget truncation ────────────────────────────────


def test_truncation_removes_least_relevant_papers(tmp_path):
    papers = [
        _make_paper(
            title=f"Paper {i}",
            arxiv_id=f"2401.{i:05d}",
            paper_id=f"p{i}",
            abstract="x" * 2000,
        )
        for i in range(20)
    ]
    lib = _make_library(tmp_path, papers)
    builder = ContextBuilder(lib)
    result = builder.build_context("paper", max_tokens=500)
    assert len(result.papers) < 20
    assert result.total_tokens <= 500


def test_truncation_never_removes_memories(tmp_path):
    papers = [
        _make_paper(
            title=f"Paper {i}",
            arxiv_id=f"2401.{i:05d}",
            paper_id=f"p{i}",
            abstract="x" * 2000,
        )
        for i in range(10)
    ]
    lib = _make_library(tmp_path, papers)
    memory = FakeMemoryStore([FakeMemoryEntry(content="Important memory")])
    builder = ContextBuilder(lib, memory_store=memory)
    result = builder.build_context("paper", max_tokens=500)
    assert len(result.memories) == 1
    assert "Important memory" in result.xml_context


# ── Graph expansion ─────────────────────────────────────────


def test_graph_expansion_adds_neighbors(tmp_path):
    p1 = _make_paper(title="Root Paper", arxiv_id="2401.00001", paper_id="root1")
    p2 = _make_paper(title="Neighbor Paper", arxiv_id="2401.00002", paper_id="neigh1")
    lib = _make_library(tmp_path, [p1, p2])
    lib.add_edge("root1", "neigh1", "cites")
    builder = ContextBuilder(lib)
    result = builder.build_context("root")
    paper_ids = {p.id for p in result.papers}
    assert "root1" in paper_ids
    assert "neigh1" in paper_ids


def test_graph_expansion_skips_missing_neighbors(tmp_path):
    p1 = _make_paper(title="Root Paper", arxiv_id="2401.00001", paper_id="root1")
    lib = _make_library(tmp_path, [p1])
    lib.add_edge("root1", "nonexistent", "cites")
    builder = ContextBuilder(lib)
    result = builder.build_context("root")
    assert len(result.papers) == 1
    assert result.papers[0].id == "root1"


# ── Memory store ────────────────────────────────────────────


def test_works_without_memory_store(tmp_path):
    lib = _make_library(tmp_path, [_make_paper()])
    builder = ContextBuilder(lib, memory_store=None)
    result = builder.build_context("test")
    assert result.memories == []
    assert 'source="system"' not in result.xml_context


def test_memory_store_error_returns_empty(tmp_path):
    class BrokenMemoryStore:
        def search_memories(self, query: str):
            raise RuntimeError("broken")

    lib = _make_library(tmp_path, [_make_paper()])
    builder = ContextBuilder(lib, memory_store=BrokenMemoryStore())
    result = builder.build_context("test")
    assert result.memories == []


# ── Debug info ──────────────────────────────────────────────


def test_debug_info_populated(tmp_path):
    papers = [_make_paper()]
    lib = _make_library(tmp_path, papers)
    builder = ContextBuilder(lib)
    result = builder.build_context("test")
    assert "papers_found" in result.debug_info
    assert "papers_after_expansion" in result.debug_info
    assert "papers_after_truncation" in result.debug_info
    assert "memories_found" in result.debug_info
    assert "estimated_tokens" in result.debug_info

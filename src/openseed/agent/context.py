"""Library-aware context engine for assembling research agent prompts."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from xml.sax.saxutils import escape

from openseed.models.paper import Paper
from openseed.storage.library import PaperLibrary


@dataclass
class ContextResult:
    """Result of context assembly for research agent prompts."""

    papers: list[Paper]
    memories: list  # MemoryEntry objects when available
    xml_context: str
    total_tokens: int
    debug_info: dict = field(default_factory=dict)


class ContextBuilder:
    """Assembles relevant papers and memories into XML context for prompts."""

    def __init__(self, library: PaperLibrary, memory_store=None) -> None:
        self._library = library
        self._memory_store = memory_store

    def build_context(self, query: str, max_tokens: int = 150_000) -> ContextResult:
        papers = self._search_papers(query)
        debug = {"papers_found": len(papers)}
        papers = self._expand_graph(papers)
        debug["papers_after_expansion"] = len(papers)
        memories = self._search_memories(query)
        debug["memories_found"] = len(memories)
        papers, memories = self._truncate_to_budget(papers, memories, max_tokens)
        debug["papers_after_truncation"] = len(papers)
        xml = self._assemble_xml(papers, memories)
        tokens = self._estimate_tokens(xml)
        debug["estimated_tokens"] = tokens
        return ContextResult(
            papers=papers,
            memories=memories,
            xml_context=xml,
            total_tokens=tokens,
            debug_info=debug,
        )

    def _search_papers(self, query: str, limit: int = 20) -> list[Paper]:
        try:
            return self._library.search_papers(query)[:limit]
        except sqlite3.OperationalError:
            return []

    def _expand_graph(self, papers: list[Paper], depth: int = 1) -> list[Paper]:
        if not papers:
            return papers
        seen_ids = {p.id for p in papers}
        top_papers = papers[:5]
        neighbors: list[Paper] = []
        for paper in top_papers:
            for edge in self._library.get_neighbors(paper.id):
                neighbor_id = edge["paper_id"]
                if neighbor_id in seen_ids:
                    continue
                seen_ids.add(neighbor_id)
                neighbor = self._library.get_paper(neighbor_id)
                if neighbor:
                    neighbors.append(neighbor)
        return papers + neighbors

    def _search_memories(self, query: str) -> list:
        if self._memory_store is None:
            return []
        try:
            return self._memory_store.search_memories(query)
        except Exception:
            return []

    def _assemble_xml(self, papers: list[Paper], memories: list) -> str:
        if not papers and not memories:
            return ""
        parts = ["<context>"]
        for paper in papers:
            paper_id = f"arxiv:{paper.arxiv_id}" if paper.arxiv_id else paper.id
            parts.append(
                f'  <paper_content source="untrusted"'
                f' paper_id="{escape(paper_id)}"'
                f' title="{escape(paper.title)}">'
            )
            if paper.abstract:
                parts.append(f"    <abstract>{escape(paper.abstract)}</abstract>")
            if paper.summary:
                parts.append(f"    <summary>{escape(paper.summary)}</summary>")
            parts.append("  </paper_content>")
        for entry in memories:
            session = escape(str(getattr(entry, "session", "")))
            timestamp = escape(str(getattr(entry, "timestamp", "")))
            content = escape(str(getattr(entry, "content", str(entry))))
            parts.append('  <research_memory source="system">')
            parts.append(
                f'    <entry session="{session}" timestamp="{timestamp}">{content}</entry>'
            )
            parts.append("  </research_memory>")
        parts.append("</context>")
        return "\n".join(parts)

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def _truncate_to_budget(
        self, papers: list[Paper], memories: list, max_tokens: int
    ) -> tuple[list[Paper], list]:
        xml = self._assemble_xml(papers, memories)
        while papers and self._estimate_tokens(xml) > max_tokens:
            papers = papers[:-1]
            xml = self._assemble_xml(papers, memories)
        return papers, memories

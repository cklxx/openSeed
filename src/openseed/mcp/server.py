"""OpenSeed MCP server — expose library and research tools via stdio."""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from openseed.config import load_config
from openseed.storage.library import PaperLibrary

mcp = FastMCP("openseed")
_lib: PaperLibrary | None = None


def _get_library() -> PaperLibrary:
    global _lib  # noqa: PLW0603
    if _lib is None:
        _lib = PaperLibrary(load_config().library_dir)
    return _lib


def _paper_summary_json(p) -> dict:
    """Compact paper representation — avoids dumping full abstract/summary."""
    tags = [t.name for t in p.tags] if p.tags else []
    return {
        "id": p.id,
        "title": p.title,
        "arxiv_id": p.arxiv_id,
        "status": p.status,
        "tags": tags,
    }


@mcp.tool()
def search_papers(query: str) -> str:
    """Use when the user asks about papers on a topic, wants to find what's in their library,
    or references a paper by keyword. Returns matching papers from the local library (not web).
    Try this first before ask_research for simple lookups."""
    papers = _get_library().search_papers(query)[:20]
    return json.dumps([_paper_summary_json(p) for p in papers])


@mcp.tool()
def get_paper(paper_id: str) -> str:
    """Use when the user asks for details about a specific paper (abstract, summary, authors)
    and you already have the paper_id from search_papers or list_papers."""
    paper = _get_library().get_paper(paper_id)
    if paper is None:
        return json.dumps({"error": f"Paper {paper_id} not found"})
    authors = [a.name for a in paper.authors] if paper.authors else []
    return json.dumps(
        {
            "id": paper.id,
            "title": paper.title,
            "arxiv_id": paper.arxiv_id,
            "authors": authors,
            "abstract": (paper.abstract or "")[:500],
            "summary": (paper.summary or "")[:1000],
            "status": paper.status,
            "tags": [t.name for t in paper.tags] if paper.tags else [],
        }
    )


@mcp.tool()
def list_papers(status: str | None = None, limit: int = 50) -> str:
    """Use when the user wants to see what papers are in their library, browse recent additions,
    or check library stats. Supports filtering by status: 'unread', 'read', 'archived'."""
    papers = _get_library().list_papers()
    if status:
        papers = [p for p in papers if p.status == status]
    return json.dumps([_paper_summary_json(p) for p in papers[:limit]])


@mcp.tool()
def get_graph(paper_id: str) -> str:
    """Use when the user asks how papers are connected, wants citation relationships,
    or asks 'what papers cite/reference this one'. Returns knowledge graph edges."""
    edges = _get_library().get_neighbors(paper_id)
    return json.dumps(edges)


@mcp.tool()
def ask_research(question: str) -> str:
    """Use ONLY when the user needs a synthesized answer that combines information across
    multiple papers — e.g. 'compare X and Y approaches' or 'what are the open problems in Z'.
    This is expensive (calls Claude API internally). For simple lookups, use search_papers."""
    from openseed.agent.assistant import ResearchAssistant

    lib = _get_library()
    assistant = ResearchAssistant(library=lib)
    return assistant.ask(question)


@mcp.tool()
def search_memories(query: str) -> str:
    """Use when the user references a prior research conversation or asks 'what did we
    discuss about X'. Searches saved conversation history from previous research sessions."""
    from openseed.agent.memory import MemoryStore

    store = MemoryStore(_get_library())
    entries = store.search_memories(query, top_k=5)
    return json.dumps(
        [
            {
                "session": e.session_id,
                "role": e.role,
                "content": e.content[:300],
                "at": e.created_at,
            }
            for e in entries
        ]
    )


def run_mcp_server() -> None:
    """Start the MCP server with stdio transport."""
    mcp.run(transport="stdio")

"""OpenSeed MCP server — expose library and research tools via stdio."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from openseed.config import load_config
from openseed.storage.library import PaperLibrary

_INSTRUCTIONS = """\
OpenSeed manages a local research paper library with AI-powered analysis.

Tool selection guide (cheapest first):
1. library_stats → answer "how many papers" without listing anything
2. search_papers / list_papers → find papers by keyword or browse
3. get_paper → drill into one paper's details (previewed; use section param for full text)
4. get_graph → citation/reference relationships between papers
5. search_memories → recall prior research conversations
6. ask_research → LAST RESORT, synthesize across papers (expensive, calls Claude API)
"""

_READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
_EXPENSIVE = ToolAnnotations(readOnlyHint=True, idempotentHint=False, openWorldHint=True)

mcp = FastMCP("openseed", instructions=_INSTRUCTIONS)
_lib: PaperLibrary | None = None

_PAGE_SIZE = 20


def _get_library() -> PaperLibrary:
    global _lib  # noqa: PLW0603
    if _lib is None:
        _lib = PaperLibrary(load_config().library_dir)
    return _lib


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    """Return (text, was_truncated). Truncates at word boundary."""
    if len(text) <= limit:
        return text, False
    return text[:limit].rsplit(" ", 1)[0] + "…", True


def _paper_brief(p) -> dict:
    """Compact paper representation for list views."""
    tags = [t.name for t in p.tags] if p.tags else []
    return {
        "id": p.id,
        "title": p.title,
        "arxiv_id": p.arxiv_id,
        "status": p.status,
        "tags": tags,
    }


def _paper_detail(p, section: str | None = None) -> dict:
    """Paper detail with progressive disclosure of long fields."""
    authors = [a.name for a in p.authors] if p.authors else []
    result: dict = {
        "id": p.id,
        "title": p.title,
        "arxiv_id": p.arxiv_id,
        "authors": authors,
        "status": p.status,
        "tags": [t.name for t in p.tags] if p.tags else [],
    }
    abstract = p.abstract or ""
    summary = p.summary or ""
    if section in ("abstract", "full"):
        result["abstract"] = abstract
    else:
        text, truncated = _truncate(abstract, 500)
        result["abstract"] = text
        if truncated:
            result["abstract_truncated"] = True
    if section in ("summary", "full"):
        result["summary"] = summary
    else:
        text, truncated = _truncate(summary, 1000)
        result["summary"] = text
        if truncated:
            result["summary_truncated"] = True
    return result


def _paginated(items: list, offset: int, page_size: int) -> dict:
    """Wrap a list in pagination metadata."""
    page = items[offset : offset + page_size]
    has_more = offset + page_size < len(items)
    return {
        "items": page,
        "total": len(items),
        "offset": offset,
        "has_more": has_more,
    }


@mcp.tool(annotations=_READ_ONLY)
def library_stats() -> str:
    """Use when the user asks how many papers they have, library size, or coverage overview.
    Cheapest tool — no paper content returned."""
    lib = _get_library()
    papers = lib.list_papers()
    from collections import Counter

    statuses = Counter(p.status for p in papers)
    tags: Counter = Counter()
    for p in papers:
        for t in p.tags:
            tags[t.name] += 1
    return json.dumps(
        {
            "total_papers": len(papers),
            "by_status": dict(statuses),
            "top_tags": dict(tags.most_common(10)),
        }
    )


@mcp.tool(annotations=_READ_ONLY)
def search_papers(
    query: Annotated[str, Field(description="Keywords to match against titles, abstracts, tags")],
    offset: Annotated[int, Field(description="Pagination offset (0, 20, 40, …)")] = 0,
) -> str:
    """Use when the user asks about papers on a topic or wants to find what's in their library.
    Searches the local library (not the web). Try this before ask_research for simple lookups.
    Returns paginated results — check has_more and use offset to get more."""
    papers = _get_library().search_papers(query)
    return json.dumps(_paginated([_paper_brief(p) for p in papers], offset, _PAGE_SIZE))


@mcp.tool(annotations=_READ_ONLY)
def get_paper(
    paper_id: Annotated[str, Field(description="Paper ID from search_papers or list_papers")],
    section: Annotated[
        Literal["preview", "abstract", "summary", "full"] | None,
        Field(description="Level of detail: preview (default), abstract, summary, or full"),
    ] = None,
) -> str:
    """Use when the user wants details about a specific paper — authors, abstract, summary.
    Default returns truncated previews. If a field shows truncated=true, call again with
    section='abstract', 'summary', or 'full' to get the complete text."""
    paper = _get_library().get_paper(paper_id)
    if paper is None:
        return json.dumps({"error": f"Paper {paper_id} not found"})
    return json.dumps(_paper_detail(paper, section))


@mcp.tool(annotations=_READ_ONLY)
def list_papers(
    status: Annotated[
        Literal["unread", "read", "archived"] | None,
        Field(description="Filter by reading status, or omit for all"),
    ] = None,
    offset: Annotated[int, Field(description="Pagination offset (0, 20, 40, …)")] = 0,
) -> str:
    """Use when the user wants to browse their library or see recent additions.
    Returns paginated results — check has_more and use offset to get more."""
    papers = _get_library().list_papers()
    if status:
        papers = [p for p in papers if p.status == status]
    return json.dumps(_paginated([_paper_brief(p) for p in papers], offset, _PAGE_SIZE))


@mcp.tool(annotations=_READ_ONLY)
def get_graph(
    paper_id: Annotated[str, Field(description="Paper ID to get citation/reference edges for")],
) -> str:
    """Use when the user asks how papers are connected, wants citation relationships,
    or asks 'what cites this paper'. Returns knowledge graph edges."""
    edges = _get_library().get_neighbors(paper_id)
    return json.dumps(edges)


@mcp.tool(annotations=_EXPENSIVE)
def ask_research(
    question: Annotated[str, Field(description="Research question to synthesize across papers")],
) -> str:
    """LAST RESORT — use only when the user needs a synthesized answer across multiple papers
    (e.g. 'compare X and Y' or 'open problems in Z'). Calls Claude API internally = slow + costly.
    For simple lookups, use search_papers + get_paper instead."""
    from openseed.agent.assistant import ResearchAssistant

    lib = _get_library()
    assistant = ResearchAssistant(library=lib)
    return assistant.ask(question)


@mcp.tool(annotations=_READ_ONLY)
def search_memories(
    query: Annotated[str, Field(description="Keywords to search in past conversation history")],
    offset: Annotated[int, Field(description="Pagination offset (0, 20, 40, …)")] = 0,
) -> str:
    """Use when the user references a prior research conversation or asks
    'what did we discuss about X'. Searches saved conversation history."""
    from openseed.agent.memory import MemoryStore

    store = MemoryStore(_get_library())
    entries = store.search_memories(query, top_k=100)
    items = [
        {"session": e.session_id, "role": e.role, "content": e.content, "at": e.created_at}
        for e in entries
    ]
    return json.dumps(_paginated(items, offset, _PAGE_SIZE))


def run_mcp_server() -> None:
    """Start the MCP server with stdio transport."""
    mcp.run(transport="stdio")

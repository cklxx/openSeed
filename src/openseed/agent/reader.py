"""Claude-powered paper reader via claude-agent-sdk."""

from __future__ import annotations

import asyncio
import re

import httpx
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query


def _make_opts(model: str, system: str) -> ClaudeAgentOptions:
    opts = ClaudeAgentOptions(
        system_prompt=system,
        disallowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Agent"],
        permission_mode="bypassPermissions",
    )
    opts.model = model
    return opts


async def _ask_async(model: str, system: str, prompt: str) -> str:
    result = ""
    async for msg in query(prompt=prompt, options=_make_opts(model, system)):
        if isinstance(msg, ResultMessage):
            result = msg.result or ""
    return result


def _ask(model: str, system: str, prompt: str) -> str:
    # Suppress the noisy anyio cancel-scope cleanup RuntimeError emitted by
    # claude-agent-sdk when the async generator exits early (cosmetic only).
    def _silence_cancel_scope(loop: asyncio.AbstractEventLoop, ctx: dict) -> None:
        exc = ctx.get("exception")
        if isinstance(exc, RuntimeError) and "cancel scope" in str(exc):
            return
        loop.default_exception_handler(ctx)

    loop = asyncio.new_event_loop()
    loop.set_exception_handler(_silence_cancel_scope)
    try:
        return loop.run_until_complete(_ask_async(model, system, prompt))
    finally:
        loop.close()


def auto_tag_paper(text: str, model: str) -> list[str]:
    """Generate 3-5 concise tags for a paper."""
    result = _ask(
        model,
        "You are a research taxonomy expert. Return ONLY 3-5 comma-separated lowercase tags "
        "(single words or short hyphenated phrases). No explanations, no numbering.",
        f"Generate tags for:\n\n{text}",
    )
    return [t.strip().lower() for t in result.split(",") if t.strip()][:5]


def generate_experiment_code(text: str, model: str) -> str:
    """Generate runnable Python experiment code based on paper content."""
    system = (
        "You are a research engineer. Based on the paper provided, write clean runnable Python "
        "experiment code implementing the core methodology. "
        "Use PyTorch or scikit-learn as appropriate. "
        "Include: imports, dataset stub, model definition, training loop, and evaluation. "
        "Add brief comments linking code to key paper concepts."
    )
    return _ask(model, system, f"Generate experiment code for:\n\n{text}")


_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")


def _fetch_citations(arxiv_ids: list[str]) -> dict[str, int]:
    """Fetch real citation counts from Semantic Scholar batch API."""
    if not arxiv_ids:
        return {}
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.post(
                "https://api.semanticscholar.org/graph/v1/paper/batch",
                json={"ids": [f"ArXiv:{aid}" for aid in arxiv_ids]},
                params={"fields": "citationCount"},
            )
            if resp.status_code == 200:
                return {
                    aid: (item.get("citationCount") or 0)
                    for aid, item in zip(arxiv_ids, resp.json())
                    if item is not None
                }
    except Exception:
        pass
    return {}


def _parse_ranked_lines(raw: str) -> list[dict]:
    papers = []
    for line in raw.strip().splitlines():
        parts = line.strip().split("|")
        if len(parts) < 3:
            continue
        arxiv_id = parts[0].strip()
        if not _ARXIV_ID_RE.match(arxiv_id):
            continue
        try:
            papers.append(
                {
                    "arxiv_id": arxiv_id,
                    "citations": int(
                        parts[1].strip().replace(",", "").replace("+", "").replace("~", "")
                    ),
                    "title": parts[2].strip(),
                    "authors": parts[3].strip() if len(parts) > 3 else "",
                    "relevance": parts[4].strip() if len(parts) > 4 else "",
                }
            )
        except (ValueError, IndexError):
            pass
    return papers


def discover_papers(search_query: str, model: str, count: int = 10) -> list[dict]:
    """Phase 1: Claude web search → parsed paper list with estimated citations."""
    system = (
        "You are a research paper discovery assistant with web search access. "
        f"Find {count} highly-cited, high-impact papers about the given topic. "
        "Strategy: first identify core concepts and related terms, then search "
        "Semantic Scholar and Google Scholar for the most cited papers in this area. "
        "Include both seminal foundational works and recent influential papers. "
        "Output ONLY pipe-separated lines — no markdown, no headers, no explanation:\n"
        "ARXIV_ID|ESTIMATED_CITATIONS|TITLE|FIRST_AUTHOR_ET_AL|ONE_LINE_RELEVANCE\n"
        "Example: 1706.03762|120000|Attention Is All You Need"
        "|Vaswani et al.|Transformer architecture\n"
        "Only include papers with valid ArXiv IDs. Sort descending by citation count."
    )
    raw = _ask(model, system, f"Find {count} papers about: {search_query}")
    return _parse_ranked_lines(raw)


def enrich_citations(papers: list[dict]) -> list[dict]:
    """Phase 2: Replace estimated citations with real counts from Semantic Scholar."""
    real = _fetch_citations([p["arxiv_id"] for p in papers])
    for p in papers:
        if p["arxiv_id"] in real:
            p["citations"] = real[p["arxiv_id"]]
    return sorted(papers, key=lambda x: x["citations"], reverse=True)


def search_papers_ranked(search_query: str, model: str, count: int = 10) -> list[dict]:
    """Full pipeline: discover via Claude + verify via Semantic Scholar."""
    return enrich_citations(discover_papers(search_query, model, count))


def search_papers_agent(search_query: str, model: str, count: int = 10) -> str:
    """Deep search using Claude web access — rich markdown output for pipeline command."""
    system = (
        "You are a research paper discovery assistant with web search access. "
        f"Find {count} high-value, highly-cited papers about the given topic. "
        "For each paper include: ArXiv ID, title, first 2 authors, year, "
        "citation count (from Semantic Scholar), and a one-sentence relevance note. "
        "Format as a markdown table: ArXiv ID | Title | Authors | Year | Citations | Relevance. "
        "Prioritize highly-cited papers. Use multiple searches to reach the target count. "
        "End with a ~150-word summary of key trends in this research area."
    )
    return _ask(model, system, f"Find {count} papers about: {search_query}")


class PaperReader:
    """Reads and analyzes papers using Claude via claude-agent-sdk."""

    def __init__(self, model: str = "claude-opus-4-6") -> None:
        self._model = model

    def summarize_paper(self, text: str, cn: bool = False) -> str:
        """Generate a structured markdown summary of a paper."""
        if cn:
            system = (
                "You are a research paper summarizer. Respond entirely in Chinese markdown: "
                "一句话概括, ## 核心贡献 (bullets), ## 方法论, ## 局限性, **相关性评分:** N/10."
            )
        else:
            system = (
                "You are a research paper summarizer. Respond in markdown with sections: "
                "one-liner, ## Key Contributions (bullets), ## Methodology, "
                "## Limitations, **Relevance Score:** N/10."
            )
        return _ask(self._model, system, f"Summarize this paper:\n\n{text}")

    def extract_key_findings(self, text: str) -> list[str]:
        """Extract key findings as a list."""
        result = _ask(
            self._model,
            "You are a research analyst. Extract key findings as a numbered list.",
            f"Extract the key findings:\n\n{text}",
        )
        return [line.strip() for line in result.strip().split("\n") if line.strip()]

    def generate_questions(self, text: str) -> list[str]:
        """Generate research questions based on a paper."""
        result = _ask(
            self._model,
            "You are a research advisor. Generate insightful questions about this paper.",
            f"Generate research questions about:\n\n{text}",
        )
        return [line.strip() for line in result.strip().split("\n") if line.strip()]

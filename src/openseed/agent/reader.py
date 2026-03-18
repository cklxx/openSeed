"""Claude-powered paper reader via claude-agent-sdk."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from collections.abc import Callable
from datetime import date

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
from claude_agent_sdk.types import AssistantMessage, ToolUseBlock

_log = logging.getLogger(__name__)


def _make_opts(model: str, system: str) -> ClaudeAgentOptions:
    opts = ClaudeAgentOptions(
        system_prompt=system,
        disallowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Agent"],
        permission_mode="bypassPermissions",
    )
    opts.model = model
    return opts


def _tool_label(block: ToolUseBlock) -> str:
    """Return a short human-readable label for a tool call."""
    name = block.name
    inp = block.input
    if name == "WebSearch":
        return f"WebSearch: {inp.get('query', '')[:60]}"
    if name == "WebFetch":
        url = inp.get("url", "")
        return f"WebFetch: {url[:60]}"
    return name


async def _ask_async(
    model: str,
    system: str,
    prompt: str,
    on_step: Callable[[str], None] | None = None,
    on_result: Callable[[object], None] | None = None,
) -> str:
    result = ""
    async for msg in query(prompt=prompt, options=_make_opts(model, system)):
        if isinstance(msg, AssistantMessage) and on_step:
            for block in msg.content:
                if isinstance(block, ToolUseBlock):
                    on_step(_tool_label(block))
        if isinstance(msg, ResultMessage):
            result = msg.result or ""
            if on_result:
                on_result(msg)
    return result


def _ask(
    model: str,
    system: str,
    prompt: str,
    on_step: Callable[[str], None] | None = None,
    on_result: Callable[[object], None] | None = None,
) -> str:
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
        return loop.run_until_complete(_ask_async(model, system, prompt, on_step, on_result))
    finally:
        loop.close()


def auto_tag_paper(
    text: str, model: str, on_step: Callable[[str], None] | None = None
) -> list[str]:
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
    from openseed.services.scholar import fetch_citation_counts

    return fetch_citation_counts(arxiv_ids)


def _parse_ranked_lines(raw: str) -> list[dict]:
    raw = re.sub(r"```[a-z]*\n?", "", raw).strip().strip("```")
    papers = []
    skipped = 0
    for line in raw.strip().splitlines():
        parts = line.strip().split("|")
        if len(parts) < 3:
            skipped += 1
            continue
        arxiv_id = parts[0].strip()
        if not _ARXIV_ID_RE.match(arxiv_id):
            skipped += 1
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
            skipped += 1
    if not papers and raw.strip():
        _log.warning(
            "discover_papers: 0 valid lines parsed (skipped=%d). Head: %.200s", skipped, raw
        )
    return papers


def discover_papers(
    search_query: str,
    model: str,
    count: int = 10,
    since_year: int | None = None,
    on_step: Callable[[str], None] | None = None,
) -> list[dict]:
    """Phase 1: Claude web search → parsed paper list with estimated citations."""
    recency = f" published in {since_year} or later" if since_year else ""
    scope = f"recent papers{recency}" if since_year else "highly-cited, high-impact papers"
    system = (
        "You are a research paper discovery assistant with web search access. "
        f"Find {count} {scope} about the given topic. "
        "Strategy: first identify core concepts and related terms, then search "
        "Semantic Scholar and Google Scholar for the most relevant papers in this area. "
        "Output ONLY pipe-separated lines — no markdown, no headers, no explanation:\n"
        "ARXIV_ID|ESTIMATED_CITATIONS|TITLE|FIRST_AUTHOR_ET_AL|ONE_LINE_RELEVANCE\n"
        "Example: 1706.03762|120000|Attention Is All You Need"
        "|Vaswani et al.|Transformer architecture\n"
        "Only include papers with valid ArXiv IDs. Sort descending by citation count."
    )
    raw = _ask(model, system, f"Find {count} papers about: {search_query}", on_step=on_step)
    return _parse_ranked_lines(raw)


def _freshness_score(arxiv_id: str, citations: int) -> tuple[int, float]:
    """Return (pub_year, score) where score = citations^0.6 * (1 + freshness).

    Freshness decays exponentially with a half-life of 18 months.
    ArXiv IDs encode publication date: YYMM.xxxxx (post-2015) or older 4-digit format.
    """
    match = re.match(r"^(\d{2})(\d{2})\.", arxiv_id)
    if match:
        yy, mm = int(match.group(1)), int(match.group(2))
        year = 2000 + yy if yy <= 99 else yy
        pub = date(year, max(1, min(mm, 12)), 1)
    else:
        pub = date(2010, 1, 1)  # unknown — treat as old
    today = date.today()
    age_months = max((today.year - pub.year) * 12 + (today.month - pub.month), 1)
    freshness = math.exp(-age_months / 18)
    return pub.year, citations**0.6 * (1 + freshness)


def enrich_citations(papers: list[dict]) -> list[dict]:
    """Phase 2: Replace estimated citations with real counts; rank by freshness-weighted score."""
    real = _fetch_citations([p["arxiv_id"] for p in papers])
    for p in papers:
        if p["arxiv_id"] in real:
            p["citations"] = real[p["arxiv_id"]]
        p["year"], p["score"] = _freshness_score(p["arxiv_id"], p["citations"])
    return sorted(papers, key=lambda x: x["score"], reverse=True)


def search_papers_ranked(search_query: str, model: str, count: int = 10) -> list[dict]:
    """Full pipeline: discover via Claude + verify via Semantic Scholar."""
    return enrich_citations(discover_papers(search_query, model, count))


def search_papers_agent(
    search_query: str,
    model: str,
    count: int = 10,
    on_step: Callable[[str], None] | None = None,
) -> str:
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
    return _ask(model, system, f"Find {count} papers about: {search_query}", on_step=on_step)


def synthesize_papers(texts: list[str], model: str) -> str:
    """Compare multiple papers: shared themes, methodology, differences, synthesis."""
    system = (
        "You are a research synthesis expert. Compare the provided papers and output markdown: "
        "## Shared Themes, ## Methodology Comparison, ## Key Differences, ## Synthesis."
    )
    body = "\n\n---\n\n".join(f"Paper {i + 1}:\n{t}" for i, t in enumerate(texts))
    return _ask(model, system, f"Synthesize these papers:\n\n{body}")


def extract_paper_visuals(text: str, model: str) -> dict:
    """Ask LLM to extract pipeline steps and metrics comparisons as JSON."""
    system = (
        "Extract from this paper into JSON with optional keys: "
        '"pipeline" (list of ≤6 concise method step names), '
        '"metrics" (list of {"name": str, "proposed": number, "baseline": number}). '
        "Return ONLY valid JSON, no markdown fences. Omit keys with no data."
    )
    raw = _ask(model, system, f"Extract visuals:\n\n{text}")
    raw = re.sub(r"```[a-z]*\n?", "", raw).strip().strip("```")
    try:
        return json.loads(raw)
    except Exception:
        return {}


def extract_references(text: str, model: str) -> list[str]:
    """Extract ArXiv IDs of papers referenced in the given text via AI."""
    system = (
        "You are a citation extraction expert. From the paper text below, identify all "
        "referenced papers that have ArXiv IDs. Return ONLY a comma-separated list of "
        "ArXiv IDs (format: YYMM.NNNNN). If no ArXiv IDs found, return 'NONE'."
    )
    raw = _ask(model, system, f"Extract ArXiv references from:\n\n{text[:8000]}")
    if "NONE" in raw.upper() or not raw.strip():
        return []
    ids = []
    for part in re.findall(r"\d{4}\.\d{4,5}", raw):
        if part not in ids:
            ids.append(part)
    return ids


class PaperReader:
    """Reads and analyzes papers using Claude via claude-agent-sdk."""

    def __init__(self, model: str = "claude-opus-4-6") -> None:
        self._model = model

    def summarize_paper(
        self, text: str, cn: bool = False, on_step: Callable[[str], None] | None = None
    ) -> str:
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
        return _ask(self._model, system, f"Summarize this paper:\n\n{text}", on_step=on_step)

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

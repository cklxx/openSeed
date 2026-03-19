"""AutoResearcher — autonomous multi-round paper discovery and analysis."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import httpx

from openseed.agent.discovery import discover_papers, enrich_citations
from openseed.agent.reader import (
    PaperReader,
    _ask,
    auto_tag_paper,
    synthesize_papers,
)
from openseed.models.paper import Paper, Tag
from openseed.models.research import ResearchSession
from openseed.services.arxiv import fetch_paper_metadata, search_papers
from openseed.storage.library import PaperLibrary

logger = logging.getLogger(__name__)

_VARIANT_SYSTEM = (
    "You are a research search strategist. Generate exactly {n} queries for a topic, "
    "each targeting a DIFFERENT angle from this taxonomy:\n"
    "  1. Core technical terminology (method names, acronyms)\n"
    "  2. Application domain / use-case framing\n"
    "  3. Comparative framing (X vs Y, alternatives to X)\n"
    "  4. Survey / benchmark framing (survey of X, evaluation of X)\n"
    "  5. Recent advances framing (2024 X, state-of-the-art X)\n"
    "Pick the {n} most distinct angles. Return one query per line, no numbering, no preamble."
)

_REPORT_SYSTEM = (
    "You are a research analyst. Generate a comprehensive research report in markdown. "
    "Sections and approximate length:\n"
    "## Executive Summary (~150 words): key finding + state of the field\n"
    "## Research Landscape (~200 words): how subfields relate, historical arc\n"
    "## Top Papers (markdown table, ALL papers): ArXiv ID | Title | Year | Key Contribution\n"
    "## Key Themes (~200 words, 4-6 bullet themes with evidence)\n"
    "## Research Gaps (~200 words, >=4 specific open problems citing papers)\n"
    "## Recommended Reading Order (ordered list with one-sentence rationale per paper)"
)

_MIN_VARIANT_YIELD = 3
_RETRY_DELAYS = [0.0, 2.0, 5.0]


class AutoResearcher:
    """Orchestrates multi-round discovery, analysis, synthesis, and reporting."""

    def __init__(self, model: str, lib: PaperLibrary) -> None:
        self._model = model
        self._lib = lib

    def run(
        self,
        topic: str,
        count: int = 15,
        depth: int = 2,
        since_year: int | None = None,
        on_step: Callable[[str], None] | None = None,
    ) -> ResearchSession:
        session = ResearchSession(topic=topic)
        variants = self._query_variants(topic, depth)
        session.query_variants = variants
        raw = self._multi_discover(variants, count, since_year)
        papers = self._batch_analyze(raw, on_step)
        session.paper_ids = [p.id for p in papers]
        session.synthesis = self._synthesize(papers, on_step)
        session.report = self._generate_report(session, papers, on_step)
        return session

    def _query_variants(self, topic: str, depth: int) -> list[str]:
        system = _VARIANT_SYSTEM.format(n=depth)
        raw = _ask(self._model, system, f"Topic: {topic}")
        lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
        return lines[:depth] or [topic]

    def _arxiv_fallback(self, query: str, count: int) -> list[dict]:
        """Direct ArXiv API search as fallback when Claude returns too few results."""
        try:
            papers = search_papers(query, max_results=count)
        except httpx.HTTPError as exc:
            logger.warning("ArXiv search failed for %r: %s", query, exc)
            return []
        except Exception:
            logger.exception("Unexpected error in ArXiv fallback for %r", query)
            return []
        return [
            {
                "arxiv_id": p.arxiv_id,
                "citations": 0,
                "title": p.title,
                "authors": p.authors[0].name if p.authors else "",
                "relevance": "arxiv-direct",
            }
            for p in papers
            if p.arxiv_id
        ]

    def _discover_variant(self, variant: str, per_query: int, since_year: int | None) -> list[dict]:
        results = enrich_citations(discover_papers(variant, self._model, per_query, since_year))
        if len(results) < _MIN_VARIANT_YIELD:
            results = enrich_citations(self._arxiv_fallback(variant, per_query))
        return results

    def _dedup_sorted(self, batches: list[list[dict]], total_count: int) -> list[dict]:
        seen: set[str] = set()
        all_papers: list[dict] = []
        for batch in batches:
            for p in batch:
                if p["arxiv_id"] not in seen:
                    seen.add(p["arxiv_id"])
                    all_papers.append(p)
        return sorted(all_papers, key=lambda x: x.get("score", 0), reverse=True)[:total_count]

    def _multi_discover(
        self, variants: list[str], total_count: int, since_year: int | None
    ) -> list[dict]:
        per_query = max(total_count // max(len(variants), 1), 5)
        with ThreadPoolExecutor(max_workers=len(variants)) as pool:
            batches = list(
                pool.map(
                    self._discover_variant,
                    variants,
                    [per_query] * len(variants),
                    [since_year] * len(variants),
                )
            )
        return self._dedup_sorted(batches, total_count)

    def _batch_analyze(
        self, raw_papers: list[dict], on_step: Callable[[str], None] | None
    ) -> list[Paper]:
        result: list[Paper] = []
        for rd in raw_papers:
            paper = self._analyze_one(rd, on_step)
            if paper:
                result.append(paper)
        return result

    def _cached_paper(self, arxiv_id: str) -> Paper | None:
        """Return existing library paper if already summarized, else None."""
        existing = next((p for p in self._lib.list_papers() if p.arxiv_id == arxiv_id), None)
        return existing if (existing and existing.summary) else None

    def _fetch_once(self, arxiv_id: str) -> Paper | None:
        """Single fetch attempt; returns None on transient error, raises on permanent."""
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(asyncio.run, fetch_paper_metadata(arxiv_id)).result()
        except (httpx.TimeoutException, httpx.HTTPStatusError):
            return None

    def _fetch_with_retry(
        self, arxiv_id: str, on_step: Callable[[str], None] | None
    ) -> Paper | None:
        for delay in _RETRY_DELAYS:
            if delay:
                time.sleep(delay)
            try:
                paper = self._fetch_once(arxiv_id)
            except httpx.HTTPError as exc:
                logger.warning("Network error fetching %s: %s", arxiv_id, exc)
                if on_step:
                    on_step(f"Skipping {arxiv_id}: {exc}")
                return None
            except Exception as exc:
                logger.exception("Unexpected error fetching %s", arxiv_id)
                if on_step:
                    on_step(f"Skipping {arxiv_id}: {exc}")
                return None
            if paper is not None:
                return paper
        if on_step:
            on_step(f"Skipping {arxiv_id}: network error after retries")
        return None

    def _analyze_one(self, rd: dict, on_step: Callable[[str], None] | None) -> Paper | None:
        if cached := self._cached_paper(rd["arxiv_id"]):
            if on_step:
                on_step(f"Reusing: {cached.title[:40]}…")
            return cached
        paper = self._fetch_with_retry(rd["arxiv_id"], on_step)
        if paper is None:
            return None
        text = paper.abstract or paper.title
        if on_step:
            on_step(f"Summarizing: {paper.title[:40]}…")
        paper.summary = PaperReader(model=self._model).summarize_paper(text, on_step=on_step)
        paper.tags = [Tag(name=t) for t in auto_tag_paper(text, self._model)]
        self._lib.add_paper(paper)
        self._lib.save_summary(paper)
        return paper

    def _synthesize(self, papers: list[Paper], on_step: Callable[[str], None] | None) -> str:
        if not papers:
            return ""
        if on_step:
            on_step("Synthesizing across papers…")
        texts = [f"Title: {p.title}\n\n{p.summary or p.abstract or p.title}" for p in papers]
        return synthesize_papers(texts, self._model)

    def _paper_report_entry(self, p: Paper) -> str:
        author = p.authors[0].name if p.authors else "Unknown"
        year = p.added_at.year
        content = p.summary or p.abstract or p.title
        return f"### {p.title}\nArXiv: {p.arxiv_id} | Year: {year} | Author: {author}\n{content}"

    def _build_report_prompt(self, session: ResearchSession, papers: list[Paper]) -> str:
        entries = "\n\n".join(self._paper_report_entry(p) for p in papers)
        index = "\n".join(f"- {p.arxiv_id}: {p.title}" for p in papers)
        return (
            f"Research topic: {session.topic}\n\n"
            f"## Cross-paper synthesis\n{session.synthesis}\n\n"
            f"## Full paper details\n{entries}\n\n"
            f"## Paper index\n{index}\n\n"
            "Instructions: build on the synthesis above — do not repeat it. "
            "The Top Papers table MUST include all papers. "
            "Research Gaps: >=4 specific gaps with evidence from papers. "
            "Recommended Reading Order: explain the reasoning."
        )

    def _generate_report(
        self,
        session: ResearchSession,
        papers: list[Paper],
        on_step: Callable[[str], None] | None,
    ) -> str:
        if not papers:
            return ""
        if on_step:
            on_step("Generating research report…")
        prompt = self._build_report_prompt(session, papers)
        report = _ask(self._model, _REPORT_SYSTEM, prompt, on_step=on_step)
        self._lib.save_report(session.id, session.topic, report)
        return report

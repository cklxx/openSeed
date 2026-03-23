"""Research strategy: gap analysis and reading order suggestions."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from openseed.agent.reader import _ask
from openseed.models.paper import Paper
from openseed.storage.library import PaperLibrary

_MIN_PAPERS_FOR_AI = 3


@dataclass
class GapAnalysis:
    cluster_name: str
    paper_count: int
    gap_description: str
    suggested_queries: list[str]
    confidence: float


@dataclass
class ReadingRecommendation:
    paper: Paper
    reason: str
    priority: int


def _parse_gap_json(raw: str) -> list[dict]:
    """Extract JSON array from AI response, tolerating markdown fences."""
    cleaned = re.sub(r"```[a-z]*\n?", "", raw).strip().strip("```")
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _parse_reading_json(raw: str) -> list[dict]:
    """Extract reading order JSON from AI response."""
    return _parse_gap_json(raw)


def _group_by_tags(papers: list[Paper]) -> dict[str, list[Paper]]:
    """Group papers by their primary tag (first tag or 'untagged')."""
    groups: dict[str, list[Paper]] = defaultdict(list)
    for p in papers:
        key = p.tags[0].name if p.tags else "untagged"
        groups[key].append(p)
    return dict(groups)


def _build_cluster_summaries(groups: dict[str, list[Paper]]) -> dict[str, list[str]]:
    return {tag: [p.title for p in papers] for tag, papers in groups.items()}


def _count_incoming_edges(library: PaperLibrary, paper_ids: set[str]) -> dict[str, int]:
    """Count incoming citation edges for each paper in the set."""
    counts: dict[str, int] = {}
    for pid in paper_ids:
        edges = library.get_edges_to(pid)
        counts[pid] = len(edges)
    return counts


class ResearchStrategy:
    """Analyzes a paper library to detect gaps and suggest reading order."""

    def __init__(
        self,
        library: PaperLibrary,
        model: str = "claude-sonnet-4-20250514",
        on_step: Callable[[str], None] | None = None,
    ) -> None:
        self._library = library
        self._model = model
        self._on_step = on_step

    def analyze_gaps(self) -> list[GapAnalysis]:
        papers = self._library.list_papers()
        if not papers:
            return []
        groups = _group_by_tags(papers)
        summaries = _build_cluster_summaries(groups)
        if len(papers) < _MIN_PAPERS_FOR_AI:
            return self._simple_gap_analysis(groups, summaries)
        return self._ai_gap_analysis(groups, summaries)

    def _simple_gap_analysis(
        self,
        groups: dict[str, list[Paper]],
        summaries: dict[str, list[str]],
    ) -> list[GapAnalysis]:
        return [
            GapAnalysis(
                cluster_name=tag,
                paper_count=len(titles),
                gap_description=f"Only {len(titles)} paper(s) on '{tag}'",
                suggested_queries=[tag],
                confidence=0.3,
            )
            for tag, titles in summaries.items()
        ]

    def _ai_gap_analysis(
        self,
        groups: dict[str, list[Paper]],
        summaries: dict[str, list[str]],
    ) -> list[GapAnalysis]:
        prompt = (
            "Given these paper clusters in a research library:\n"
            f"{json.dumps(summaries, indent=2)}\n\n"
            "Identify research gaps — areas referenced but not well covered. "
            "Return a JSON array of objects with keys: "
            '"cluster_name", "gap_description", "suggested_queries" (list of search strings), '
            '"confidence" (0-1). Return ONLY valid JSON, no markdown.'
        )
        system = "You are a research librarian. Analyze coverage gaps. Return ONLY JSON."
        try:
            raw = _ask(self._model, system, prompt, on_step=self._on_step)
            parsed = _parse_gap_json(raw)
        except Exception:
            parsed = []
        if not parsed:
            return self._simple_gap_analysis(groups, summaries)
        return self._gaps_from_parsed(parsed, groups)

    def _gaps_from_parsed(
        self, parsed: list[dict], groups: dict[str, list[Paper]]
    ) -> list[GapAnalysis]:
        results = []
        for item in parsed:
            name = item.get("cluster_name", "unknown")
            results.append(
                GapAnalysis(
                    cluster_name=name,
                    paper_count=len(groups.get(name, [])),
                    gap_description=item.get("gap_description", ""),
                    suggested_queries=item.get("suggested_queries", []),
                    confidence=float(item.get("confidence", 0.5)),
                )
            )
        return sorted(results, key=lambda g: g.confidence, reverse=True)

    def suggest_reading_order(self, topic: str) -> list[ReadingRecommendation]:
        papers = self._library.search_papers(topic)
        if not papers:
            return []
        paper_ids = {p.id for p in papers}
        in_counts = _count_incoming_edges(self._library, paper_ids)
        has_edges = any(c > 0 for c in in_counts.values())
        if has_edges:
            papers = sorted(papers, key=lambda p: in_counts.get(p.id, 0), reverse=True)
        else:
            papers = sorted(papers, key=lambda p: p.added_at)
        if len(papers) < _MIN_PAPERS_FOR_AI:
            return self._simple_reading_order(papers, has_edges)
        return self._ai_reading_order(papers, has_edges)

    def _simple_reading_order(
        self, papers: list[Paper], has_edges: bool
    ) -> list[ReadingRecommendation]:
        basis = "citation count" if has_edges else "date added"
        return [
            ReadingRecommendation(paper=p, reason=f"Ordered by {basis}", priority=i + 1)
            for i, p in enumerate(papers)
        ]

    def _ai_reading_order(
        self, papers: list[Paper], has_edges: bool
    ) -> list[ReadingRecommendation]:
        titles = [{"id": p.id, "title": p.title} for p in papers]
        prompt = (
            f"Papers (pre-sorted by {'citations' if has_edges else 'date'}):\n"
            f"{json.dumps(titles, indent=2)}\n\n"
            "Suggest a reading order. Return a JSON array of objects with keys: "
            '"id", "reason". First item = read first. Return ONLY valid JSON.'
        )
        system = "You are a research advisor. Suggest reading order. Return ONLY JSON."
        try:
            raw = _ask(self._model, system, prompt, on_step=self._on_step)
            parsed = _parse_reading_json(raw)
        except Exception:
            parsed = []
        if not parsed:
            return self._simple_reading_order(papers, has_edges)
        return self._reading_from_parsed(parsed, papers)

    def _reading_from_parsed(
        self, parsed: list[dict], papers: list[Paper]
    ) -> list[ReadingRecommendation]:
        by_id = {p.id: p for p in papers}
        results: list[ReadingRecommendation] = []
        seen: set[str] = set()
        for i, item in enumerate(parsed):
            pid = item.get("id", "")
            if pid in by_id and pid not in seen:
                seen.add(pid)
                results.append(
                    ReadingRecommendation(
                        paper=by_id[pid],
                        reason=item.get("reason", ""),
                        priority=i + 1,
                    )
                )
        for p in papers:
            if p.id not in seen:
                results.append(
                    ReadingRecommendation(
                        paper=p, reason="Not ranked by AI", priority=len(results) + 1
                    )
                )
        return results

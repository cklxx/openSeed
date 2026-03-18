"""Semantic Scholar API client."""

from __future__ import annotations

import logging
import time

import httpx

_log = logging.getLogger(__name__)
_S2_BASE = "https://api.semanticscholar.org/graph/v1/paper"
_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_RATE_DELAY = 3.0


def fetch_citation_counts(arxiv_ids: list[str]) -> dict[str, int]:
    """Fetch real citation counts from Semantic Scholar batch API."""
    if not arxiv_ids:
        return {}
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(
                f"{_S2_BASE}/batch",
                json={"ids": [f"ArXiv:{aid}" for aid in arxiv_ids]},
                params={"fields": "citationCount"},
            )
            if resp.status_code == 429:
                _log.warning("Semantic Scholar rate-limited; using estimated citations")
                return {}
            if resp.status_code != 200:
                _log.warning("Semantic Scholar returned %d", resp.status_code)
                return {}
            return {
                aid: (item.get("citationCount") or 0)
                for aid, item in zip(arxiv_ids, resp.json())
                if item is not None
            }
    except httpx.TimeoutException:
        _log.warning("Semantic Scholar timed out; using estimated citations")
    except Exception as exc:
        _log.warning("Semantic Scholar error: %s", exc)
    return {}


def get_references(arxiv_id: str) -> list[str]:
    """Fetch ArXiv IDs of papers referenced by the given paper."""
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.get(
                f"{_S2_BASE}/ArXiv:{arxiv_id}/references",
                params={"fields": "externalIds", "limit": 100},
            )
            if resp.status_code in (429, 404):
                return []
            if resp.status_code != 200:
                _log.warning("S2 references returned %d for %s", resp.status_code, arxiv_id)
                return []
            refs = []
            for item in resp.json().get("data", []):
                cited = item.get("citedPaper", {})
                ext = cited.get("externalIds") or {}
                if aid := ext.get("ArXiv"):
                    refs.append(aid)
            return refs
    except httpx.TimeoutException:
        _log.warning("S2 references timed out for %s", arxiv_id)
    except Exception as exc:
        _log.warning("S2 references error for %s: %s", arxiv_id, exc)
    return []


def get_recommendations(arxiv_id: str, limit: int = 5) -> list[dict]:
    """Fetch recommended papers via Semantic Scholar recommendations API."""
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.get(
                f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/ArXiv:{arxiv_id}",
                params={"fields": "title,externalIds,citationCount,year", "limit": limit},
            )
            if resp.status_code in (429, 404):
                return []
            if resp.status_code != 200:
                return []
            results = []
            for paper in resp.json().get("recommendedPapers", []):
                ext = paper.get("externalIds") or {}
                if aid := ext.get("ArXiv"):
                    results.append(
                        {
                            "arxiv_id": aid,
                            "title": paper.get("title", ""),
                            "citations": paper.get("citationCount", 0),
                            "year": paper.get("year"),
                        }
                    )
            return results
    except (httpx.TimeoutException, Exception) as exc:
        _log.warning("S2 recommendations error for %s: %s", arxiv_id, exc)
    return []


def batch_get_references(
    arxiv_ids: list[str], on_progress: callable | None = None
) -> dict[str, list[str]]:
    """Fetch references for multiple papers with rate limiting."""
    result: dict[str, list[str]] = {}
    for i, aid in enumerate(arxiv_ids):
        if i > 0:
            time.sleep(_RATE_DELAY)
        result[aid] = get_references(aid)
        if on_progress:
            on_progress(i + 1, len(arxiv_ids))
    return result

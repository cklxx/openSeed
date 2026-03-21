"""Semantic Scholar API client — async-first with sync wrappers."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import httpx

_log = logging.getLogger(__name__)
_S2_BASE = "https://api.semanticscholar.org/graph/v1/paper"
_S2_RECS = "https://api.semanticscholar.org/recommendations/v1/papers/forpaper"
_TIMEOUT = httpx.Timeout(30.0, connect=5.0)
_MAX_RETRIES = 3


async def _get_retried(client: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response | None:
    """GET with exponential backoff on 429 and retries on timeout."""
    for attempt in range(_MAX_RETRIES):
        try:
            resp = await client.get(url, **kwargs)
            if resp.status_code != 429:
                return resp
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(2**attempt)
        except httpx.TimeoutException:
            if attempt == _MAX_RETRIES - 1:
                raise
            await asyncio.sleep(2**attempt)
    return None


async def _post_retried(client: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response | None:
    """POST with exponential backoff on 429 and retries on timeout."""
    for attempt in range(_MAX_RETRIES):
        try:
            resp = await client.post(url, **kwargs)
            if resp.status_code != 429:
                return resp
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(2**attempt)
        except httpx.TimeoutException:
            if attempt == _MAX_RETRIES - 1:
                raise
            await asyncio.sleep(2**attempt)
    return None


def _parse_references(data: list[dict]) -> list[str]:
    refs = []
    for item in data:
        cited = item.get("citedPaper", {})
        ext = cited.get("externalIds") or {}
        if aid := ext.get("ArXiv"):
            refs.append(aid)
    return refs


def _parse_recommendations(papers: list[dict]) -> list[dict]:
    results = []
    for paper in papers:
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


async def fetch_citation_counts_async(arxiv_ids: list[str]) -> dict[str, int]:
    """Fetch real citation counts from Semantic Scholar batch API."""
    if not arxiv_ids:
        return {}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await _post_retried(
                client,
                f"{_S2_BASE}/batch",
                json={"ids": [f"ArXiv:{aid}" for aid in arxiv_ids]},
                params={"fields": "citationCount"},
            )
        if resp is None:
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


async def get_references_async(arxiv_id: str) -> list[str]:
    """Fetch ArXiv IDs of papers referenced by the given paper."""
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await _get_retried(
                client,
                f"{_S2_BASE}/ArXiv:{arxiv_id}/references",
                params={"fields": "externalIds", "limit": 100},
            )
        if resp is None or resp.status_code in (429, 404):
            return []
        if resp.status_code != 200:
            _log.warning("S2 references returned %d for %s", resp.status_code, arxiv_id)
            return []
        return _parse_references(resp.json().get("data", []))
    except httpx.TimeoutException:
        _log.warning("S2 references timed out for %s", arxiv_id)
    except Exception as exc:
        _log.warning("S2 references error for %s: %s", arxiv_id, exc)
    return []


async def get_recommendations_async(arxiv_id: str, limit: int = 5) -> list[dict]:
    """Fetch recommended papers via Semantic Scholar recommendations API."""
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await _get_retried(
                client,
                f"{_S2_RECS}/ArXiv:{arxiv_id}",
                params={"fields": "title,externalIds,citationCount,year", "limit": limit},
            )
        if resp is None or resp.status_code in (429, 404):
            return []
        if resp.status_code != 200:
            return []
        return _parse_recommendations(resp.json().get("recommendedPapers", []))
    except Exception as exc:
        _log.warning("S2 recommendations error for %s: %s", arxiv_id, exc)
    return []


async def batch_get_references_async(
    arxiv_ids: list[str], on_progress: Callable | None = None
) -> dict[str, list[str]]:
    """Fetch references for multiple papers concurrently (max 5 in-flight)."""
    sem = asyncio.Semaphore(5)
    completed = 0

    async def _fetch_one(aid: str) -> tuple[str, list[str]]:
        nonlocal completed
        async with sem:
            refs = await get_references_async(aid)
        completed += 1
        if on_progress:
            on_progress(completed, len(arxiv_ids))
        return aid, refs

    pairs = await asyncio.gather(*[_fetch_one(aid) for aid in arxiv_ids])
    return dict(pairs)


def _run_sync(coro):
    """Run an async coroutine in a fresh event loop — safe for sync callers."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def fetch_citation_counts(arxiv_ids: list[str]) -> dict[str, int]:
    """Sync wrapper — fetch real citation counts from Semantic Scholar batch API."""
    return _run_sync(fetch_citation_counts_async(arxiv_ids))


def get_references(arxiv_id: str) -> list[str]:
    """Sync wrapper — fetch ArXiv IDs of papers referenced by the given paper."""
    return _run_sync(get_references_async(arxiv_id))


def get_recommendations(arxiv_id: str, limit: int = 5) -> list[dict]:
    """Sync wrapper — fetch recommended papers via Semantic Scholar recommendations API."""
    return _run_sync(get_recommendations_async(arxiv_id, limit))


def batch_get_references(
    arxiv_ids: list[str], on_progress: Callable | None = None
) -> dict[str, list[str]]:
    """Sync wrapper — fetch references for multiple papers concurrently."""
    return _run_sync(batch_get_references_async(arxiv_ids, on_progress))

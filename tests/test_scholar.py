"""Tests for openseed.services.scholar — async and sync wrappers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from openseed.services.scholar import (
    _get_retried,
    _parse_recommendations,
    _parse_references,
    _post_retried,
    batch_get_references,
    batch_get_references_async,
    fetch_citation_counts,
    fetch_citation_counts_async,
    get_recommendations,
    get_recommendations_async,
    get_references,
    get_references_async,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(status: int, json_data) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.json.return_value = json_data
    return resp


# ---------------------------------------------------------------------------
# _parse_references
# ---------------------------------------------------------------------------


def test_parse_references_extracts_arxiv_ids():
    data = [
        {"citedPaper": {"externalIds": {"ArXiv": "1234.56789"}}},
        {"citedPaper": {"externalIds": {"DOI": "10.1234/foo"}}},  # no ArXiv
        {"citedPaper": {}},  # missing externalIds
    ]
    assert _parse_references(data) == ["1234.56789"]


def test_parse_references_empty():
    assert _parse_references([]) == []


# ---------------------------------------------------------------------------
# _parse_recommendations
# ---------------------------------------------------------------------------


def test_parse_recommendations_filters_non_arxiv():
    papers = [
        {"externalIds": {"ArXiv": "2301.00001"}, "title": "A", "citationCount": 10, "year": 2023},
        {"externalIds": {"DOI": "x"}, "title": "B"},  # no ArXiv
    ]
    result = _parse_recommendations(papers)
    assert len(result) == 1
    assert result[0] == {"arxiv_id": "2301.00001", "title": "A", "citations": 10, "year": 2023}


# ---------------------------------------------------------------------------
# _get_retried
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_retried_returns_on_success():
    resp = _make_response(200, {})
    client = AsyncMock()
    client.get = AsyncMock(return_value=resp)
    result = await _get_retried(client, "http://example.com")
    assert result is resp
    assert client.get.call_count == 1


@pytest.mark.asyncio
async def test_get_retried_retries_on_429_then_succeeds():
    rate_limited = _make_response(429, {})
    success = _make_response(200, {})
    client = AsyncMock()
    client.get = AsyncMock(side_effect=[rate_limited, success])
    with patch("openseed.services.scholar.asyncio.sleep", new_callable=AsyncMock):
        result = await _get_retried(client, "http://example.com")
    assert result is success
    assert client.get.call_count == 2


@pytest.mark.asyncio
async def test_get_retried_returns_none_after_all_429():
    rate_limited = _make_response(429, {})
    client = AsyncMock()
    client.get = AsyncMock(return_value=rate_limited)
    with patch("openseed.services.scholar.asyncio.sleep", new_callable=AsyncMock):
        result = await _get_retried(client, "http://example.com")
    assert result is None
    assert client.get.call_count == 3


@pytest.mark.asyncio
async def test_get_retried_reraises_timeout_on_last_attempt():
    client = AsyncMock()
    client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
    with patch("openseed.services.scholar.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(httpx.TimeoutException):
            await _get_retried(client, "http://example.com")


# ---------------------------------------------------------------------------
# _post_retried
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_retried_returns_on_success():
    resp = _make_response(200, {})
    client = AsyncMock()
    client.post = AsyncMock(return_value=resp)
    result = await _post_retried(client, "http://example.com")
    assert result is resp
    assert client.post.call_count == 1


@pytest.mark.asyncio
async def test_post_retried_returns_none_after_all_429():
    rate_limited = _make_response(429, {})
    client = AsyncMock()
    client.post = AsyncMock(return_value=rate_limited)
    with patch("openseed.services.scholar.asyncio.sleep", new_callable=AsyncMock):
        result = await _post_retried(client, "http://example.com")
    assert result is None
    assert client.post.call_count == 3


# ---------------------------------------------------------------------------
# fetch_citation_counts_async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_citation_counts_empty():
    assert await fetch_citation_counts_async([]) == {}


@pytest.mark.asyncio
async def test_fetch_citation_counts_success():
    resp = _make_response(200, [{"citationCount": 42}, {"citationCount": None}, None])
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("openseed.services.scholar.httpx.AsyncClient", return_value=mock_client):
        result = await fetch_citation_counts_async(["1706.03762", "1810.04805", "0000.00000"])

    assert result == {"1706.03762": 42, "1810.04805": 0}


@pytest.mark.asyncio
async def test_fetch_citation_counts_429():
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("openseed.services.scholar.httpx.AsyncClient", return_value=mock_client):
        with patch(
            "openseed.services.scholar._post_retried", new_callable=AsyncMock, return_value=None
        ):
            result = await fetch_citation_counts_async(["1706.03762"])

    assert result == {}


@pytest.mark.asyncio
async def test_fetch_citation_counts_timeout():
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("openseed.services.scholar.httpx.AsyncClient", return_value=mock_client):
        result = await fetch_citation_counts_async(["1706.03762"])

    assert result == {}


# ---------------------------------------------------------------------------
# get_references_async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_references_success():
    data = [
        {"citedPaper": {"externalIds": {"ArXiv": "1706.03762"}}},
        {"citedPaper": {"externalIds": {}}},
    ]
    resp = _make_response(200, {"data": data})
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("openseed.services.scholar.httpx.AsyncClient", return_value=mock_client):
        result = await get_references_async("2301.00001")

    assert result == ["1706.03762"]


@pytest.mark.asyncio
async def test_get_references_404():
    resp = _make_response(404, {})
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("openseed.services.scholar.httpx.AsyncClient", return_value=mock_client):
        result = await get_references_async("0000.00000")

    assert result == []


@pytest.mark.asyncio
async def test_get_references_network_error_returns_empty():
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=Exception("network failure"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("openseed.services.scholar.httpx.AsyncClient", return_value=mock_client):
        result = await get_references_async("1706.03762")

    assert result == []


# ---------------------------------------------------------------------------
# get_recommendations_async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_recommendations_success():
    papers = [
        {"externalIds": {"ArXiv": "2301.00001"}, "title": "T", "citationCount": 5, "year": 2023}
    ]
    resp = _make_response(200, {"recommendedPapers": papers})
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("openseed.services.scholar.httpx.AsyncClient", return_value=mock_client):
        result = await get_recommendations_async("1706.03762")

    assert len(result) == 1
    assert result[0]["arxiv_id"] == "2301.00001"


@pytest.mark.asyncio
async def test_get_recommendations_none_response():
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_make_response(429, {}))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("openseed.services.scholar.httpx.AsyncClient", return_value=mock_client):
        with patch("openseed.services.scholar.asyncio.sleep", new_callable=AsyncMock):
            result = await get_recommendations_async("1706.03762")

    assert result == []


# ---------------------------------------------------------------------------
# batch_get_references_async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_get_references_concurrent():
    """Verify all papers are fetched and progress callback is called."""
    ids = ["1706.03762", "1810.04805", "2301.00001"]
    call_log: list[int] = []

    async def fake_get_refs(arxiv_id: str) -> list[str]:
        return [f"ref-{arxiv_id}"]

    with patch("openseed.services.scholar.get_references_async", side_effect=fake_get_refs):
        result = await batch_get_references_async(ids, on_progress=lambda n, t: call_log.append(n))

    assert set(result.keys()) == set(ids)
    for aid in ids:
        assert result[aid] == [f"ref-{aid}"]
    assert sorted(call_log) == [1, 2, 3]


@pytest.mark.asyncio
async def test_batch_get_references_one_failure_does_not_crash_batch():
    """A failing paper returns [] but the rest succeed."""

    async def fake_get_refs(arxiv_id: str) -> list[str]:
        if arxiv_id == "bad":
            return []
        return ["ref-ok"]

    ids = ["good", "bad"]
    with patch("openseed.services.scholar.get_references_async", side_effect=fake_get_refs):
        result = await batch_get_references_async(ids)

    assert result["good"] == ["ref-ok"]
    assert result["bad"] == []


# ---------------------------------------------------------------------------
# _run_sync
# ---------------------------------------------------------------------------


def test_run_sync_executes_coroutine():
    from openseed.services.scholar import _run_sync

    async def _coro():
        return 42

    assert _run_sync(_coro()) == 42


# ---------------------------------------------------------------------------
# Sync wrappers
# ---------------------------------------------------------------------------


def test_sync_fetch_citation_counts():
    with patch("openseed.services.scholar._run_sync", return_value={"1706.03762": 50}) as m:
        result = fetch_citation_counts(["1706.03762"])
        assert result == {"1706.03762": 50}
        m.assert_called_once()


def test_sync_get_references():
    with patch("openseed.services.scholar._run_sync", return_value=["1706.03762"]) as m:
        result = get_references("2301.00001")
        assert result == ["1706.03762"]
        m.assert_called_once()


def test_sync_get_recommendations():
    expected = [{"arxiv_id": "2301.00001", "title": "T", "citations": 5, "year": 2023}]
    with patch("openseed.services.scholar._run_sync", return_value=expected) as m:
        result = get_recommendations("1706.03762")
        assert result == expected
        m.assert_called_once()


def test_sync_batch_get_references():
    expected = {"1706.03762": ["ref1"], "1810.04805": ["ref2"]}
    with patch("openseed.services.scholar._run_sync", return_value=expected) as m:
        result = batch_get_references(["1706.03762", "1810.04805"])
        assert result == expected
        m.assert_called_once()

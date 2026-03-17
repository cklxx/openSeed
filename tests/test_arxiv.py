"""Tests for ArXiv ID parsing."""

from __future__ import annotations

import pytest

from openseed.services.arxiv import parse_arxiv_id


class TestParseArxivId:
    @pytest.mark.parametrize(
        "url, expected",
        [
            ("https://arxiv.org/abs/1706.03762", "1706.03762"),
            ("https://arxiv.org/abs/2301.00234v2", "2301.00234"),
            ("1706.03762", "1706.03762"),
            ("2301.00234v3", "2301.00234"),
            ("https://arxiv.org/pdf/1706.03762.pdf", "1706.03762"),
        ],
    )
    def test_valid(self, url: str, expected: str) -> None:
        assert parse_arxiv_id(url) == expected

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/paper",
            "not-an-arxiv-id",
            "",
            "123.456",  # too short
        ],
    )
    def test_invalid(self, url: str) -> None:
        assert parse_arxiv_id(url) is None

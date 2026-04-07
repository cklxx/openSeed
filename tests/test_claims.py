"""Tests for claim extraction, matching, and alerts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openseed.models.claims import Alert, Claim, ClaimEdge
from openseed.models.paper import Paper
from openseed.storage.library import PaperLibrary

# ── Helpers ───────────────────────────────────────────────────


def _claim(pid: str, text: str, ctype: str = "finding") -> dict:
    return {"paper_id": pid, "claim_text": text, "claim_type": ctype}


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def lib(tmp_path: Path) -> PaperLibrary:
    return PaperLibrary(tmp_path / "library")


@pytest.fixture
def paper_a() -> Paper:
    return Paper(
        id="paper_a",
        title="Transformers outperform RNNs",
        abstract="We show transformers achieve SOTA results.",
        arxiv_id="2401.00001",
    )


@pytest.fixture
def paper_b() -> Paper:
    return Paper(
        id="paper_b",
        title="Recurrent networks revisited",
        abstract="We demonstrate RNNs remain competitive.",
        arxiv_id="2401.00002",
    )


SAMPLE_CLAIMS_A = [
    {
        "claim_text": "Transformers outperform RNNs on translation",
        "claim_type": "finding",
        "section": "Results",
        "source_quote": "Our transformer achieves 28.4 BLEU",
    },
    {
        "claim_text": "Self-attention is more parallelizable",
        "claim_type": "finding",
        "section": "Introduction",
        "source_quote": "Allows significantly more parallelization",
    },
]


# ── Schema Migration Tests ────────────────────────────────────


class TestSchemaMigration:
    def test_fresh_install_has_v2_tables(self, lib: PaperLibrary) -> None:
        tables = {
            r[0]
            for r in lib._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "paper_claims" in tables
        assert "claim_edges" in tables
        assert "alerts" in tables

    def test_schema_version_is_2(self, lib: PaperLibrary) -> None:
        row = lib._conn.execute("SELECT version FROM schema_version").fetchone()
        assert row[0] == 2

    def test_claims_status_column(self, lib: PaperLibrary) -> None:
        assert lib._has_column("papers", "claims_status")

    def test_full_text_column(self, lib: PaperLibrary) -> None:
        assert lib._has_column("papers", "full_text")

    def test_migration_idempotent(self, lib: PaperLibrary) -> None:
        lib._upgrade_schema()
        row = lib._conn.execute("SELECT version FROM schema_version").fetchone()
        assert row[0] == 2

    def test_cascade_delete(
        self,
        lib: PaperLibrary,
        paper_a: Paper,
    ) -> None:
        lib.add_paper(paper_a)
        ids = lib.add_claims([_claim("paper_a", "test claim")])
        assert len(ids) == 1
        lib.add_claim_edge(ids[0], ids[0], "supports", 0.9)
        lib.remove_paper("paper_a")
        assert lib.get_claims_for_paper("paper_a") == []

    def test_fts_triggers_sync(
        self,
        lib: PaperLibrary,
        paper_a: Paper,
    ) -> None:
        lib.add_paper(paper_a)
        lib.add_claims(
            [
                _claim("paper_a", "attention mechanism", "method"),
            ]
        )
        results = lib.search_claims_fts("attention")
        assert len(results) >= 1
        assert results[0]["claim_text"] == "attention mechanism"


# ── Claim CRUD Tests ──────────────────────────────────────────


class TestClaimCRUD:
    def test_add_and_get(
        self,
        lib: PaperLibrary,
        paper_a: Paper,
    ) -> None:
        lib.add_paper(paper_a)
        ids = lib.add_claims(
            [
                _claim("paper_a", "claim 1"),
                _claim("paper_a", "claim 2", "assumption"),
            ]
        )
        assert len(ids) == 2
        assert len(lib.get_claims_for_paper("paper_a")) == 2

    def test_duplicate_ignored(
        self,
        lib: PaperLibrary,
        paper_a: Paper,
    ) -> None:
        lib.add_paper(paper_a)
        lib.add_claims([_claim("paper_a", "same claim")])
        lib.add_claims([_claim("paper_a", "same claim")])
        assert len(lib.get_claims_for_paper("paper_a")) == 1

    def test_clear_atomic(
        self,
        lib: PaperLibrary,
        paper_a: Paper,
    ) -> None:
        lib.add_paper(paper_a)
        lib.add_claims(
            [
                _claim("paper_a", "c1"),
                _claim("paper_a", "c2"),
            ]
        )
        assert lib.clear_claims("paper_a") == 2
        assert lib.get_claims_for_paper("paper_a") == []

    def test_status_lifecycle(
        self,
        lib: PaperLibrary,
        paper_a: Paper,
    ) -> None:
        lib.add_paper(paper_a)
        assert lib.get_claims_status("paper_a") is None
        lib.set_claims_status("paper_a", "pending")
        assert lib.get_claims_status("paper_a") == "pending"
        lib.set_claims_status("paper_a", "complete")
        assert lib.get_claims_status("paper_a") == "complete"

    def test_full_text_storage(
        self,
        lib: PaperLibrary,
        paper_a: Paper,
    ) -> None:
        lib.add_paper(paper_a)
        assert lib.get_full_text("paper_a") is None
        lib.save_full_text("paper_a", "Full paper text here...")
        assert lib.get_full_text("paper_a") == "Full paper text here..."

    def test_fts_sanitization(
        self,
        lib: PaperLibrary,
        paper_a: Paper,
    ) -> None:
        lib.add_paper(paper_a)
        lib.add_claims(
            [
                _claim("paper_a", 'claim with "quotes" AND OR'),
            ]
        )
        results = lib.search_claims_fts('claim with "quotes"')
        assert isinstance(results, list)


# ── Alerts Tests ──────────────────────────────────────────────


class TestAlerts:
    def _setup_alert(
        self,
        lib: PaperLibrary,
        pa: Paper,
        pb: Paper,
    ) -> int:
        lib.add_paper(pa)
        lib.add_paper(pb)
        ids_a = lib.add_claims([_claim("paper_a", "claim A")])
        ids_b = lib.add_claims([_claim("paper_b", "claim B")])
        eid = lib.add_claim_edge(
            ids_a[0],
            ids_b[0],
            "contradicts",
            0.9,
            "disagree",
        )
        return lib.add_alert(eid, "contradicts", "A vs B", 0.9)

    def test_add_and_list(
        self,
        lib: PaperLibrary,
        paper_a: Paper,
        paper_b: Paper,
    ) -> None:
        self._setup_alert(lib, paper_a, paper_b)
        alerts = lib.list_alerts(unread_only=True)
        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "contradicts"

    def test_idempotency(
        self,
        lib: PaperLibrary,
        paper_a: Paper,
        paper_b: Paper,
    ) -> None:
        lib.add_paper(paper_a)
        lib.add_paper(paper_b)
        ids_a = lib.add_claims([_claim("paper_a", "claim A")])
        ids_b = lib.add_claims([_claim("paper_b", "claim B")])
        eid = lib.add_claim_edge(
            ids_a[0],
            ids_b[0],
            "contradicts",
            0.9,
        )
        first = lib.add_alert(eid, "contradicts", "A vs B", 0.9)
        second = lib.add_alert(eid, "contradicts", "dup", 0.9)
        assert first is not None
        assert second is None

    def test_update(
        self,
        lib: PaperLibrary,
        paper_a: Paper,
        paper_b: Paper,
    ) -> None:
        aid = self._setup_alert(lib, paper_a, paper_b)
        assert lib.update_alert(aid, is_read=1)
        assert len(lib.list_alerts(unread_only=True)) == 0
        all_a = lib.list_alerts(unread_only=False)
        assert len(all_a) == 1 and all_a[0]["is_read"] is True

    def test_useful_feedback(
        self,
        lib: PaperLibrary,
        paper_a: Paper,
        paper_b: Paper,
    ) -> None:
        aid = self._setup_alert(lib, paper_a, paper_b)
        lib.update_alert(aid, is_useful=1)
        alerts = lib.list_alerts(unread_only=False)
        assert alerts[0]["is_useful"] is True

    def test_papers_needing_claims(
        self,
        lib: PaperLibrary,
        paper_a: Paper,
    ) -> None:
        lib.add_paper(paper_a)
        assert len(lib.papers_needing_claims()) == 1
        lib.set_claims_status("paper_a", "complete")
        assert len(lib.papers_needing_claims()) == 0


# ── Pydantic Model Tests ─────────────────────────────────────


class TestModels:
    def test_claim_model(self) -> None:
        c = Claim(
            paper_id="p1",
            claim_text="test",
            claim_type="finding",
            source_quote="ev",
        )
        assert c.confidence == 1.0

    def test_claim_edge_model(self) -> None:
        e = ClaimEdge(
            source_claim_id=1,
            target_claim_id=2,
            relation="contradicts",
            confidence=0.85,
        )
        assert e.relation == "contradicts"

    def test_alert_model(self) -> None:
        a = Alert(
            claim_edge_id=1,
            alert_type="contradicts",
            summary="test",
            confidence=0.9,
        )
        assert a.is_read is False and a.is_useful is None

    def test_invalid_claim_type(self) -> None:
        with pytest.raises(Exception):
            Claim(paper_id="p1", claim_text="t", claim_type="bad")

    def test_invalid_relation(self) -> None:
        with pytest.raises(Exception):
            ClaimEdge(
                source_claim_id=1,
                target_claim_id=2,
                relation="bad",
                confidence=0.5,
            )


# ── Claim Extraction Tests (mocked Claude) ───────────────────

_PATCH_ASK = "openseed.agent.claims._ask_json"


class TestClaimExtraction:
    @patch(_PATCH_ASK)
    def test_happy_path(
        self,
        mock_ask: MagicMock,
        lib: PaperLibrary,
        paper_a: Paper,
    ) -> None:
        from openseed.agent.claims import extract_claims

        lib.add_paper(paper_a)
        mock_ask.return_value = SAMPLE_CLAIMS_A
        claims = extract_claims(
            "paper_a",
            "Full text...",
            "claude-sonnet-4-6",
            lib,
        )
        assert len(claims) == 2
        assert lib.get_claims_status("paper_a") == "complete"

    @patch(_PATCH_ASK)
    def test_short_paper(
        self,
        mock_ask: MagicMock,
        lib: PaperLibrary,
        paper_a: Paper,
    ) -> None:
        from openseed.agent.claims import extract_claims

        lib.add_paper(paper_a)
        mock_ask.return_value = [SAMPLE_CLAIMS_A[0]]
        claims = extract_claims(
            "paper_a",
            "Short abstract",
            "claude-sonnet-4-6",
            lib,
        )
        assert len(claims) == 1
        assert lib.get_claims_status("paper_a") == "complete"

    def test_no_text_raises(
        self,
        lib: PaperLibrary,
        paper_a: Paper,
    ) -> None:
        from openseed.agent.claims import extract_claims

        lib.add_paper(paper_a)
        with pytest.raises(ValueError, match="No text"):
            extract_claims("paper_a", "", "claude-sonnet-4-6", lib)

    @patch(_PATCH_ASK)
    def test_timeout_marks_failed(
        self,
        mock_ask: MagicMock,
        lib: PaperLibrary,
        paper_a: Paper,
    ) -> None:
        from openseed.agent.claims import extract_claims

        lib.add_paper(paper_a)
        mock_ask.side_effect = TimeoutError("timeout")
        with pytest.raises(TimeoutError):
            extract_claims("paper_a", "text", "claude-sonnet-4-6", lib)
        assert lib.get_claims_status("paper_a") == "failed"

    @patch(_PATCH_ASK)
    def test_atomic_replacement(
        self,
        mock_ask: MagicMock,
        lib: PaperLibrary,
        paper_a: Paper,
    ) -> None:
        from openseed.agent.claims import extract_claims

        lib.add_paper(paper_a)
        mock_ask.return_value = SAMPLE_CLAIMS_A
        extract_claims("paper_a", "text", "claude-sonnet-4-6", lib)
        assert len(lib.get_claims_for_paper("paper_a")) == 2
        mock_ask.return_value = [SAMPLE_CLAIMS_A[0]]
        extract_claims("paper_a", "text", "claude-sonnet-4-6", lib)
        assert len(lib.get_claims_for_paper("paper_a")) == 1

    @patch(_PATCH_ASK)
    def test_evidence_quotes(
        self,
        mock_ask: MagicMock,
        lib: PaperLibrary,
        paper_a: Paper,
    ) -> None:
        from openseed.agent.claims import extract_claims

        lib.add_paper(paper_a)
        mock_ask.return_value = SAMPLE_CLAIMS_A
        extract_claims("paper_a", "text", "claude-sonnet-4-6", lib)
        claims = lib.get_claims_for_paper("paper_a")
        assert claims[0]["source_quote"] is not None


# ── Matcher Tests (mocked Claude) ────────────────────────────

_PATCH_MATCH = "openseed.agent.matcher._ask_json"


class TestMatcher:
    @patch(_PATCH_MATCH)
    def test_happy_path(
        self,
        mock_ask: MagicMock,
        lib: PaperLibrary,
        paper_a: Paper,
        paper_b: Paper,
    ) -> None:
        from openseed.agent.matcher import match_claims

        lib.add_paper(paper_a)
        lib.add_paper(paper_b)
        lib.add_claims(
            [
                _claim("paper_a", "transformer models achieve SOTA"),
            ]
        )
        lib.add_claims(
            [
                _claim("paper_b", "transformer models fail on long"),
            ]
        )
        mock_ask.return_value = [
            {
                "target_index": 0,
                "relation": "contradicts",
                "confidence": 0.85,
                "reasoning": "disagree",
            }
        ]
        edges, alerts = match_claims(
            "paper_a",
            "claude-sonnet-4-6",
            lib,
        )
        assert edges >= 1 and alerts >= 1

    def test_zero_candidates(
        self,
        lib: PaperLibrary,
        paper_a: Paper,
    ) -> None:
        from openseed.agent.matcher import match_claims

        lib.add_paper(paper_a)
        lib.add_claims(
            [
                _claim("paper_a", "zyxwvutsrq completely unique"),
            ]
        )
        edges, alerts = match_claims(
            "paper_a",
            "claude-sonnet-4-6",
            lib,
        )
        assert edges == 0 and alerts == 0

    @patch(_PATCH_MATCH)
    def test_supports_no_alert(
        self,
        mock_ask: MagicMock,
        lib: PaperLibrary,
        paper_a: Paper,
        paper_b: Paper,
    ) -> None:
        from openseed.agent.matcher import match_claims

        lib.add_paper(paper_a)
        lib.add_paper(paper_b)
        lib.add_claims(
            [
                _claim("paper_a", "attention mechanism accuracy"),
            ]
        )
        lib.add_claims(
            [
                _claim("paper_b", "attention mechanism training"),
            ]
        )
        mock_ask.return_value = [
            {
                "target_index": 0,
                "relation": "supports",
                "confidence": 0.95,
                "reasoning": "agrees",
            }
        ]
        edges, alerts = match_claims(
            "paper_a",
            "claude-sonnet-4-6",
            lib,
        )
        assert edges >= 1 and alerts == 0

    @patch(_PATCH_MATCH)
    def test_low_confidence_no_alert(
        self,
        mock_ask: MagicMock,
        lib: PaperLibrary,
        paper_a: Paper,
        paper_b: Paper,
    ) -> None:
        from openseed.agent.matcher import match_claims

        lib.add_paper(paper_a)
        lib.add_paper(paper_b)
        lib.add_claims(
            [
                _claim("paper_a", "neural network performance"),
            ]
        )
        lib.add_claims(
            [
                _claim("paper_b", "neural network performance bad"),
            ]
        )
        mock_ask.return_value = [
            {
                "target_index": 0,
                "relation": "contradicts",
                "confidence": 0.3,
                "reasoning": "weak",
            }
        ]
        edges, alerts = match_claims(
            "paper_a",
            "claude-sonnet-4-6",
            lib,
        )
        assert edges >= 1 and alerts == 0

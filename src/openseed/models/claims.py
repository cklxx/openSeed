"""Pydantic models for claim extraction and cross-paper matching."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Claim(BaseModel):
    """An atomic, falsifiable assertion extracted from a paper."""

    id: int | None = None
    paper_id: str
    claim_text: str
    claim_type: Literal["finding", "assumption", "method", "limitation"]
    section: str | None = None
    source_quote: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ClaimEdge(BaseModel):
    """A relationship between two claims from different papers."""

    id: int | None = None
    source_claim_id: int
    target_claim_id: int
    relation: Literal["supports", "contradicts", "refines", "extends"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str | None = None


class Alert(BaseModel):
    """A surfaced insight for the user from a claim edge."""

    id: int | None = None
    claim_edge_id: int
    alert_type: Literal["contradicts", "refines"]
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    is_read: bool = False
    is_dismissed: bool = False
    is_useful: bool | None = None

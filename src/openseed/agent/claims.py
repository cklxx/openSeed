"""Claim extraction from papers via Claude."""

from __future__ import annotations

import logging
import time

from pydantic import ValidationError

from openseed.agent.reader import _ask_json
from openseed.models.claims import Claim
from openseed.storage.library import PaperLibrary

logger = logging.getLogger(__name__)

_EXTRACT_SYSTEM = (
    "You are a research claim extractor. Given paper text, extract atomic claims. "
    "Each claim is ONE falsifiable assertion — no conjunctions. "
    "Return a JSON array of objects with keys: "
    '"claim_text" (self-contained statement), '
    '"claim_type" ("finding"|"assumption"|"method"|"limitation"), '
    '"section" (paper section name or null), '
    '"source_quote" (verbatim excerpt supporting this claim, max 200 chars). '
    "Target 8-12 claims for a full paper, fewer for short papers. "
    "Return ONLY a valid JSON array, no markdown fences."
)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0


def _extract_raw(model: str, text: str) -> list[dict]:
    """Call Claude to extract claims. Retries on rate-limit."""
    prompt = f"Extract atomic claims from this paper:\n\n{text[:15000]}"
    for attempt in range(_MAX_RETRIES):
        try:
            result = _ask_json(model, _EXTRACT_SYSTEM, prompt)
            if isinstance(result, list):
                return result
            return []
        except Exception as exc:
            if "rate" in str(exc).lower() and attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_BASE ** (attempt + 1))
                continue
            raise
    return []


def _validate_claims(raw: list[dict], paper_id: str) -> list[Claim]:
    """Validate raw dicts into Claim models, skipping invalid ones."""
    claims: list[Claim] = []
    for item in raw:
        try:
            claims.append(Claim(paper_id=paper_id, **item))
        except (ValidationError, TypeError):
            continue
    return claims


def extract_claims(
    paper_id: str,
    text: str,
    model: str,
    library: PaperLibrary,
) -> list[Claim]:
    """Extract claims from paper text and store them. Atomic replacement."""
    if not text or not text.strip():
        raise ValueError("No text available for claim extraction")

    library.set_claims_status(paper_id, "pending")
    try:
        raw = _extract_raw(model, text)
        claims = _validate_claims(raw, paper_id)
        library.clear_claims(paper_id)
        if claims:
            library.add_claims([c.model_dump(exclude={"id"}) for c in claims])
        library.set_claims_status(paper_id, "complete")
        logger.info("Extracted %d claims for paper %s", len(claims), paper_id)
        return claims
    except Exception:
        library.set_claims_status(paper_id, "failed")
        logger.exception("Claim extraction failed for paper %s", paper_id)
        raise


def get_paper_text(paper_id: str, library: PaperLibrary) -> str | None:
    """Get best available text for a paper: full_text > abstract."""
    full = library.get_full_text(paper_id)
    if full:
        return full
    paper = library.get_paper(paper_id)
    if paper and paper.abstract:
        return paper.abstract
    return None

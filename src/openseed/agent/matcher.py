"""Cross-paper claim matching and alert generation via Claude."""

from __future__ import annotations

import logging

from pydantic import ValidationError

from openseed.agent.reader import _ask_json
from openseed.models.claims import ClaimEdge
from openseed.storage.library import PaperLibrary

logger = logging.getLogger(__name__)

_ALERT_TYPES = {"contradicts", "refines"}
_CONFIDENCE_THRESHOLD = 0.7
_FTS_CANDIDATES_PER_CLAIM = 10
_BATCH_SIZE = 5

_CLASSIFY_SYSTEM = (
    "You are a research claim relationship classifier. Given a source claim and "
    "candidate target claims from different papers, classify each relationship. "
    "Return a JSON array of objects with keys: "
    '"target_index" (0-based index of the candidate), '
    '"relation" ("supports"|"contradicts"|"refines"|"extends"|"irrelevant"), '
    '"confidence" (0.0-1.0), '
    '"reasoning" (one sentence explaining the relationship). '
    "Return ONLY a valid JSON array, no markdown fences."
)


def _find_candidates(
    claim: dict,
    paper_id: str,
    library: PaperLibrary,
) -> list[dict]:
    """Find FTS5 candidates for a claim, excluding same-paper matches."""
    results = library.search_claims_fts(claim["claim_text"], limit=_FTS_CANDIDATES_PER_CLAIM * 2)
    seen_papers: set[str] = set()
    candidates: list[dict] = []
    for r in results:
        if r["paper_id"] == paper_id or r["paper_id"] in seen_papers:
            continue
        seen_papers.add(r["paper_id"])
        candidates.append(r)
        if len(candidates) >= _FTS_CANDIDATES_PER_CLAIM:
            break
    return candidates


def _classify_batch(
    model: str,
    source: dict,
    targets: list[dict],
) -> list[dict]:
    """Classify relationships between a source claim and target claims."""
    target_lines = "\n".join(f"  [{i}] {t['claim_text']}" for i, t in enumerate(targets))
    prompt = f"Source claim: {source['claim_text']}\n\nCandidate target claims:\n{target_lines}"
    try:
        raw = _ask_json(model, _CLASSIFY_SYSTEM, prompt)
        return raw if isinstance(raw, list) else []
    except (ValueError, Exception):
        logger.warning("Classification failed for claim %s", source.get("id"))
        return []


def _store_edges(
    source_id: int,
    targets: list[dict],
    classifications: list[dict],
    library: PaperLibrary,
) -> list[int]:
    """Store classified edges and return edge IDs."""
    edge_ids: list[int] = []
    for c in classifications:
        idx = c.get("target_index")
        if idx is None or idx < 0 or idx >= len(targets):
            continue
        relation = c.get("relation", "irrelevant")
        if relation in ("irrelevant", "duplicate"):
            continue
        confidence = c.get("confidence", 0.0)
        try:
            ClaimEdge(
                source_claim_id=source_id,
                target_claim_id=targets[idx]["id"],
                relation=relation,
                confidence=confidence,
            )
        except ValidationError:
            continue
        eid = library.add_claim_edge(
            source_id,
            targets[idx]["id"],
            relation,
            confidence,
            c.get("reasoning"),
        )
        if eid:
            edge_ids.append(eid)
    return edge_ids


def _generate_alerts(
    edge_ids: list[int],
    classifications: list[dict],
    targets: list[dict],
    source: dict,
    library: PaperLibrary,
) -> int:
    """Create alerts for high-confidence contradictions/refinements."""
    alert_count = 0
    for c in classifications:
        idx = c.get("target_index")
        if idx is None or idx < 0 or idx >= len(targets):
            continue
        relation = c.get("relation", "irrelevant")
        confidence = c.get("confidence", 0.0)
        if relation not in _ALERT_TYPES or confidence < _CONFIDENCE_THRESHOLD:
            continue
        summary = (
            f'{relation.title()}: "{source["claim_text"][:80]}" vs '
            f'"{targets[idx]["claim_text"][:80]}"'
        )
        src_id = source["id"]
        tgt_id = targets[idx]["id"]
        edge = library._conn.execute(
            "SELECT id FROM claim_edges WHERE source_claim_id = ? AND target_claim_id = ?",
            (src_id, tgt_id),
        ).fetchone()
        if edge:
            added = library.add_alert(edge[0], relation, summary, confidence)
            if added:
                alert_count += 1
    return alert_count


def match_claims(
    paper_id: str,
    model: str,
    library: PaperLibrary,
) -> tuple[int, int]:
    """Match a paper's claims against the library. Returns (edges, alerts)."""
    claims = library.get_claims_for_paper(paper_id)
    if not claims:
        return 0, 0

    total_edges = 0
    total_alerts = 0

    for claim in claims:
        candidates = _find_candidates(claim, paper_id, library)
        if not candidates:
            continue

        for i in range(0, len(candidates), _BATCH_SIZE):
            batch = candidates[i : i + _BATCH_SIZE]
            results = _classify_batch(model, claim, batch)
            edge_ids = _store_edges(claim["id"], batch, results, library)
            total_edges += len(edge_ids)
            alerts = _generate_alerts(edge_ids, results, batch, claim, library)
            total_alerts += alerts

    logger.info("Matched paper %s: %d edges, %d alerts", paper_id, total_edges, total_alerts)
    return total_edges, total_alerts

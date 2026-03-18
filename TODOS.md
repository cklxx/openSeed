# TODOS

## Phase 2

### Web Dashboard (FastAPI + htmx)
- **What:** Local web UI for browsing library, viewing summaries, exploring knowledge graph visually
- **Why:** Knowledge graph is 10x more useful when visual; makes OpenSeed shareable with colleagues
- **Pros:** Transforms OpenSeed from power-user CLI to something you demo; graph visualization is the killer feature
- **Cons:** New dependency surface (FastAPI, Jinja2, htmx); maintenance of two interfaces
- **Context:** SQLite + knowledge graph from v0.6.0 provide the data layer. Server-rendered HTML with htmx for interactivity — not a full SPA. CLI remains primary; web is the read-heavy companion.
- **Effort:** L (human: ~2 weeks) / M (CC: ~1 hour)
- **Priority:** P2
- **Depends on:** SQLite migration (v0.6.0), knowledge graph (v0.6.0)

### Reader Module Split
- **What:** Split `agent/reader.py` into focused modules (discovery.py, analysis.py, etc.)
- **Why:** reader.py accumulates responsibilities across releases (discovery, enrichment, tagging, summarization, visuals, synthesis, graph extraction). After v0.8.0 it'll approach 400 lines with 6+ concerns.
- **Pros:** Each module has a single responsibility; easier to navigate and test in isolation
- **Cons:** More files; imports change across the codebase
- **Context:** Individual functions stay within the 15-line limit so it's not urgent. But the module-level cohesion is low — discovery and LaTeX export have nothing in common. Start by extracting `_fetch_citations`/`enrich_citations` to `scholar.py` (already planned for v0.6.0), then evaluate what remains.
- **Effort:** M (human: ~2 days) / S (CC: ~15 min)
- **Priority:** P3
- **Depends on:** v0.8.0 landing (so all new code is in place before reorganizing)

## Phase 3

### Multi-User Research Sessions
- **What:** Share research sessions with collaborators — shared library views, collaborative annotations, synthesized perspectives
- **Why:** Research is social. A single-user tool caps at individual productivity. Team features unlock network effects.
- **Pros:** Differentiates OpenSeed from personal note tools; enables research groups
- **Cons:** Requires sync infrastructure (CRDTs or cloud SQLite like Turso); auth/permissions complexity
- **Context:** Builds on web dashboard. Sync layer is the hard part — evaluate Turso, Litestream, or CRDT-based approaches.
- **Effort:** XL (human: ~1 month) / L (CC: ~2 hours)
- **Priority:** P3
- **Depends on:** Web dashboard (Phase 2), sync layer design

### Multi-Source Discovery (Twitter/X, Conferences, RSS)
- **What:** Watch not just ArXiv but also Twitter academic feeds, conference proceedings, and RSS sources for new papers
- **Why:** Researchers discover papers through many channels — ArXiv is necessary but not sufficient
- **Pros:** Broader coverage; catches papers before they hit ArXiv; conference proceedings
- **Cons:** Each source has different APIs, rate limits, and data formats; Twitter API is expensive
- **Context:** Builds on scheduled watching infrastructure from v0.7.0. Start with RSS (simple), then conference feeds, then Twitter.
- **Effort:** L (human: ~1 week) / M (CC: ~45 min)
- **Priority:** P3
- **Depends on:** Scheduled watching (v0.7.0)

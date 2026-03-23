# TODOS

## Phase 2.5 — Deferred from Smart Research Agent (Phase 2)

- **Auto-Literature Review from Graph Clusters** — P2, M (CC: ~S).
  Takes a knowledge graph cluster and generates a structured literature review (themes,
  methodology evolution, open questions) with LaTeX/BibTeX output. Builds on synthesize_papers
  + latex.py + graph clusters. Depends on Phase 2 context engine for better paper retrieval.

- **Local Embeddings for Semantic Search** — P2, M (CC: ~S).
  Embed paper abstracts/summaries using a local model and store vectors in SQLite. Enables
  conceptual similarity search when FTS5 keyword matching fails (different terminology for
  same concepts). Depends on Phase 2 context engine. Adds ~500MB dependency.

- **Smart Watch Suggestions from Research Themes** — P3, S (CC: ~S).
  Agent analyzes research memory + library to auto-suggest watch queries based on research
  interests. Depends on Phase 2 memory module being populated with research themes.

## Completed (v1.0)

## Completed

- **Web Dashboard (FastAPI + htmx)** — v0.9.0. Pages: dashboard, papers, paper detail, graph, digests, sessions. `openseed web` command.
- **Reader Module Split** — v0.9.0. Extracted discovery.py from reader.py.
- **Multi-Source Discovery (RSS)** — v0.9.0. RSS/Atom feed support via `watch add-rss`. services/rss.py.
- **Multi-User Research Sessions** — v0.9.0. Session export/import via `research export` / `research import`. services/sharing.py.

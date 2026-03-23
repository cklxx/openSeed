# OpenSeed

AI-powered research workflow CLI — discover, read, analyze, and synthesize academic papers with Claude.

## What it does

- **Smart search** — finds papers via Claude + ArXiv, ranks by real citation counts (Semantic Scholar)
- **Autonomous research** — multi-round discovery → analysis → synthesis pipeline that writes a full report
- **AI summarization** — structured summaries: contributions, methodology, limitations, relevance score
- **Paper comparison** — side-by-side analysis of methodology, assumptions, and contradictions
- **Strategy analysis** — gap detection across your library, reading recommendations
- **Experiment code gen** — generate runnable PyTorch/sklearn code from a paper
- **MCP server** — expose your library as tools for Claude Code / Claude Desktop
- **Web dashboard** — FastAPI-based UI for browsing your library

## Search

![openseed paper search "attention"](screenshot-20260317-144253.png)

Papers are ranked by real citation counts from Semantic Scholar.

## Install

```bash
pip install openseed

openseed doctor    # check environment
openseed setup     # configure auth + model
```

**Auth** — any of these work:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # Anthropic API key
claude setup-token                     # OAuth (openseed setup will detect it)
```

## Commands

```bash
# Search & discover
openseed paper search "diffusion models" --count 20
openseed agent search "multi-agent systems"

# Autonomous research (discover → analyze → synthesize → report)
openseed research run "ViT image classification"

# Manage your library
openseed paper add https://arxiv.org/abs/1706.03762
openseed paper list
openseed paper show <id>

# Analyze papers
openseed agent summarize <id>          # English summary
openseed agent summarize <id> --cn     # Chinese summary
openseed agent review <id>             # peer review
openseed agent compare <id1> <id2>     # side-by-side comparison
openseed agent ask "What is RLHF?"     # research Q&A
openseed agent codegen <id>            # generate experiment code

# MCP server (for Claude Code / Claude Desktop)
openseed mcp                           # start MCP server
```

## Architecture

```
src/openseed/
├── cli/             # Click CLI: paper, agent, research, experiment
├── agent/           # AI-powered analysis
│   ├── reader.py        # structured summarize / analyze
│   ├── assistant.py     # freeform research Q&A
│   ├── discovery.py     # paper discovery via Claude + S2
│   ├── compare.py       # side-by-side paper comparison
│   ├── autoresearch.py  # autonomous multi-round research engine
│   ├── strategy.py      # gap analysis + reading recommendations
│   ├── memory.py        # FTS5-backed conversation history
│   ├── context.py       # library-aware prompt assembly
│   └── latex.py         # related-work export with BibTeX
├── services/        # External integrations
│   ├── arxiv.py         # ArXiv metadata fetch + search
│   ├── scholar.py       # Semantic Scholar API (citations, refs)
│   ├── pdf.py           # PDF text extraction (PyMuPDF)
│   ├── rss.py           # RSS/Atom feed discovery
│   ├── watch.py         # watch execution service
│   ├── cron.py          # crontab management
│   ├── digest.py        # digest generation
│   └── sharing.py       # session export/import
├── storage/         # Persistence
│   ├── library.py       # SQLite CRUD + knowledge graph
│   ├── pool.py          # thread-safe connection pool (WAL)
│   └── migrate.py       # JSON → SQLite migration
├── models/          # Pydantic v2 models
├── mcp/             # MCP server (library as Claude tools)
├── web/             # FastAPI dashboard
├── auth.py          # Anthropic client factory
├── config.py        # paths, default model
└── doctor.py        # environment health checks
```

## License

MIT

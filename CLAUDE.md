# OpenSeed — Claude Code Instructions

## STOP — Read this first

- Priority: **correctness > simplicity > speed**
- Before any code change: `git diff --stat` + `git log --oneline -5`
- On any user correction: codify a rule before resuming work

---

## About

OpenSeed is an AI-powered research workflow CLI. It manages a local paper library (ArXiv fetch, PDF extraction) and provides Claude-powered summarization, review, and Q&A.

---

## Code Style (MANDATORY)

- Max function body: **15 lines**. Extract or redesign if exceeded.
- No comments that restate code. Only "why" comments for non-obvious decisions.
- Prefer composition over inheritance. Prefer data transforms over mutation.
- Every abstraction must justify itself: used <2 places → inline it.
- No TODOs in committed code. Delete dead code paths immediately.
- Type signatures are documentation. Verbose names > comments.
- When two approaches are equally correct, pick the one with fewer moving parts.

---

## Non-negotiable standards

- Lint: `ruff check src/ tests/` and `ruff format src/ tests/` must pass before commit.
- Tests: `pytest tests/ -v` must pass before commit.
- Line length: 100 chars max.
- Python 3.11+ — use `X | Y` union syntax, not `Union[X, Y]`.
- Delete dead code outright. No `# deprecated` or commented-out blocks.
- Modify only files relevant to the task.

---

## Architecture

- **Runtime**: Python 3.11+
- **AI**: `anthropic` SDK — client via `auth.make_anthropic_client()` (supports both `ANTHROPIC_API_KEY` and `CLAUDE_CODE_SETUP_TOKEN`)
- **CLI**: Click + Rich
- **Models**: Pydantic v2
- **Layout**: src-layout (`src/openseed/`)

## Module Map

| Module | Purpose |
|---|---|
| `cli/` | Click groups: `paper`, `experiment`, `agent` |
| `models/` | Pydantic models: Paper, Author, Tag, Experiment, ExperimentRun |
| `storage/library.py` | JSON-backed CRUD for papers + experiments |
| `services/arxiv.py` | ArXiv metadata fetch + search (sync + async) |
| `services/pdf.py` | PDF text extraction via PyMuPDF |
| `agent/reader.py` | PaperReader — structured summarize/analyze via Claude |
| `agent/assistant.py` | ResearchAssistant — freeform ask/review via Claude |
| `auth.py` | `make_anthropic_client()`, `has_anthropic_auth()`, `run_claude_setup_token()` |
| `doctor.py` | Environment health checks with `CheckResult` + fix hints |
| `config.py` | `OpenSeedConfig`, paths, default model |

---

## Conditional context loading

Only read when the trigger matches — do not bulk-load.

| Trigger | Read |
|---|---|
| ArXiv fetch / search issues | `src/openseed/services/arxiv.py` |
| PDF extraction issues | `src/openseed/services/pdf.py` |
| Agent AI features | `src/openseed/agent/reader.py`, `assistant.py` |
| Auth / API key issues | `src/openseed/auth.py` |
| Storage / data bugs | `src/openseed/storage/library.py` |
| CLI command issues | `src/openseed/cli/<command>.py` |
| Config / paths | `src/openseed/config.py` |

---

## Common Commands

```bash
make install          # pip install -e ".[dev]"
make test             # pytest -v
make lint             # ruff check src/ tests/
make format           # ruff format src/ tests/
openseed setup        # Configure auth + model
openseed doctor       # Environment health check
openseed paper add    # Add paper by ArXiv URL
openseed paper list   # List library
openseed agent ask    # Ask research question
```

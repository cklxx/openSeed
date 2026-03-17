# OpenSeed — Claude Code Config

## Architecture

- **Runtime**: Python 3.11+
- **AI**: Anthropic SDK (`anthropic`), Claude Agent SDK (`claude-agent-sdk`)
- **CLI**: Click + Rich
- **Models**: Pydantic v2
- **Layout**: src-layout (`src/openseed/`)

## Module Map

| Module | Purpose |
|---|---|
| `cli/` | Click command groups: `paper`, `experiment`, `agent` |
| `models/` | Pydantic models: Paper, Author, Annotation, Tag, Experiment, ExperimentRun |
| `storage/` | JSON-based library CRUD (papers, experiments) |
| `services/` | ArXiv metadata fetch, PDF text extraction |
| `agent/` | Claude-powered paper reader and research assistant |
| `config.py` | OpenSeedConfig, paths, model defaults |

## Conventions

- **src-layout**: all source under `src/openseed/`
- **Type hints**: all public functions fully typed
- **Docstrings**: Google style
- **Output**: Rich tables, panels, markdown rendering
- **Linting**: ruff (line-length 100, target py311)
- **Testing**: pytest with fixtures in `conftest.py`

## Priority

Safety > Correctness > Maintainability > Performance

## Common Commands

```bash
make install          # pip install -e ".[dev]"
make test             # pytest -v
make lint             # ruff check src/ tests/
make format           # ruff format src/ tests/
openseed paper add    # Add paper by URL
openseed paper list   # List library
openseed agent ask    # Ask research question
```

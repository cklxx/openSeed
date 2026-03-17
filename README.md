# OpenSeed

AI-powered Research Workflow Management CLI.

## Features

- **Paper Management** — Import from ArXiv, tag, search, track reading status
- **AI Reading** — Claude-powered summaries, key findings extraction, question generation
- **Annotations** — Per-page notes with tags
- **Experiment Tracking** — Link experiments to papers, track runs and metrics
- **Research Assistant** — Ask questions about papers, get AI-driven reviews

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Initialize library
openseed init

# Add a paper from ArXiv
openseed paper add https://arxiv.org/abs/2301.00001

# List your library
openseed paper list

# Summarize a paper with AI
openseed agent summarize <paper-id>

# Ask a research question
openseed agent ask "What are the key contributions of this paper?"
```

## Development

```bash
make install      # Install in editable mode with dev deps
make test         # Run tests
make lint         # Lint with ruff
make format       # Auto-format with ruff
make typecheck    # Run mypy
make clean        # Remove build artifacts
```

## Architecture

src-layout with Click CLI, Pydantic v2 models, JSON storage, and Claude AI integration.

```
src/openseed/
├── cli/          # Click command groups
├── models/       # Pydantic data models
├── storage/      # JSON library CRUD
├── services/     # ArXiv API, PDF extraction
├── agent/        # Claude-powered reader & assistant
└── config.py     # Configuration
```

## License

MIT

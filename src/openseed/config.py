"""OpenSeed configuration."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import toml
from pydantic import BaseModel, Field


def _config_dir() -> Path:
    """Return config base dir, overridable via OPENSEED_CONFIG_DIR env var."""
    env = os.environ.get("OPENSEED_CONFIG_DIR")
    return Path(env) if env else Path.home() / ".openseed"


class OpenSeedConfig(BaseModel):
    """Global configuration."""

    library_dir: Path = Field(default_factory=lambda: _config_dir() / "library")
    config_dir: Path = Field(default_factory=_config_dir)
    default_model: str = "claude-sonnet-4-6"


def _config_path() -> Path:
    return _config_dir() / "config.toml"


def load_config() -> OpenSeedConfig:
    """Load config from disk, returning defaults if missing."""
    base = _config_dir()
    path = base / "config.toml"
    if path.exists():
        data = tomllib.loads(path.read_text())
        return OpenSeedConfig(**data)
    return OpenSeedConfig(library_dir=base / "library", config_dir=base)


def save_config(config: OpenSeedConfig) -> None:
    """Persist config to disk."""
    path = _config_path()
    ensure_dirs(config)
    path.write_text(toml.dumps(config.model_dump(mode="json")))


def ensure_dirs(config: OpenSeedConfig | None = None) -> None:
    """Create required directories."""
    cfg = config or OpenSeedConfig()
    cfg.config_dir.mkdir(parents=True, exist_ok=True)
    cfg.library_dir.mkdir(parents=True, exist_ok=True)

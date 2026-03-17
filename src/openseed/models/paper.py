"""Paper-related data models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class Author(BaseModel):
    """Paper author."""

    name: str
    affiliation: str | None = None
    email: str | None = None


class Tag(BaseModel):
    """Classification tag."""

    name: str
    color: str = "blue"


class Annotation(BaseModel):
    """Paper annotation with optional page reference."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    page: int | None = None
    text: str
    note: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: list[Tag] = Field(default_factory=list)


class Paper(BaseModel):
    """Research paper entry in the library."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str
    authors: list[Author] = Field(default_factory=list)
    abstract: str = ""
    arxiv_id: str | None = None
    url: str | None = None
    pdf_path: str | None = None
    tags: list[Tag] = Field(default_factory=list)
    annotations: list[Annotation] = Field(default_factory=list)
    added_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    summary: str | None = None
    status: Literal["unread", "reading", "read", "archived"] = "unread"

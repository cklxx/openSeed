"""Research assistant powered by claude-agent-sdk."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openseed.agent.reader import _ask

if TYPE_CHECKING:
    from openseed.models.paper import Paper


class ResearchAssistant:
    """General-purpose research assistant using Claude via claude-agent-sdk."""

    def __init__(self, model: str = "claude-opus-4-6") -> None:
        self._model = model

    def ask(self, question: str, context: str = "") -> str:
        """Ask a research question."""
        prompt = f"Context:\n{context}\n\nQuestion: {question}" if context else question
        return _ask(self._model, "You are a knowledgeable research assistant.", prompt)

    def review_paper(self, paper: Paper) -> str:
        """Generate a constructive peer review of a paper."""
        text = f"Title: {paper.title}\n"
        if paper.authors:
            text += f"Authors: {', '.join(a.name for a in paper.authors)}\n"
        if paper.abstract:
            text += f"\nAbstract:\n{paper.abstract}\n"
        if paper.summary:
            text += f"\nSummary:\n{paper.summary}\n"
        return _ask(
            self._model,
            "You are a peer reviewer. Provide a constructive review.",
            f"Review this paper:\n\n{text}",
        )

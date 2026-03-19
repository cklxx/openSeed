"""Claude-powered paper reader via claude-agent-sdk."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import re
from collections.abc import Callable

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
from claude_agent_sdk.types import AssistantMessage, ToolUseBlock


def _make_opts(model: str, system: str) -> ClaudeAgentOptions:
    opts = ClaudeAgentOptions(
        system_prompt=system,
        disallowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Agent"],
        permission_mode="bypassPermissions",
    )
    opts.model = model
    return opts


def _tool_label(block: ToolUseBlock) -> str:
    """Return a short human-readable label for a tool call."""
    name = block.name
    inp = block.input
    if name == "WebSearch":
        return f"WebSearch: {inp.get('query', '')[:60]}"
    if name == "WebFetch":
        url = inp.get("url", "")
        return f"WebFetch: {url[:60]}"
    return name


async def _ask_async(
    model: str,
    system: str,
    prompt: str,
    on_step: Callable[[str], None] | None = None,
    on_result: Callable[[object], None] | None = None,
) -> str:
    result = ""
    async for msg in query(prompt=prompt, options=_make_opts(model, system)):
        if isinstance(msg, AssistantMessage) and on_step:
            for block in msg.content:
                if isinstance(block, ToolUseBlock):
                    on_step(_tool_label(block))
        if isinstance(msg, ResultMessage):
            result = msg.result or ""
            if on_result:
                on_result(msg)
    return result


def _ask(
    model: str,
    system: str,
    prompt: str,
    on_step: Callable[[str], None] | None = None,
    on_result: Callable[[object], None] | None = None,
) -> str:
    # Run in a dedicated thread so this always gets a fresh event loop,
    # safe to call from both sync and async contexts (no nested-loop error).
    def _silence_cancel_scope(loop: asyncio.AbstractEventLoop, ctx: dict) -> None:
        exc = ctx.get("exception")
        if isinstance(exc, RuntimeError) and "cancel scope" in str(exc):
            return
        loop.default_exception_handler(ctx)

    def _run() -> str:
        loop = asyncio.new_event_loop()
        loop.set_exception_handler(_silence_cancel_scope)
        try:
            return loop.run_until_complete(_ask_async(model, system, prompt, on_step, on_result))
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_run).result()


def auto_tag_paper(
    text: str, model: str, on_step: Callable[[str], None] | None = None
) -> list[str]:
    """Generate 3-5 concise tags for a paper."""
    result = _ask(
        model,
        "You are a research taxonomy expert. Return ONLY 3-5 comma-separated lowercase tags "
        "(single words or short hyphenated phrases). No explanations, no numbering.",
        f"Generate tags for:\n\n{text}",
    )
    return [t.strip().lower() for t in result.split(",") if t.strip()][:5]


def generate_experiment_code(text: str, model: str) -> str:
    """Generate runnable Python experiment code based on paper content."""
    system = (
        "You are a research engineer. Based on the paper provided, write clean runnable Python "
        "experiment code implementing the core methodology. "
        "Use PyTorch or scikit-learn as appropriate. "
        "Include: imports, dataset stub, model definition, training loop, and evaluation. "
        "Add brief comments linking code to key paper concepts."
    )
    return _ask(model, system, f"Generate experiment code for:\n\n{text}")


def synthesize_papers(texts: list[str], model: str) -> str:
    """Compare multiple papers: shared themes, methodology, differences, synthesis."""
    system = (
        "You are a research synthesis expert. Compare the provided papers and output markdown: "
        "## Shared Themes, ## Methodology Comparison, ## Key Differences, ## Synthesis."
    )
    body = "\n\n---\n\n".join(f"Paper {i + 1}:\n{t}" for i, t in enumerate(texts))
    return _ask(model, system, f"Synthesize these papers:\n\n{body}")


def extract_paper_visuals(text: str, model: str) -> dict:
    """Ask LLM to extract pipeline steps and metrics comparisons as JSON."""
    system = (
        "Extract from this paper into JSON with optional keys: "
        '"pipeline" (list of ≤6 concise method step names), '
        '"metrics" (list of {"name": str, "proposed": number, "baseline": number}). '
        "Return ONLY valid JSON, no markdown fences. Omit keys with no data."
    )
    raw = _ask(model, system, f"Extract visuals:\n\n{text}")
    raw = re.sub(r"```[a-z]*\n?", "", raw).strip().strip("```")
    try:
        return json.loads(raw)
    except Exception:
        return {}


def extract_references(text: str, model: str) -> list[str]:
    """Extract ArXiv IDs of papers referenced in the given text via AI."""
    system = (
        "You are a citation extraction expert. From the paper text below, identify all "
        "referenced papers that have ArXiv IDs. Return ONLY a comma-separated list of "
        "ArXiv IDs (format: YYMM.NNNNN). If no ArXiv IDs found, return 'NONE'."
    )
    raw = _ask(model, system, f"Extract ArXiv references from:\n\n{text[:8000]}")
    if "NONE" in raw.upper() or not raw.strip():
        return []
    ids = []
    for part in re.findall(r"\d{4}\.\d{4,5}", raw):
        if part not in ids:
            ids.append(part)
    return ids


class PaperReader:
    """Reads and analyzes papers using Claude via claude-agent-sdk."""

    def __init__(self, model: str = "claude-opus-4-6") -> None:
        self._model = model

    def summarize_paper(
        self, text: str, cn: bool = False, on_step: Callable[[str], None] | None = None
    ) -> str:
        """Generate a structured markdown summary of a paper."""
        if cn:
            system = (
                "You are a research paper summarizer. Respond entirely in Chinese markdown: "
                "一句话概括, ## 核心贡献 (bullets), ## 方法论, ## 局限性, **相关性评分:** N/10."
            )
        else:
            system = (
                "You are a research paper summarizer. Respond in markdown with sections: "
                "one-liner, ## Key Contributions (bullets), ## Methodology, "
                "## Limitations, **Relevance Score:** N/10."
            )
        return _ask(self._model, system, f"Summarize this paper:\n\n{text}", on_step=on_step)

    def extract_key_findings(self, text: str) -> list[str]:
        """Extract key findings as a list."""
        result = _ask(
            self._model,
            "You are a research analyst. Extract key findings as a numbered list.",
            f"Extract the key findings:\n\n{text}",
        )
        return [line.strip() for line in result.strip().split("\n") if line.strip()]

    def generate_questions(self, text: str) -> list[str]:
        """Generate research questions based on a paper."""
        result = _ask(
            self._model,
            "You are a research advisor. Generate insightful questions about this paper.",
            f"Generate research questions about:\n\n{text}",
        )
        return [line.strip() for line in result.strip().split("\n") if line.strip()]

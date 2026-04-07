"""Minimal FastAPI app that renders a markdown file with Claude-like typography."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response

app = FastAPI()

_root_path: Path | None = None
_mode: str | None = None


def _get_root_path() -> Path:
    global _root_path
    if _root_path is None:
        _root_path = Path(os.environ["_OPENSEED_READ_FILE"])
    return _root_path


def _get_mode() -> str:
    global _mode
    if _mode is None:
        _mode = os.environ.get("_OPENSEED_READ_MODE", "file")
    return _mode


def _list_markdown_files(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.md"), key=lambda p: p.relative_to(directory))


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    root = _get_root_path()
    if _get_mode() == "dir":
        files = _list_markdown_files(root)
        return _build_dir_html(root.name, root, files)
    content = root.read_text(encoding="utf-8")
    return _build_html(root.name, content)


@app.get("/view/{file_path:path}", response_class=HTMLResponse)
async def view_file(file_path: str) -> HTMLResponse:
    """Render a specific markdown file from the directory."""
    root = _get_root_path()
    resolved = (root / file_path).resolve()
    if not str(resolved).startswith(str(root)) or not resolved.is_file():
        return HTMLResponse(status_code=404, content="Not found")
    content = resolved.read_text(encoding="utf-8")
    return HTMLResponse(_build_html(resolved.name, content, back_link=True))


@app.get("/assets/{file_path:path}")
async def serve_local_asset(file_path: str) -> Response:
    """Serve images/files relative to the root path's directory."""
    root = _get_root_path()
    base = root if root.is_dir() else root.parent
    resolved = (base / file_path).resolve()
    if not str(resolved).startswith(str(base)):
        return Response(status_code=403)
    if not resolved.is_file():
        return Response(status_code=404)
    import mimetypes

    mime = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
    return Response(content=resolved.read_bytes(), media_type=mime)


def _build_html(filename: str, content: str, *, back_link: bool = False) -> str:
    back = (
        '<a href="/" style="text-decoration:none">← Index</a> <span class="sep">/</span>'
        if back_link
        else ""
    )
    return (
        HTML_TEMPLATE.replace("{{FILENAME}}", filename)
        .replace("{{CONTENT}}", json.dumps(content))
        .replace("{{BACK_LINK}}", back)
    )


def _build_dir_html(dirname: str, root: Path, files: list[Path]) -> str:
    items = []
    for f in files:
        rel = f.relative_to(root)
        size_kb = f.stat().st_size / 1024
        items.append(
            f'<li><a href="/view/{rel}">{rel}</a><span class="size">{size_kb:.0f} KB</span></li>'
        )
    file_list = "\n".join(items) if items else "<li>No markdown files found.</li>"
    return (
        DIR_HTML_TEMPLATE.replace("{{DIRNAME}}", dirname)
        .replace("{{FILE_LIST}}", file_list)
        .replace("{{COUNT}}", str(len(files)))
    )


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{FILENAME}} — OpenSeed Reader</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.11.1/build/styles/atom-one-light.min.css">
<style>
*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

:root {
    --bg: #FAF9F6;
    --surface: #FFFFFF;
    --text: #2D2B28;
    --text-secondary: #6B6560;
    --accent: #C96442;
    --accent-hover: #A8503A;
    --border: #E8E4DF;
    --code-bg: #F5F2EE;
    --blockquote-bg: #FDF8F4;
    --blockquote-border: #D4A574;
    --shadow: 0 1px 3px rgba(45, 43, 40, 0.06), 0 1px 2px rgba(45, 43, 40, 0.04);
}

html {
    font-size: 17px;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

body {
    font-family: "Source Serif 4", Georgia, "Times New Roman", serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.78;
    padding: 0;
}

/* Top bar */
.topbar {
    position: sticky;
    top: 0;
    z-index: 10;
    background: rgba(250, 249, 246, 0.85);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    font-family: "Inter", system-ui, sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 8px;
}
.topbar .logo {
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.02em;
}
.topbar .sep { opacity: 0.3; }

/* Content container */
.container {
    max-width: 740px;
    margin: 0 auto;
    padding: 48px 24px 120px;
}

/* Typography */
.content h1, .content h2, .content h3, .content h4, .content h5, .content h6 {
    font-family: "Inter", system-ui, sans-serif;
    color: var(--text);
    line-height: 1.3;
    margin-top: 2em;
    margin-bottom: 0.6em;
    font-weight: 600;
    letter-spacing: -0.02em;
}
.content h1 {
    font-size: 2em;
    font-weight: 700;
    margin-top: 0;
    padding-bottom: 0.4em;
    border-bottom: 2px solid var(--border);
}
.content h2 {
    font-size: 1.5em;
    padding-bottom: 0.3em;
    border-bottom: 1px solid var(--border);
}
.content h3 { font-size: 1.22em; }
.content h4 { font-size: 1.05em; color: var(--text-secondary); }

.content p {
    margin-bottom: 1.2em;
}

.content a {
    color: var(--accent);
    text-decoration: none;
    border-bottom: 1px solid transparent;
    transition: border-color 0.15s;
}
.content a:hover {
    color: var(--accent-hover);
    border-bottom-color: var(--accent-hover);
}

/* Lists */
.content ul, .content ol {
    margin-bottom: 1.2em;
    padding-left: 1.6em;
}
.content li {
    margin-bottom: 0.35em;
}
.content li > ul, .content li > ol {
    margin-top: 0.35em;
    margin-bottom: 0;
}

/* Blockquote */
.content blockquote {
    border-left: 3px solid var(--blockquote-border);
    background: var(--blockquote-bg);
    margin: 1.5em 0;
    padding: 1em 1.4em;
    border-radius: 0 8px 8px 0;
    color: var(--text-secondary);
    font-style: italic;
}
.content blockquote p:last-child { margin-bottom: 0; }

/* Inline code */
.content code {
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: 0.88em;
    background: var(--code-bg);
    color: var(--accent);
    padding: 0.15em 0.4em;
    border-radius: 4px;
}

/* Code blocks */
.content pre {
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2em 1.4em;
    margin: 1.5em 0;
    overflow-x: auto;
    font-size: 0.88rem;
    line-height: 1.6;
}
.content pre code {
    background: none;
    color: var(--text);
    padding: 0;
    border-radius: 0;
}

/* Tables */
.content table {
    width: 100%;
    border-collapse: collapse;
    margin: 1.5em 0;
    font-size: 0.95rem;
}
.content th {
    font-family: "Inter", system-ui, sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-secondary);
    text-align: left;
    padding: 10px 14px;
    border-bottom: 2px solid var(--border);
}
.content td {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
}
.content tbody tr:hover {
    background: var(--blockquote-bg);
}

/* Images */
.content img {
    max-width: 100%;
    height: auto;
    border-radius: 10px;
    margin: 1.5em auto;
    display: block;
    box-shadow: var(--shadow);
}

/* Horizontal rule */
.content hr {
    border: none;
    height: 1px;
    background: var(--border);
    margin: 2.5em 0;
}

/* KaTeX display math */
.content .katex-display {
    margin: 1.5em 0;
    overflow-x: auto;
    padding: 0.5em 0;
}

/* Checkbox lists (task lists) */
.content input[type="checkbox"] {
    margin-right: 0.4em;
    accent-color: var(--accent);
}

/* Selection */
::selection {
    background: rgba(201, 100, 66, 0.15);
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }

/* Dark mode */
@media (prefers-color-scheme: dark) {
    :root {
        --bg: #1A1918;
        --surface: #242220;
        --text: #E8E4DF;
        --text-secondary: #9B9590;
        --accent: #E07A55;
        --accent-hover: #F09070;
        --border: #3A3632;
        --code-bg: #2A2724;
        --blockquote-bg: #242018;
        --blockquote-border: #B08050;
        --shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
    }
}
</style>
</head>
<body>

<div class="topbar">
    <span class="logo">OpenSeed</span>
    <span class="sep">/</span>
    {{BACK_LINK}}
    <span>{{FILENAME}}</span>
</div>

<div class="container">
    <div class="content" id="content"></div>
</div>

<script src="https://cdn.jsdelivr.net/npm/marked@15.0.7/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/contrib/auto-render.min.js"></script>
<script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.11.1/build/highlight.min.js"></script>
<script>
(function() {
    const raw = {{CONTENT}};

    // Protect math blocks from marked parsing
    const mathBlocks = [];
    function stashMath(text) {
        // Display math: $$...$$
        text = text.replace(/\$\$([\s\S]+?)\$\$/g, (_, m) => {
            mathBlocks.push({display: true, tex: m});
            return `<span class="math-placeholder" data-idx="${mathBlocks.length - 1}"></span>`;
        });
        // Inline math: $...$  (not $$, not \$)
        text = text.replace(/(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)/g, (_, m) => {
            mathBlocks.push({display: false, tex: m});
            return `<span class="math-placeholder" data-idx="${mathBlocks.length - 1}"></span>`;
        });
        return text;
    }

    // Rewrite relative image paths to go through /assets/
    function rewriteImages(html) {
        return html.replace(
            /(<img\s[^>]*src=")(?!https?:\/\/|data:|\/assets\/)([^"]+)(")/g,
            '$1/assets/$2$3'
        );
    }

    const stashed = stashMath(raw);

    marked.setOptions({
        gfm: true,
        breaks: false,
        highlight: function(code, lang) {
            if (lang && hljs.getLanguage(lang)) {
                return hljs.highlight(code, {language: lang}).value;
            }
            return hljs.highlightAuto(code).value;
        }
    });

    let html = marked.parse(stashed);
    html = rewriteImages(html);

    // Restore math blocks
    html = html.replace(
        /<span class="math-placeholder" data-idx="(\d+)"><\/span>/g,
        (_, idx) => {
            const b = mathBlocks[parseInt(idx)];
            try {
                return katex.renderToString(b.tex.trim(), {
                    displayMode: b.display,
                    throwOnError: false
                });
            } catch(e) {
                return b.display ? `$$${b.tex}$$` : `$${b.tex}$`;
            }
        }
    );

    document.getElementById('content').innerHTML = html;
})();
</script>
</body>
</html>
"""

DIR_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{DIRNAME}} — OpenSeed Reader</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
:root {
    --bg: #FAF9F6; --surface: #FFFFFF; --text: #2D2B28;
    --text-secondary: #6B6560; --accent: #C96442; --accent-hover: #A8503A;
    --border: #E8E4DF; --hover-bg: #FDF8F4;
}
html { font-size: 17px; -webkit-font-smoothing: antialiased; }
body {
    font-family: "Inter", system-ui, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6;
}
.topbar {
    position: sticky; top: 0; z-index: 10;
    background: rgba(250, 249, 246, 0.85);
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    font-size: 0.82rem; font-weight: 500; color: var(--text-secondary);
    display: flex; align-items: center; gap: 8px;
}
.topbar .logo { font-weight: 700; color: var(--accent); letter-spacing: -0.02em; }
.topbar .sep { opacity: 0.3; }
.container { max-width: 740px; margin: 0 auto; padding: 48px 24px 120px; }
h1 {
    font-size: 2em; font-weight: 700; letter-spacing: -0.02em;
    margin-bottom: 0.3em; border-bottom: 2px solid var(--border); padding-bottom: 0.4em;
}
.meta { color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 2em; }
ul { list-style: none; padding: 0; }
li {
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid var(--border); transition: background 0.1s;
}
li:hover { background: var(--hover-bg); }
li a {
    flex: 1; padding: 14px 16px; color: var(--text);
    text-decoration: none; font-weight: 500; font-size: 0.95rem;
}
li a:hover { color: var(--accent); }
li .size {
    padding: 14px 16px; color: var(--text-secondary);
    font-size: 0.82rem; white-space: nowrap;
}
@media (prefers-color-scheme: dark) {
    :root {
        --bg: #1A1918; --surface: #242220; --text: #E8E4DF;
        --text-secondary: #9B9590; --accent: #E07A55; --accent-hover: #F09070;
        --border: #3A3632; --hover-bg: #242018;
    }
}
</style>
</head>
<body>
<div class="topbar">
    <span class="logo">OpenSeed</span>
    <span class="sep">/</span>
    <span>{{DIRNAME}}</span>
</div>
<div class="container">
    <h1>{{DIRNAME}}</h1>
    <p class="meta">{{COUNT}} markdown files</p>
    <ul>{{FILE_LIST}}</ul>
</div>
</body>
</html>
"""

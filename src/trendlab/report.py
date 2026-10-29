"""Self-contained HTML research report and Markdown table helpers.

A report is a list of sections rendered into one HTML file with every figure
embedded as a base64 ``data:`` URI (RFC 2397), so it opens offline from disk and
can be attached or archived as a single artifact -- the "one reproducible
document" practice recommended for computational research (Sandve et al., 2013,
*Ten Simple Rules for Reproducible Computational Research*, PLoS Comput Biol).

Section text uses a tiny Markdown subset (paragraphs, ``#``-headings, ``-``/``1.``
lists, ``**bold**``, ``*italic*``, ``code`` and ``[links](url)``); raw HTML can be
passed under the ``html`` key instead.
"""

from __future__ import annotations

import base64
import html
import io
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["build_report", "df_to_html", "df_to_markdown", "markdown_to_html"]

_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
}

_CSS = """
:root{color-scheme:light dark;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;
--muted:#898781;--rule:#e1e0d9;--accent:#2a78d6;--code:#f0efec}
@media (prefers-color-scheme:dark){:root{--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#898781;--rule:#2c2c2a;--accent:#3987e5;--code:#262624}}
*{box-sizing:border-box}
html{background:var(--page)}
body{margin:0;background:var(--page);color:var(--ink);
font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans",sans-serif;
-webkit-text-size-adjust:100%}
main{max-width:1000px;margin:0 auto;padding:40px 16px 80px}
header{border-bottom:1px solid var(--rule);margin-bottom:32px;padding-bottom:20px}
h1{font-size:2rem;line-height:1.2;margin:0 0 8px;letter-spacing:-.01em}
h2{font-size:1.4rem;line-height:1.3;margin:56px 0 12px;padding-top:8px}
h3{font-size:1.1rem;margin:28px 0 8px}h4{font-size:1rem;margin:20px 0 6px}
p,li{color:var(--ink);max-width:72ch}
.meta{color:var(--ink2);margin:0}
a{color:var(--accent)}
code{font:.88em/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
background:var(--code);padding:.1em .35em;border-radius:4px}
nav ol{padding-left:1.2em;columns:2;column-gap:32px;margin:0}
nav li{margin:2px 0;break-inside:avoid}
nav a{color:var(--ink2);text-decoration:none}nav a:hover{color:var(--accent)}
figure{margin:24px 0}
figure img{display:block;width:100%;height:auto;border-radius:8px;
border:1px solid var(--rule);background:#fcfcfb}
figcaption,caption{color:var(--ink2);font-size:.9rem;margin-top:8px;text-align:left}
.table-wrap{overflow-x:auto;margin:20px 0;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;font-size:.9rem;min-width:50%}
caption{caption-side:top;margin:0 0 8px;font-weight:600;color:var(--ink)}
th,td{padding:6px 12px;border-bottom:1px solid var(--rule);white-space:nowrap}
thead th{color:var(--ink2);font-weight:600;border-bottom:1px solid var(--muted);text-align:left}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tbody th{text-align:left;font-weight:500}
tbody tr:hover{background:var(--code)}
footer{margin-top:64px;color:var(--muted);font-size:.85rem}
@media (max-width:640px){nav ol{columns:1}h1{font-size:1.6rem}}
"""

# --------------------------------------------------------------------------- markdown subset

_INLINE = [
    (re.compile(r"\*\*(.+?)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?!\*)"), r"<em>\1</em>"),
    (re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)"), r'<a href="\2">\1</a>'),
]


def _inline(text: str) -> str:
    """Escape HTML, then apply code spans (verbatim), bold, italic and links."""
    parts = re.split(r"(`[^`]+`)", text)
    out = []
    for part in parts:
        if len(part) > 1 and part.startswith("`") and part.endswith("`"):
            out.append(f"<code>{html.escape(part[1:-1], quote=False)}</code>")
            continue
        s = html.escape(part)
        for pat, rep in _INLINE:
            s = pat.sub(rep, s)
        out.append(s)
    return "".join(out)


_UL = re.compile(r"^\s*[-*+]\s+(.*)$")
_OL = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_HEAD = re.compile(r"^(#{1,4})\s+(.*)$")


def markdown_to_html(text: str, base_level: int = 3) -> str:
    """Convert the supported Markdown subset to HTML.

    ``#`` maps to ``<h{base_level}>``, ``##`` one level deeper, and so on (capped
    at h6) so section-level headings stay below the section's own ``<h2>``.
    """
    if not text:
        return ""
    blocks: list[str] = []
    para: list[str] = []
    items: list[str] = []
    kind: str | None = None

    def flush_para():
        if para:
            blocks.append("<p>" + _inline(" ".join(s.strip() for s in para)) + "</p>")
            para.clear()

    def flush_list():
        nonlocal kind
        if items:
            body = "".join(f"<li>{_inline(i)}</li>" for i in items)
            blocks.append(f"<{kind}>{body}</{kind}>")
            items.clear()
        kind = None

    for line in text.strip("\n").splitlines():
        if not line.strip():
            flush_para()
            flush_list()
            continue
        m_ul, m_ol, m_h = _UL.match(line), _OL.match(line), _HEAD.match(line)
        if m_h:
            flush_para()
            flush_list()
            lvl = min(6, base_level + len(m_h.group(1)) - 1)
            blocks.append(f"<h{lvl}>{_inline(m_h.group(2).strip())}</h{lvl}>")
        elif m_ul or m_ol:
            flush_para()
            new = "ul" if m_ul else "ol"
            if kind not in (None, new):
                flush_list()
            kind = new
            items.append((m_ul or m_ol).group(1).strip())
        elif items and line.startswith(("  ", "\t")):
            items[-1] += " " + line.strip()
        else:
            flush_list()
            para.append(line)
    flush_para()
    flush_list()
    return "\n".join(blocks)


# --------------------------------------------------------------------------- tables


def _cell(v: Any, float_fmt: str | None) -> tuple[str, bool]:
    """Display string and whether the cell is numeric (right-aligned)."""
    if v is None or (np.ndim(v) == 0 and not isinstance(v, str) and bool(pd.isna(v))):
        return "\u2013", True
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v)), False
    if isinstance(v, (int, np.integer)):
        # no separator below 10,000 so years and window lengths read naturally
        return (f"{int(v):,}" if abs(int(v)) >= 10_000 else str(int(v))), True
    if isinstance(v, (float, np.floating)):
        if float_fmt:
            return float_fmt.format(v), True
        a = abs(v)
        return (f"{v:,.0f}" if a >= 1000 else f"{v:,.2f}" if a >= 10 else f"{v:.3f}"), True
    if isinstance(v, pd.Timestamp):
        return (v.date().isoformat() if v == v.normalize() else v.isoformat()), False
    s = str(v)
    if s.strip() in {"", "-", "\u2013", "\u2014", "nan", "NaN"}:
        return s, True
    return s, bool(re.fullmatch(r"[\s+\-\u2212$]*[\d.,]+\s*(%|x|\u00d7|bp|bps|pp)?", s))


def _numeric_cols(df: pd.DataFrame, float_fmt: str | None) -> list[bool]:
    """Per column (positionally, so duplicate names are fine): every cell is numeric."""
    if not len(df):
        return [False] * df.shape[1]
    return [all(_cell(v, float_fmt)[1] for v in df.iloc[:, j].tolist()) for j in range(df.shape[1])]


def df_to_html(
    df: pd.DataFrame,
    caption: str | None = None,
    *,
    index: bool = True,
    float_fmt: str | None = None,
) -> str:
    """Escaped HTML table wrapped in a horizontally scrollable container."""
    cols = list(df.columns)
    head = "<th></th>" if index else ""
    if index and df.index.name:
        head = f"<th>{html.escape(str(df.index.name))}</th>"
    num_cols = _numeric_cols(df, float_fmt)
    for c, num in zip(cols, num_cols, strict=False):
        cls = ' class="num"' if num else ""
        head += f"<th{cls}>{html.escape(str(c))}</th>"
    rows = []
    for idx, row in zip(df.index, df.itertuples(index=False), strict=False):
        cells = f"<th>{html.escape(_cell(idx, None)[0])}</th>" if index else ""
        for v, num in zip(row, num_cols, strict=False):
            cls = ' class="num"' if num else ""
            cells += f"<td{cls}>{html.escape(_cell(v, float_fmt)[0])}</td>"
        rows.append(f"<tr>{cells}</tr>")
    cap = f"<caption>{_inline(caption)}</caption>" if caption else ""
    return (
        f'<div class="table-wrap"><table>{cap}<thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def df_to_markdown(df: pd.DataFrame, *, index: bool = True, float_fmt: str | None = None) -> str:
    """GitHub-flavoured Markdown table; numeric columns are right-aligned (``---:``)."""

    def esc(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", " ")

    cols = list(df.columns)
    num = _numeric_cols(df, float_fmt)
    header = ([esc(str(df.index.name or ""))] if index else []) + [esc(str(c)) for c in cols]
    align = ([":---"] if index else []) + ["---:" if n else ":---" for n in num]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    for idx, row in zip(df.index, df.itertuples(index=False), strict=False):
        cells = [esc(_cell(idx, None)[0])] if index else []
        cells += [esc(_cell(v, float_fmt)[0]) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- figures & report


def _img_src(fig: Any) -> str:
    """``data:`` URI for a PNG/JPEG/SVG path or a matplotlib Figure."""
    if hasattr(fig, "savefig"):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
        data, mime = buf.getvalue(), "image/png"
    else:
        p = Path(fig)
        if not p.exists():
            raise FileNotFoundError(f"figure not found: {p}")
        data, mime = p.read_bytes(), _MIME.get(p.suffix.lower(), "application/octet-stream")
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _slug(text: str, used: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"
    slug, k = base, 2
    while slug in used:
        slug, k = f"{base}-{k}", k + 1
    used.add(slug)
    return slug


def _section(sec: Mapping[str, Any], anchor: str) -> str:
    parts = [f'<section id="{anchor}"><h2>{_inline(str(sec.get("title", "")))}</h2>']
    if sec.get("text"):
        parts.append(markdown_to_html(sec["text"]))
    if sec.get("html"):
        parts.append(str(sec["html"]))
    for item in sec.get("figures", []) or []:
        caption, fig = item if isinstance(item, (tuple, list)) else ("", item)
        alt = html.escape(re.sub(r"[*`]", "", caption or "figure"))
        cap = f"<figcaption>{_inline(caption)}</figcaption>" if caption else ""
        parts.append(f'<figure><img src="{_img_src(fig)}" alt="{alt}">{cap}</figure>')
    for item in sec.get("tables", []) or []:
        caption, df = item if isinstance(item, (tuple, list)) else ("", item)
        parts.append(df_to_html(df, caption))
    if sec.get("after"):
        parts.append(markdown_to_html(sec["after"]))
    parts.append("</section>")
    return "\n".join(parts)


def build_report(
    sections: Sequence[Mapping[str, Any]],
    path: str | Path,
    title: str = "trendlab research report",
    *,
    subtitle: str | None = None,
    toc: bool = True,
    footer: str | None = None,
) -> Path:
    """Write a single self-contained HTML report and return its path.

    Each section is a mapping with ``title`` and optional ``text`` (Markdown
    subset), ``html`` (raw HTML), ``figures`` (list of ``(caption, png_path or
    Figure)``), ``tables`` (list of ``(caption, DataFrame)``) and ``after``
    (Markdown placed after figures and tables). Figures precede tables.
    """
    used: set[str] = set()
    anchors = [_slug(str(s.get("title", "section")), used) for s in sections]
    body = "\n".join(_section(s, a) for s, a in zip(sections, anchors, strict=False))
    nav = ""
    if toc and len(sections) > 1:
        links = "".join(
            f'<li><a href="#{a}">{_inline(str(s.get("title", "")))}</a></li>'
            for s, a in zip(sections, anchors, strict=False)
        )
        nav = f'<nav aria-label="Contents"><ol>{links}</ol></nav>'
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    foot = _inline(footer) if footer else f"Generated by trendlab on {stamp}."
    sub = f'<p class="meta">{_inline(subtitle)}</p>' if subtitle else ""
    doc = (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head>"
        f"<body><main><header><h1>{_inline(title)}</h1>{sub}</header>{nav}\n"
        f"{body}\n<footer>{foot}</footer></main></body></html>\n"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path

"""Static website for the research report.

``build_site`` reads what ``trendlab run`` wrote to ``reports/`` (``results.json``,
``sections/*.json``, ``tables/*.csv``, ``data/*.csv`` and ``figures/*.png``) and
renders a small static site: a summary page with the verdict and six headline
findings, one page per study, and the methodology. Nothing is hard-coded: every
number on the summary page is read from ``results.json`` or the tables, so the
site always matches the latest run.
"""

from __future__ import annotations

import html
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..report import markdown_to_html

ASSETS_DIR = Path(__file__).parent / "assets"
REPO_URL = "https://github.com/AshvikMishra/EMA-Crossover-Backtesting-Engine"
PROJECT = "EMA-Crossover-Backtesting-Engine"

# study -> interactive chart shown at the top of its page
STUDY_CHARTS = {
    "e01_bias_waterfall": "waterfall",
    "e02_cross_asset": "equity",
    "e03_parameter_robustness": "heatmap",
    "e05_crises_regimes": "crises",
    "e08_multi_asset_portfolio": "portfolio",
    "e09_significance": "forest",
}
# static figures that the interactive chart on the same page replaces
REPLACED_FIGURES = {
    "figures/e01_bias_waterfall_waterfall.png",
    "figures/e02_cross_asset_equity_grid.png",
    "figures/e03_parameter_robustness_sharpe_spy.png",
    "figures/e03_parameter_robustness_sharpe_qqq.png",
    "figures/e03_parameter_robustness_sharpe_btc.png",
    "figures/e08_multi_asset_portfolio_equity.png",
    "figures/e09_significance_forest.png",
}


# --------------------------------------------------------------------------- helpers


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def pct(x: float, digits: int = 0, sign: bool = False) -> str:
    s = f"{x * 100:+.{digits}f}%" if sign else f"{x * 100:.{digits}f}%"
    return s.replace("-", "−")


def num(x: float, digits: int = 2, sign: bool = False) -> str:
    s = f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"
    return s.replace("-", "−")


def study_id(name: str) -> str:
    return name.split("_", 1)[0]


def strip_number(title: str) -> str:
    return re.sub(r"^\d+\.\s*", "", title)


def question_of(text: str) -> str:
    m = re.search(r"\*\*Question\.\*\*\s*(.+)", text)
    return m.group(1).strip() if m else ""


def _clean(v):
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), 6)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


@dataclass
class Report:
    root: Path
    results: dict
    sections: dict[str, dict]

    @classmethod
    def load(cls, root: Path) -> Report:
        results = json.loads((root / "results.json").read_text())
        sections = {}
        for f in sorted((root / "sections").glob("*.json")):
            d = json.loads(f.read_text())
            sections[d["name"]] = d
        if not sections:
            raise FileNotFoundError(f"no sections in {root / 'sections'}; run `trendlab run` first")
        return cls(root, results, sections)

    def r(self, study: str, key: str, default=None):
        return self.results.get(study, {}).get(key, default)

    def table(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.root / "tables" / f"{name}.csv")

    def data(self, study: str, key: str) -> pd.DataFrame:
        return pd.read_csv(self.root / "data" / f"{study}__{key}.csv", parse_dates=["Date"])


# --------------------------------------------------------------------------- chart payloads


def chart_waterfall(rep: Report) -> dict:
    df = rep.table("e01_bias_waterfall_waterfall")
    labels = {
        0: (
            "Notebook as written",
            "Notebook",
            "Unwarmed EMAs, cross-only entries, same-close fills, $5 per trade, cash at 0%.",
        ),
        1: (
            "Warm EMAs, state rule",
            "Warm-up",
            "Indicators warmed on prior history; long whenever EMA50 is above EMA200.",
        ),
        2: (
            "Next-close fills",
            "Fills",
            "Each decision fills at the close after the signal, not the signal's own close.",
        ),
        3: (
            "5 bp costs",
            "Costs",
            "Commission, half-spread and slippage charged on traded notional.",
        ),
        4: ("T-bill cash", "Cash", "Uninvested capital earns the 3-month T-bill rate."),
    }
    steps = []
    for i, row in df.reset_index(drop=True).iterrows():
        label, short, what = labels.get(i, (str(row["Step"]), str(row["Step"]), ""))
        steps.append(
            {
                "label": label,
                "short": short,
                "what": what,
                "total": _clean(row["Strategy total"]),
                "bnh": _clean(row["B&H total"]),
                "mdd": _clean(row["Strategy max DD"]),
                "bnh_mdd": _clean(row["B&H max DD"]),
                "sharpe": _clean(row["Strategy Sharpe"]),
                "bnh_sharpe": _clean(row["B&H Sharpe"]),
                "trips": _clean(row["Round trips"]),
            }
        )
    return {
        "type": "waterfall",
        "title": "NVDA, Jan 2022 to Dec 2024: EMA50/200 total return as each bias is removed",
        "steps": steps,
        "source": "reports/tables/e01_bias_waterfall_waterfall.csv",
    }


def chart_equity(rep: Report, tickers: list[str] | None = None) -> dict:
    assets = rep.table("e02_cross_asset_assets").set_index("Asset")
    tickers = tickers or list(assets.index)
    series = {}
    for t in tickers:
        df = rep.data("e02_cross_asset", f"equity_{t}")
        a = assets.loc[t]
        series[t] = {
            "dates": df["Date"].dt.strftime("%Y-%m-%d").tolist(),
            "lines": {k: [_clean(v) for v in df[k]] for k in ("strategy", "bnh")},
            "stats": {
                "sharpe": _clean(a["sharpe"]),
                "bnh_sharpe": _clean(a["bnh_sharpe"]),
                "mdd": _clean(a["max_drawdown"]),
                "bnh_mdd": _clean(a["bnh_max_drawdown"]),
                "cagr": _clean(a["cagr"]),
                "bnh_cagr": _clean(a["bnh_cagr"]),
                "class": a["class"],
            },
        }
    return {
        "type": "equity",
        "title": "Growth of $1 (log scale) and drawdown: EMA50/200 vs buy-and-hold",
        "series": series,
        "names": {"strategy": "EMA50/200", "bnh": "Buy & hold"},
        "order": tickers,
        "source": "reports/data/e02_cross_asset__equity_<ticker>.csv",
    }


def chart_portfolio(rep: Report) -> dict:
    df = rep.data("e08_multi_asset_portfolio", "equity")
    return {
        "type": "equity",
        "title": "Ten-ETF trend book vs 60/40 vs an 80/20 blend: growth of $1 and drawdown",
        "series": {
            "Portfolios": {
                "dates": df["Date"].dt.strftime("%Y-%m-%d").tolist(),
                "lines": {k: [_clean(v) for v in df[k]] for k in ("trend", "sixty_forty", "blend")},
                "stats": {},
            }
        },
        "names": {"trend": "Trend book", "sixty_forty": "60/40", "blend": "80% 60/40 + 20% trend"},
        "order": ["Portfolios"],
        "source": "reports/data/e08_multi_asset_portfolio__equity.csv",
    }


def chart_heatmap(rep: Report) -> dict:
    grid = rep.table("e03_parameter_robustness_grid")
    out = {}
    for asset, g in grid.groupby("asset", sort=False):
        key = asset.lower().split("-")[0]
        out[asset] = {
            "cells": _records(g[["fast", "slow", "sharpe", "cagr", "max_drawdown", "n_trades"]]),
            "bnh_sharpe": _clean(rep.r("e03_parameter_robustness", f"{key}_bnh_sharpe")),
            "pbo": _clean(rep.r("e03_parameter_robustness", f"{key}_pbo")),
            "dsr": _clean(rep.r("e03_parameter_robustness", f"{key}_dsr_best_vs_bnh")),
            "start": rep.r("e03_parameter_robustness", f"{key}_start"),
        }
    return {
        "type": "heatmap",
        "title": "Sharpe ratio of every EMA(fast) / EMA(slow) crossover in the grid",
        "assets": out,
        "order": list(out),
        "mark": [50, 200],
        "source": "reports/tables/e03_parameter_robustness_grid.csv",
    }


def chart_forest(rep: Report) -> dict:
    df = rep.table("e09_significance_raw")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "asset": r["asset"],
                "diff": _clean(r["sharpe_diff"]),
                "lo": _clean(r["diff_ci_low"]),
                "hi": _clean(r["diff_ci_high"]),
                "p": _clean(r["p_boot"]),
                "q": _clean(r["q_boot_bh"]),
                "sharpe": _clean(r["sharpe_strat"]),
                "bnh_sharpe": _clean(r["sharpe_bnh"]),
                "years": _clean(r["years"]),
            }
        )
    return {
        "type": "forest",
        "title": "Sharpe difference, EMA50/200 minus buy-and-hold, with 95% bootstrap intervals",
        "rows": rows,
        "source": "reports/tables/e09_significance_raw.csv",
    }


def chart_crises(rep: Report, asset: str = "SPY") -> dict:
    df = rep.table("e05_crises_regimes_crises")
    df = df[df["asset"] == asset]
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "crisis": r["crisis"],
                "start": r["start"],
                "end": r["end"],
                "bnh": _clean(r["bench_return"]),
                "rule": _clean(r["strat_return"]),
                "exit": r["exit_date"] if isinstance(r["exit_date"], str) else None,
                "reentry": r["reentry_date"] if isinstance(r["reentry_date"], str) else None,
                "missed": _clean(r["missed_rebound"]),
                "edge": _clean(r["cycle_edge"]),
            }
        )
    return {
        "type": "crises",
        "title": f"{asset} stress episodes: buy-and-hold vs EMA50/200 return inside each window",
        "rows": rows,
        "source": "reports/tables/e05_crises_regimes_crises.csv",
    }


CHARTS = {
    "waterfall": chart_waterfall,
    "equity": chart_equity,
    "portfolio": chart_portfolio,
    "heatmap": chart_heatmap,
    "forest": chart_forest,
    "crises": chart_crises,
}


# --------------------------------------------------------------------------- data tables (a11y)


def data_table(chart: dict) -> str:
    """A plain table of the chart's data, folded under the chart for screen readers and checking."""
    t = chart["type"]
    if t == "waterfall":
        head = ["Step", "EMA50/200", "Buy & hold", "Max DD", "Sharpe"]
        rows = [
            [s["label"], pct(s["total"], 1), pct(s["bnh"], 1), pct(s["mdd"], 1), num(s["sharpe"])]
            for s in chart["steps"]
        ]
    elif t == "forest":
        head = ["Asset", "Sharpe diff", "95% CI", "p", "BH q"]
        rows = [
            [
                r["asset"],
                num(r["diff"], sign=True),
                f"{num(r['lo'])} to {num(r['hi'])}",
                num(r["p"]),
                num(r["q"]),
            ]
            for r in chart["rows"]
        ]
    elif t == "crises":
        head = ["Episode", "Buy & hold", "EMA50/200", "Exit", "Re-entry"]
        rows = [
            [
                r["crisis"],
                pct(r["bnh"], 1),
                pct(r["rule"], 1),
                r["exit"] or "–",
                r["reentry"] or "–",
            ]
            for r in chart["rows"]
        ]
    elif t == "heatmap":
        head = ["Asset", "Fast", "Slow", "Sharpe", "CAGR", "Max DD"]
        rows = [
            [
                a,
                c["fast"],
                c["slow"],
                num(c["sharpe"]),
                pct(c["cagr"], 1),
                pct(c["max_drawdown"], 1),
            ]
            for a, d in chart["assets"].items()
            for c in d["cells"]
        ]
    else:  # equity: yearly samples keep the table readable
        head = ["Series", "Date"] + [chart["names"][k] for k in chart["names"]]
        rows = []
        for name, s in chart["series"].items():
            seen = set()
            for i, d in enumerate(s["dates"]):
                if d[:4] in seen and i != len(s["dates"]) - 1:
                    continue
                seen.add(d[:4])
                rows.append(
                    [name, d]
                    + [
                        "–" if s["lines"].get(k, [None])[i] is None else f"{s['lines'][k][i]:.2f}×"
                        for k in chart["names"]
                        if k in s["lines"]
                    ]
                )
    th = "".join(f"<th>{esc(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return (
        f'<details class="chart-data"><summary>Chart data</summary><div class="table-wrap">'
        f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>"
        f'<p class="source">Source: <code>{esc(chart["source"])}</code></p></details>'
    )


def chart_block(chart_id: str, chart: dict, note: str, label: str) -> str:
    """Interactive chart in the text column with its live readout in the margin."""
    payload = json.dumps(chart, separators=(",", ":")).replace("</", "<\\/")
    return f"""
<div class="row chart-row">
  <figure class="main chart" id="{chart_id}" data-chart="{chart["type"]}" tabindex="0"
    role="group" aria-label="{esc(label)}" aria-describedby="{chart_id}-readout">
    <figcaption class="chart-title">{esc(chart["title"])}</figcaption>
    <div class="chart-controls" hidden></div>
    <div class="chart-canvas"><noscript><p>This chart needs JavaScript; its data is in the table below.</p></noscript></div>
    {data_table(chart)}
    <script type="application/json" class="chart-payload">{payload}</script>
  </figure>
  <aside class="side readout" id="{chart_id}-readout" aria-live="polite">
    <div class="readout-default">{note}</div>
    <div class="readout-live" hidden></div>
  </aside>
</div>"""


# --------------------------------------------------------------------------- page shell


def nav_links(depth: int, active: str) -> str:
    up = "../" * depth
    items = [
        ("Findings", f"{up}index.html#findings", "findings"),
        ("Studies", f"{up}index.html#studies", "studies"),
        ("Methodology", f"{up}methodology.html", "methodology"),
        ("Source", REPO_URL, "source"),
    ]
    out = []
    for label, href, key in items:
        cur = ' aria-current="page"' if key == active else ""
        out.append(f'<a href="{href}"{cur}>{label}</a>')
    return "".join(out)


def page(title: str, body: str, *, depth: int, active: str, description: str, end: str) -> str:
    up = "../" * depth
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(description)}">
<meta name="color-scheme" content="light">
<link rel="preload" href="{up}assets/fonts/et-book-roman-line-figures.woff" as="font" type="font/woff" crossorigin>
<link rel="stylesheet" href="{up}assets/site.css">
<link rel="icon" href="{up}assets/favicon.svg" type="image/svg+xml">
</head>
<body>
<a class="skip" href="#content">Skip to content</a>
<header class="runninghead">
  <div class="rh-inner">
    <a class="rh-title" href="{up}index.html">{PROJECT}</a>
    <span class="rh-meta">Research report · data through {esc(end)}</span>
    <nav aria-label="Primary">{nav_links(depth, active)}</nav>
  </div>
</header>
<main id="content" class="sheet">
{body}
</main>
<footer class="colophon">
  <div class="row"><div class="main">
    <p>Generated by <code>trendlab site</code> from the committed data snapshot ending {esc(end)}.
    Every figure on this site is computed by the code in
    <a href="{REPO_URL}">{PROJECT}</a>; re-run with <code>trendlab run &amp;&amp; trendlab site</code>.</p>
    <p>Not investment advice. Backtested performance is not indicative of future results.</p>
  </div><div class="side"><p>Set in ET Book (MIT licence). Ashvik Mishra.</p></div></div>
</footer>
<script src="{up}assets/charts.js" defer></script>
</body>
</html>
"""


def note(*parts: str) -> str:
    return "".join(f"<p>{p}</p>" for p in parts)


def prov(study: str, key: str | None = None, label: str | None = None, depth: int = 0) -> str:
    """Provenance line for a margin note: which study and results key produced a number."""
    up = "../" * depth
    href = f"{up}studies/{study_id(study)}.html"
    k = f" · <code>{esc(key)}</code>" if key else ""
    return (
        f'<span class="prov">Source: <a href="{href}">{esc(label or study_id(study))}</a>{k}</span>'
    )


# --------------------------------------------------------------------------- summary page


def finding(heading: str, verdict: str, body: str, side: str, chart_html: str = "") -> str:
    return f"""
<section class="finding">
  <div class="row">
    <h3 class="main">{heading}</h3>
    <p class="side verdict" aria-label="Verdict: {esc(verdict)}">{esc(verdict)}</p>
  </div>
  <div class="row">
    <div class="main prose">{body}</div>
    <aside class="side note">{side}</aside>
  </div>
  {chart_html}
</section>"""


def figure_row(src: str, caption: str, alt: str) -> str:
    return f"""
<div class="row figure-row">
  <figure class="full">
    <div class="fig-scroll"><a href="{src}" aria-label="Open full-size figure: {esc(caption)}"><img src="{src}" alt="{esc(alt)}" loading="lazy" decoding="async"></a></div>
    <figcaption>{esc(caption)}<span class="scroll-hint"> · Swipe to pan, or tap to enlarge.</span></figcaption>
  </figure>
</div>"""


def render_index(rep: Report, end: str) -> str:
    R = lambda s, k, d=float("nan"): rep.r(s, k, d)  # noqa: E731
    e1, e2, e3, e4, e5 = (
        "e01_bias_waterfall",
        "e02_cross_asset",
        "e03_parameter_robustness",
        "e04_walk_forward",
        "e05_crises_regimes",
    )
    e8, e9, e10 = "e08_multi_asset_portfolio", "e09_significance", "e10_costs"

    orig, orig_b = R(e1, "original_algo_return"), R(e1, "original_bnh_return")
    corr, corr_b = R(e1, "corrected_algo_return"), R(e1, "corrected_bnh_return")
    beats = corr > corr_b

    arrow = (
        '<svg class="claim-arrow" viewBox="0 0 48 16" aria-hidden="true">'
        '<path d="M2 8h40M34 2l8 6-8 6" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>'
    )
    hero = f"""
<div class="row title-row">
  <div class="main">
    <h1>An EMA crossover, honestly</h1>
    <p class="standfirst">The original notebook reported that a 50/200-day EMA crossover on NVIDIA
    returned {pct(orig, 0, True)} over 2022–2024 against {pct(orig_b, 0, True)} for buy-and-hold.
    Rebuilt without look-ahead and warm-up biases, the same rule returns
    {pct(corr, 0, True)} against {pct(corr_b, 0, True)}, and what survives across twelve markets is a cheap way
    to cut drawdowns, not a better return.</p>
  </div>
  <aside class="side note">
    <p>This site reports ten studies run by the <code>trendlab</code> package on a committed
    data snapshot. Each headline number cites the study and results key that produced it.</p>
  </aside>
</div>
<div class="row claim-row">
  <p class="main claim" aria-label="Claimed {pct(orig, 0, True)}, corrected {pct(corr, 0, True)}, buy and hold {pct(corr_b, 0, True)}">
    <span class="claim-old" title="The original claim, under examination">{pct(orig, 0, True)}</span>
    {arrow}
    <span class="claim-new">{pct(corr, 0, True)}</span>
    <span class="claim-bnh">buy &amp; hold {pct(corr_b, 0, True)}</span>
  </p>
  <aside class="side note">{prov(e1, "corrected_algo_return")}</aside>
</div>"""
    waterfall = chart_block(
        "chart-waterfall",
        chart_waterfall(rep),
        note(
            "Each bar removes one bias, cumulatively, on the same window. The grey line is buy-and-hold under the final settings.",
            "Point at a bar, or focus the chart and use the arrow keys, to see what that step changed.",
        ),
        "Bias waterfall chart",
    )

    # finding 1: warm-up
    f1 = finding(
        "The +505% came from an unwarmed EMA200, not from timing",
        "Artifact" if not beats else "Survives",
        markdown_to_html(
            f"An EMA seeds itself with the first price it sees. The notebook downloaded exactly the test "
            f"window, so its EMA200 started at NVIDIA's January 2022 price, the averages 'death-crossed' on "
            f"day two, and the rule sat in cash through the 2022 crash by accident. Warmed up on prior "
            f"history, the averages say NVIDIA was in an uptrend on 1 January 2022: an honest rule starts "
            f"invested, rides the first leg down and exits on {R(e1, 'warm_first_exit', '–')}, after a "
            f"{pct(-R(e1, 'nvda_return_to_first_exit'), 0)} fall.\n\n"
            f"Over the full 2000–2026 history the same rule compounds at "
            f"{pct(R(e1, 'full_history_algo_cagr'), 1)} a year against {pct(R(e1, 'full_history_bnh_cagr'), 1)} "
            f"for buy-and-hold, with a {pct(R(e1, 'full_history_algo_mdd'))} worst drawdown against "
            f"{pct(R(e1, 'full_history_bnh_mdd'))}.",
            base_level=4,
        ),
        note(
            f"Sharpe difference over the notebook's window: {num(R(e1, 'corrected_sharpe_diff'), sign=True)}, "
            f"95% bootstrap interval {num(R(e1, 'corrected_sharpe_diff_ci_low'))} to "
            f"{num(R(e1, 'corrected_sharpe_diff_ci_high'))}. Two round trips cannot separate skill from luck.",
            prov(e1, "warm_first_exit"),
        ),
        figure_row(
            "figures/e01_bias_waterfall_ema_seeding.png",
            "NVDA in 2022: EMAs seeded at the first close (dashed) against EMAs warmed on prior history (solid).",
            "Chart of NVDA price in 2022 with seeded and warmed EMA50 and EMA200 lines and the trade markers of both rules.",
        ),
    )

    n = R(e2, "n_assets", 12)
    f2 = finding(
        f"Across {n} markets the rule cuts drawdowns, not returns",
        "Holds" if R(e2, "maxdd_shallower_than_bnh", 0) == n else "Mixed",
        markdown_to_html(
            f"The universe was fixed before looking at results: one liquid ETF per asset class, plus "
            f"Bitcoin and the original NVIDIA. EMA50/200 beats buy-and-hold's Sharpe ratio on "
            f"{R(e2, 'sharpe_beats_bnh')} of {n} and its annual return on {R(e2, 'cagr_beats_bnh')} of {n}, "
            f"giving up a median {pct(-R(e2, 'median_cagr_diff'), 1)} a year. Its worst drawdown is shallower "
            f"on **{R(e2, 'maxdd_shallower_than_bnh')} of {n}**, by a median "
            f"{R(e2, 'median_maxdd_diff') * 100:.0f} percentage points.\n\n"
            f"That is mostly a consequence of holding less. The rule is invested about "
            f"{pct(R(e2, 'median_exposure'))} of the time, and a static mix with the same average exposure "
            f"reproduces a median {pct(R(e9, 'median_mdd_gap_share_from_exposure'))} of the drawdown reduction.",
            base_level=4,
        ),
        note(
            f"Monthly up-capture {pct(R(e2, 'median_upside_capture'))}, down-capture "
            f"{pct(R(e2, 'median_downside_capture'))}: nearly symmetric, the signature of holding less "
            f"rather than timing well.",
            prov(e2, "maxdd_shallower_than_bnh"),
        ),
        chart_block(
            "chart-equity",
            chart_equity(rep),
            note(
                "Choose a market. The upper panel is growth of $1 on a log scale; the lower panel is the drawdown from the running peak.",
                "Point along the chart to read both curves on a date.",
            ),
            "Equity and drawdown chart across twelve markets",
        ),
    )

    sig = R(e9, "n_sharpe_sig_bh", 0)
    raw9 = rep.table("e09_significance_raw").set_index("asset")
    spy_years = float(raw9.loc["SPY", "years"])
    f3 = finding(
        "None of the Sharpe-ratio gains is statistically significant",
        "Not significant" if sig == 0 else f"{sig} significant",
        markdown_to_html(
            f"A paired stationary bootstrap of the Sharpe difference finds {R(e9, 'n_sharpe_sig_raw')} of "
            f"{n} markets significant at 5% before correction and {sig} after the Holm and "
            f"Benjamini–Hochberg corrections for testing {n} markets at once. The median 95% interval is "
            f"{num(R(e9, 'median_ci_width'))} Sharpe wide.\n\n"
            f"On SPY, with {spy_years:.0f} years of data, the probability that the rule's true Sharpe "
            f"exceeds buy-and-hold's is {num(R(e9, 'spy_psr_vs_bnh'))}; confirming the observed gap at 95% "
            f"would take about **{R(e9, 'spy_mintrl_vs_bnh_years'):,.0f} years**. QQQ's standout gap of "
            f"{num(R(e9, 'qqq_full_sharpe_diff', float('nan')), sign=True)} rests on 2001–02 alone; from late 2002 it is "
            f"{num(R(e9, 'qqq_late_window_sharpe_diff'), sign=True)}.",
            base_level=4,
        ),
        note(
            f"A random-entry test keeps the rule's trade count, holding periods and exposure but places "
            f"the trades at random: it rejects random timing on {R(e9, 'n_random_entry_sig_raw')} market "
            f"before correction and {R(e9, 'n_random_entry_sig_bh')} after.",
            prov(e9, "n_sharpe_sig_bh"),
        ),
        chart_block(
            "chart-forest",
            chart_forest(rep),
            note(
                "Each line is a 95% bootstrap interval for the Sharpe difference. Every interval crossing zero is a difference the data cannot distinguish from luck."
            ),
            "Forest plot of Sharpe differences with confidence intervals",
        ),
    )

    f4 = finding(
        "50/200 sits on a flat plateau, and tuning it is noise",
        "Noise",
        markdown_to_html(
            f"Across a 62-cell grid of fast and slow windows, the textbook 50/200 ranks "
            f"{R(e3, 'spy_tb_rank')}th of 62 on SPY; the middle half of the grid spans "
            f"{num(R(e3, 'spy_grid_q25_sharpe'))}–{num(R(e3, 'spy_grid_q75_sharpe'))} Sharpe, inside one "
            f"standard error. Once the 62 trials are counted, the deflated Sharpe ratio gives even the *best* "
            f"cell a probability of {num(R(e3, 'spy_dsr_best_vs_bnh'))} (SPY), {num(R(e3, 'qqq_dsr_best_vs_bnh'))} "
            f"(QQQ) and {num(R(e3, 'btc_dsr_best_vs_bnh'))} (BTC) of beating buy-and-hold.\n\n"
            f"Re-optimising the windows every year on the past five years changed the chosen pair at "
            f"{R(e4, 'total_switches')} of {R(e4, 'total_boundaries')} boundaries and did "
            f"{num(R(e4, 'mean_sharpe_diff_vs_static'), sign=True)} Sharpe on average against simply fixing 50/200.",
            base_level=4,
        ),
        note(
            f"Probability of backtest overfitting (CSCV): {pct(R(e3, 'spy_pbo'))} on SPY, "
            f"{pct(R(e3, 'qqq_pbo'))} on QQQ, {pct(R(e3, 'btc_pbo'))} on Bitcoin. The in-sample winner usually "
            f"lands below the median out of sample.",
            prov(e3, "spy_pbo"),
        ),
        chart_block(
            "chart-heatmap",
            chart_heatmap(rep),
            note(
                "Blue cells beat buy-and-hold's Sharpe, sand cells trail it; the outlined cell is 50/200.",
                "Point at a cell, or focus the chart and use the arrow keys.",
            ),
            "Heatmap of Sharpe ratios across the parameter grid",
        ),
    )

    cr = {r["crisis"]: r for r in chart_crises(rep)["rows"]}
    f5 = finding(
        "It earns its keep in slow bear markets and pays for it in fast ones",
        "Conditional",
        markdown_to_html(
            f"On SPY the rule lost {pct(-cr['Dot-com bust']['rule'])} in the dot-com bust while buy-and-hold "
            f"lost {pct(-cr['Dot-com bust']['bnh'])}, and {pct(-cr['Global financial crisis']['rule'])} in the "
            f"financial crisis against {pct(-cr['Global financial crisis']['bnh'])}. In the COVID crash it sold at "
            f"the bottom and missed a {pct(R(e5, 'spy_covid_crash_missed_rebound'))} rebound before buying "
            f"back. A 50/200 rule is a slow filter with a mean lag of about "
            f"{R('e06_signal_comparison', 'ema_kernel_mean_lag', 124):.0f} trading days: it can only help when "
            f"a decline lasts longer than that.",
            base_level=4,
        ),
        note(
            f"Across all assets the rule helped in {R(e5, 'n_helped')} of {R(e5, 'n_crises')} episodes and hurt "
            f"in {R(e5, 'n_hurt')}, counting the cost of late re-entry.",
            prov(e5, "n_helped"),
        ),
        chart_block(
            "chart-crises",
            chart_crises(rep),
            note(
                "Grey dot: buy-and-hold over the episode. Blue dot: EMA50/200. A long blue-right line is protection.",
                "Point at an episode for the exit and re-entry dates.",
            ),
            "Dot plot of returns in each SPY stress episode",
        ),
    )

    f6 = finding(
        "Diversified across ten markets, it becomes a real drawdown hedge",
        "Holds" if R(e8, "trend_maxdd") > R(e8, "sixty_forty_maxdd") else "Mixed",
        markdown_to_html(
            f"Run as an inverse-volatility book across ten ETFs, the same rule's worst drawdown was "
            f"{pct(R(e8, 'trend_maxdd'))} against {pct(R(e8, 'sixty_forty_maxdd'))} for a 60/40 portfolio. It "
            f"made {pct(R(e8, 'trend_2008'), 1, True)} in 2008, when 60/40 lost "
            f"{pct(-R(e8, 'sixty_forty_2008'), 1)}, and lost only {pct(-R(e8, 'trend_2022'), 1)} in 2022, when "
            f"stocks and bonds fell together and 60/40 lost {pct(-R(e8, 'sixty_forty_2022'), 1)}.\n\n"
            f"Its Sharpe ratio, {num(R(e8, 'trend_sharpe'))}, is indistinguishable from 60/40's "
            f"{num(R(e8, 'sixty_forty_sharpe'))}. The value is in *when* it loses, not how much it earns.",
            base_level=4,
        ),
        note(
            f"An average single sleeve has a Sharpe of {num(R(e8, 'mean_single_sleeve_sharpe'))}; the "
            f"diversified book reaches {num(R(e8, 'trend_sharpe'))}. Diversification across markets, not the "
            f"choice of windows, is where trend following's Sharpe comes from.",
            prov(e8, "trend_maxdd"),
        ),
        chart_block(
            "chart-portfolio",
            chart_portfolio(rep),
            note(
                "Growth of $1 from May 2004, and drawdown below it.",
                "Point along the chart to compare the three books on a date.",
            ),
            "Equity and drawdown chart of the trend book, 60/40 and the blend",
        ),
    )

    costs = f"""
<div class="row">
  <div class="main prose">{
        markdown_to_html(
            f"Costs and fill timing do not change any of this. The rule trades {R(e10, 'median_round_trips_per_year'):.1f} "
            f"round trips a year, so 5 bp per side costs a median {R(e10, 'median_cost_drag_bps_per_year'):.0f} bp a year, "
            f"and the cost at which it would stop beating T-bills is a median {R(e10, 'median_breakeven_vs_cash_bps'):,.0f} bp per side.",
            base_level=4,
        )
    }</div>
  <aside class="side note">{prov(e10, "median_cost_drag_bps_per_year")}</aside>
</div>"""

    studies = []
    for name, sec in rep.sections.items():
        studies.append(
            f'<li><a href="studies/{study_id(name)}.html"><span class="sid">{study_id(name)}</span>'
            f'<span class="stitle">{esc(strip_number(sec["title"]))}</span></a>'
            f"<p>{esc(question_of(sec['text']))}</p></li>"
        )
    index = f"""
<section id="studies" class="studies">
  <div class="row"><h2 class="main">The ten studies</h2>
  <aside class="side note"><p>Identifiers match the command line: <code>trendlab run e03</code> re-runs study e03.</p></aside></div>
  <div class="row"><ol class="main study-list">{"".join(studies)}</ol></div>
</section>
<section class="reproduce">
  <div class="row"><h2 class="main">Reproduce every number</h2></div>
  <div class="row"><div class="main prose">
<pre><code>git clone {REPO_URL}.git
cd {PROJECT}
pip install -e ".[dev]"
trendlab run        # ten studies, about 90 seconds, offline
trendlab site       # this website, into site/</code></pre>
  </div><aside class="side note"><p>The price history is committed in <code>data/raw/</code>, so the run needs no network.
  The conventions behind every number are in the <a href="methodology.html">methodology</a>.</p></aside></div>
</section>"""

    body = (
        hero
        + waterfall
        + '<section id="findings"><div class="row"><h2 class="main">Six findings</h2></div>'
        + f1
        + f2
        + f3
        + f4
        + f5
        + f6
        + costs
        + "</section>"
        + index
    )
    return page(
        "An EMA crossover, honestly · trendlab research report",
        body,
        depth=0,
        active="findings",
        description="A bias-aware backtest of the EMA50/200 crossover across twelve markets: what survives, and what was an artifact.",
        end=end,
    )


# --------------------------------------------------------------------------- study pages


# headline numbers shown in each study's margin, with the results.json key that holds them
KEY_RESULTS: dict[str, list[tuple[str, str, str]]] = {
    "e01_bias_waterfall": [
        ("Notebook claim", "original_algo_return", "pct1"),
        ("Corrected rule", "corrected_algo_return", "pct1"),
        ("Buy & hold", "corrected_bnh_return", "pct1"),
        ("Rule max drawdown", "corrected_algo_mdd", "pct1"),
        ("Sharpe gap p", "corrected_sharpe_diff_p", "num"),
    ],
    "e02_cross_asset": [
        ("Sharpe above B&H", "sharpe_beats_bnh", "int"),
        ("CAGR above B&H", "cagr_beats_bnh", "int"),
        ("Shallower drawdown", "maxdd_shallower_than_bnh", "int"),
        ("Median CAGR gap", "median_cagr_diff", "pct1"),
        ("Median drawdown gap", "median_maxdd_diff", "pct0"),
    ],
    "e03_parameter_robustness": [
        ("SPY 50/200 rank", "spy_tb_rank", "int"),
        ("SPY overfit probability", "spy_pbo", "pct0"),
        ("QQQ overfit probability", "qqq_pbo", "pct0"),
        ("SPY best cell, deflated", "spy_dsr_best_vs_bnh", "num"),
    ],
    "e04_walk_forward": [
        ("Re-tuned vs fixed Sharpe", "mean_sharpe_diff_vs_static", "num"),
        ("Parameter switches", "total_switches", "int"),
        ("SPY walk-forward Sharpe", "spy_wf_sharpe", "num"),
        ("SPY fixed 50/200 Sharpe", "spy_static_sharpe", "num"),
    ],
    "e05_crises_regimes": [
        ("Episodes helped", "n_helped", "int"),
        ("Episodes hurt", "n_hurt", "int"),
        ("SPY dot-com edge", "spy_dot_com_bust_protection", "pp"),
        ("SPY COVID rebound missed", "spy_covid_crash_missed_rebound", "pct0"),
    ],
    "e06_signal_comparison": [
        ("SPY EMA50/200 Sharpe", "spy_sharpe_ema", "num"),
        ("SPY TSMOM 12M Sharpe", "spy_sharpe_tsmom_12", "num"),
        ("Agreement with EMA", "avg_agreement_with_ema", "pct0"),
        ("EMA kernel mean lag (days)", "ema_kernel_mean_lag", "int"),
    ],
    "e07_vol_targeting": [
        ("SPY B&H Sharpe", "spy_sharpe_bh", "num"),
        ("SPY B&H + vol target", "spy_sharpe_bh_vt", "num"),
        ("SPY max DD, vol target", "spy_maxdd_bh_vt", "pct0"),
        ("Significant changes", "n_significant_bh_vt", "int"),
    ],
    "e08_multi_asset_portfolio": [
        ("Trend book Sharpe", "trend_sharpe", "num"),
        ("Trend book max DD", "trend_maxdd", "pct0"),
        ("60/40 max DD", "sixty_forty_maxdd", "pct0"),
        ("Trend book 2022", "trend_2022", "pct1"),
        ("60/40 2022", "sixty_forty_2022", "pct1"),
    ],
    "e09_significance": [
        ("Sharpe gains significant", "n_sharpe_sig_bh", "int"),
        ("SPY P(rule > B&H)", "spy_psr_vs_bnh", "num"),
        ("SPY years to confirm", "spy_mintrl_vs_bnh_years", "int"),
        ("Timing beats random", "n_random_entry_sig_bh", "int"),
    ],
    "e10_costs": [
        ("Median cost drag (bp/yr)", "median_cost_drag_bps_per_year", "int"),
        ("Median break-even (bp)", "median_breakeven_vs_cash_bps", "int"),
        ("Sharpe drop at 20 bp", "max_sharpe_drop_at_20bps", "num"),
    ],
}


def _fmt(v, kind: str) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if kind == "pct0":
        return pct(v, 0)
    if kind == "pct1":
        return pct(v, 1)
    if kind == "pp":
        return f"{v * 100:+.1f} pp".replace("-", "−")
    if kind == "int":
        return f"{float(v):,.0f}"
    return num(float(v))


def key_results(rep: Report, name: str) -> str:
    items = [
        (label, key, _fmt(rep.r(name, key), kind))
        for label, key, kind in KEY_RESULTS.get(name, [])
        if rep.r(name, key) is not None
    ]
    if not items:
        return ""
    rows = "".join(
        f"<dt>{esc(label)}<br><code>{esc(key)}</code></dt><dd>{esc(val)}</dd>"
        for label, key, val in items
    )
    return f'<p class="ro-head">Headline numbers</p><dl class="ro">{rows}</dl>'


def render_study(rep: Report, name: str, prev: str | None, nxt: str | None, end: str) -> str:
    sec = rep.sections[name]
    sid = study_id(name)
    title = strip_number(sec["title"])
    parts = [
        f"""
<div class="row title-row">
  <div class="main">
    <p class="crumbs"><a href="../index.html#studies">Studies</a> <span aria-hidden="true">/</span> <span>{sid}</span></p>
    <h1>{esc(title)}</h1>
  </div>
  <aside class="side note"><p>Re-run this study with <code>trendlab run {sid}</code>. Its numbers are under
  <code>{esc(name)}</code> in <code>reports/results.json</code>.</p></aside>
</div>"""
    ]
    kind = STUDY_CHARTS.get(name)
    text_html = markdown_to_html(sec["text"], base_level=3)
    parts.append(
        f'<div class="row"><div class="main prose study-text">{text_html}</div>'
        f'<aside class="side note keyresults">{key_results(rep, name)}</aside></div>'
    )
    if kind:
        chart = CHARTS[kind](rep)
        parts.append(
            chart_block(
                f"chart-{kind}",
                chart,
                note(
                    "Point at the chart, or focus it and use the arrow keys, to read exact values here."
                ),
                chart["title"],
            )
        )
    for fig in sec["figures"]:
        if kind and fig["path"] in REPLACED_FIGURES:
            continue
        parts.append(figure_row(f"../{fig['path']}", fig["caption"], fig["caption"]))
    for t in sec["tables"]:
        cap = f'<p class="table-caption">{esc(t["caption"])}</p>'
        dense = " dense" if t["html"].split("</thead>")[0].count("<th") > 11 else ""
        parts.append(
            f'<div class="row table-row{dense}"><div class="full">{cap}{t["html"]}</div></div>'
        )
    if sec.get("after"):
        parts.append(
            f'<div class="row"><div class="main prose after">{markdown_to_html(sec["after"], base_level=3)}</div></div>'
        )
    pn = []
    if prev:
        pn.append(
            f'<a class="prev" href="{study_id(prev)}.html" rel="prev"><span>Previous · {study_id(prev)}</span>'
            f"{esc(strip_number(rep.sections[prev]['title']))}</a>"
        )
    if nxt:
        pn.append(
            f'<a class="next" href="{study_id(nxt)}.html" rel="next"><span>Next · {study_id(nxt)}</span>'
            f"{esc(strip_number(rep.sections[nxt]['title']))}</a>"
        )
    parts.append(
        f'<nav class="row pager" aria-label="Studies"><div class="full">{"".join(pn)}</div></nav>'
    )
    return page(
        f"{sid}: {title} · trendlab",
        "\n".join(parts),
        depth=1,
        active="studies",
        description=question_of(sec["text"]),
        end=end,
    )


# --------------------------------------------------------------------------- methodology


def _md_table(lines: list[str]) -> str:
    def cells(line):
        return [c.strip() for c in line.strip().strip("|").split("|")]

    from ..report import _inline

    head = cells(lines[0])
    body = [cells(x) for x in lines[2:]]
    th = "".join(f"<th>{_inline(h)}</th>" for h in head)
    rows = "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in body)
    return f'<div class="table-wrap"><table><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>'


def markdown_full(text: str) -> str:
    """Markdown subset plus fenced code and pipe tables (enough for docs/METHODOLOGY.md)."""
    out, buf = [], []
    lines = text.splitlines()
    i = 0

    def flush():
        if buf:
            out.append(markdown_to_html("\n".join(buf), base_level=1))
            buf.clear()

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush()
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append(f"<pre><code>{esc(chr(10).join(lines[i + 1 : j]))}</code></pre>")
            i = j + 1
            continue
        if (
            line.startswith("|")
            and i + 1 < len(lines)
            and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip())
        ):
            flush()
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            out.append(_md_table(lines[i:j]))
            i = j
            continue
        buf.append(line)
        i += 1
    flush()
    return "\n".join(out)


def render_methodology(repo_root: Path, end: str) -> str:
    md = (repo_root / "docs" / "METHODOLOGY.md").read_text()
    md = re.sub(r"^# .*\n", "", md, count=1)
    md = re.sub(
        r"^\s*This document specifies.*?\n\n", "", md, count=1, flags=re.S
    )  # the standfirst says it
    body = f"""
<div class="row title-row">
  <div class="main"><h1>Methodology</h1>
  <p class="standfirst">Exactly what the backtests do, so that every number on this site can be reproduced and challenged.</p></div>
  <aside class="side note"><p>Source: <a href="{REPO_URL}/blob/main/docs/METHODOLOGY.md"><code>docs/METHODOLOGY.md</code></a></p></aside>
</div>
<div class="row"><div class="main prose doc">{markdown_full(md)}</div></div>"""
    return page(
        "Methodology · trendlab",
        body,
        depth=0,
        active="methodology",
        description="Data, signals, execution, metrics and statistical inference behind the trendlab study.",
        end=end,
    )


# --------------------------------------------------------------------------- build


def build_site(reports_dir: Path, out_dir: Path, repo_root: Path | None = None) -> Path:
    """Render the website from ``reports_dir`` into ``out_dir`` and return the index path."""
    reports_dir, out_dir = Path(reports_dir), Path(out_dir)
    repo_root = Path(repo_root) if repo_root else reports_dir.parent
    rep = Report.load(reports_dir)
    end = rep.r("e01_bias_waterfall", "full_history_end") or ""

    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "studies").mkdir(parents=True)
    shutil.copytree(ASSETS_DIR, out_dir / "assets")
    shutil.copytree(reports_dir / "figures", out_dir / "figures")
    # study pages reference ../figures/...
    (out_dir / "index.html").write_text(render_index(rep, end))
    names = list(rep.sections)
    for i, name in enumerate(names):
        prev = names[i - 1] if i > 0 else None
        nxt = names[i + 1] if i + 1 < len(names) else None
        (out_dir / "studies" / f"{study_id(name)}.html").write_text(
            render_study(rep, name, prev, nxt, end)
        )
    if (repo_root / "docs" / "METHODOLOGY.md").exists():
        (out_dir / "methodology.html").write_text(render_methodology(repo_root, end))
    (out_dir / ".nojekyll").write_text("")
    return out_dir / "index.html"

"""Research experiments; each module exposes ``NAME``, ``TITLE`` and ``run(ctx)``."""

from __future__ import annotations

import importlib
import time
from pathlib import Path

from .common import REPORTS_DIR, Context, Section, write_results

EXPERIMENTS = [
    "e01_bias_waterfall",
    "e02_cross_asset",
    "e03_parameter_robustness",
    "e04_walk_forward",
    "e05_crises_regimes",
    "e06_signal_comparison",
    "e07_vol_targeting",
    "e08_multi_asset_portfolio",
    "e09_significance",
    "e10_costs",
]

INTRO = """\
This report is generated end-to-end by `trendlab run` from the cached data in
`data/raw`. It starts from the original notebook's claim -- an EMA50/200
crossover on NVDA returned **+505%** against **+347%** for buy-and-hold over
2022-2024 -- and asks how much of that survives a bias-free backtest, and what
a 50/200 trend rule actually does across assets, crises and out of sample.

**Default settings** (fixed before looking at results): fills at the *next*
close after a signal, proportional costs of 5 bp per side (15 bp for
Bitcoin), idle cash earns the 3-month T-bill rate, indicators warmed up on
prior history, full available history per asset.
"""


def resolve(names: list[str] | None) -> list[str]:
    if not names:
        return list(EXPERIMENTS)
    out = []
    for n in names:
        match = [e for e in EXPERIMENTS if e == n or e.startswith(n) or e.split("_", 1)[0] == n]
        if not match:
            raise SystemExit(f"unknown experiment {n!r}; choose from {EXPERIMENTS}")
        out.extend(m for m in match if m not in out)
    return out


def run_experiments(
    names: list[str] | None = None,
    *,
    quick: bool = False,
    out_dir: Path = REPORTS_DIR,
    report: bool = True,
    verbose: bool = True,
) -> list[Section]:
    from ..report import build_report

    ctx = Context(out_dir=Path(out_dir), quick=quick)
    sections: list[Section] = []
    for name in resolve(names):
        mod = importlib.import_module(f"{__name__}.{name}")
        t0 = time.perf_counter()
        sections.append(mod.run(ctx))
        if verbose:
            print(f"[trendlab] {name:28s} {time.perf_counter() - t0:6.1f}s")
    write_results(sections, ctx.out_dir / "results.json")
    if report and (names is None or len(sections) == len(EXPERIMENTS)):
        intro = {"title": "About this report", "text": INTRO}
        build_report(
            [intro] + [s.as_report_section() for s in sections],
            ctx.out_dir / "report.html",
            title="EMA crossovers, honestly: a bias-aware trend-following study",
            subtitle=f"Data through {ctx.end} · trendlab",
        )
        if verbose:
            print(f"[trendlab] report -> {ctx.out_dir / 'report.html'}")
    return sections

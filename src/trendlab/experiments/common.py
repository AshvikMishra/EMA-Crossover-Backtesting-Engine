"""Shared plumbing for the research experiments.

Every experiment module exposes ``NAME``, ``TITLE`` and ``run(ctx) -> Section``.
A :class:`Section` carries Markdown text, figure paths and tables for the HTML
report, plus a flat ``results`` dict of headline numbers that is written to
``reports/results.json`` (the README quotes those numbers).

Default research settings (fixed up front, not tuned):

* fills at the **next close** after the signal,
* proportional costs of 5 bp per side for ETFs/stocks and 15 bp for crypto,
* idle cash earns the 3-month T-bill rate (FRED DTB3),
* full available history, ending at :data:`universe.RESEARCH_END`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from ..data import load_prices, risk_free_yield  # noqa: E402
from ..engine import BacktestConfig, BacktestResult, CostModel, run_backtest  # noqa: E402
from ..report import df_to_markdown  # noqa: E402
from ..strategies import BuyAndHold, Strategy  # noqa: E402
from ..universe import ASSETS, RESEARCH_END  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = Path(
    os.environ.get(
        "TRENDLAB_REPORTS_DIR",
        REPO_ROOT / "reports"
        if (REPO_ROOT / "pyproject.toml").exists()
        else Path.cwd() / "reports",
    )
)

ETF_COSTS = CostModel(commission_bps=1.0, half_spread_bps=2.0, slippage_bps=2.0)
CRYPTO_COSTS = CostModel(commission_bps=10.0, half_spread_bps=2.5, slippage_bps=2.5)


def costs_for(ticker: str) -> CostModel:
    return CRYPTO_COSTS if ASSETS.get(ticker, {}).get("class") == "Crypto" else ETF_COSTS


@dataclass
class Section:
    """One report section produced by an experiment."""

    name: str
    title: str
    text: str = ""
    figures: list[tuple[str, Path]] = field(default_factory=list)
    tables: list[tuple[str, pd.DataFrame]] = field(default_factory=list)
    after: str = ""
    results: dict[str, Any] = field(default_factory=dict)
    data: dict[str, pd.DataFrame] = field(default_factory=dict)  # chart series for the website

    def as_report_section(self) -> dict:
        return {
            "title": self.title,
            "text": self.text,
            "figures": self.figures,
            "tables": self.tables,
            "after": self.after,
        }


@dataclass
class Context:
    """Data access, defaults and output locations shared by all experiments."""

    out_dir: Path = REPORTS_DIR
    end: str = RESEARCH_END
    quick: bool = False  # smaller grids / fewer simulations for smoke tests
    _prices: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    _rf: pd.Series | None = field(default=None, repr=False)

    @property
    def fig_dir(self) -> Path:
        d = self.out_dir / "figures"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def table_dir(self) -> Path:
        d = self.out_dir / "tables"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def prices(self, ticker: str) -> pd.DataFrame:
        if ticker not in self._prices:
            self._prices[ticker] = load_prices(ticker, end=self.end)
        return self._prices[ticker]

    @property
    def rf(self) -> pd.Series:
        if self._rf is None:
            self._rf = risk_free_yield()
        return self._rf

    def config(self, ticker: str, **overrides) -> BacktestConfig:
        kw = {"execution": "next_close", "costs": costs_for(ticker), "end": self.end}
        kw.update(overrides)
        return BacktestConfig(**kw)

    def backtest(self, ticker: str, strategy: Strategy, **overrides) -> BacktestResult:
        return run_backtest(
            self.prices(ticker), strategy, self.config(ticker, **overrides), self.rf
        )

    def pair(
        self, ticker: str, strategy: Strategy, **overrides
    ) -> tuple[BacktestResult, BacktestResult]:
        """Strategy and buy-and-hold over the *same* window (the strategy's)."""
        res = self.backtest(ticker, strategy, **overrides)
        overrides = {**overrides, "start": res.start}
        bnh = self.backtest(ticker, BuyAndHold(), **overrides)
        return res, bnh

    def fig_path(self, name: str) -> Path:
        return self.fig_dir / f"{name}.png"

    def save_table(self, name: str, df: pd.DataFrame, index: bool = True) -> pd.DataFrame:
        """Write ``df`` as CSV (raw numbers) and Markdown (display) under reports/tables."""
        df.to_csv(self.table_dir / f"{name}.csv", index=index)
        (self.table_dir / f"{name}.md").write_text(df_to_markdown(df, index=index) + "\n")
        return df


def write_results(sections: list[Section], path: Path) -> None:
    payload = {s.name: s.results for s in sections}
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(payload)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True, default=_json_default) + "\n")


def weekly_growth(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Growth of $1 from per-bar returns, sampled at week ends (compact chart data)."""
    df = pd.DataFrame({k: (1.0 + v).cumprod() for k, v in series.items()})
    return df.resample("W-FRI").last().dropna(how="all")


def write_sections(sections: list[Section], out_dir: Path) -> None:
    """Persist each section's text, figures, tables and chart data for the website.

    ``sections/<name>.json`` holds the narrative and display tables (as HTML);
    ``data/<name>__<key>.csv`` holds chart series.
    """
    from ..report import df_to_html

    sec_dir, data_dir = out_dir / "sections", out_dir / "data"
    sec_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    for s in sections:
        payload = {
            "name": s.name,
            "title": s.title,
            "text": s.text,
            "after": s.after,
            "figures": [
                {"caption": cap, "path": Path(path).relative_to(out_dir).as_posix()}
                for cap, path in s.figures
            ],
            "tables": [{"caption": cap, "html": df_to_html(df)} for cap, df in s.tables],
            "data": sorted(s.data),
        }
        (sec_dir / f"{s.name}.json").write_text(json.dumps(payload, indent=1) + "\n")
        for key, df in s.data.items():
            df.to_csv(data_dir / f"{s.name}__{key}.csv", float_format="%.6g")


def _json_default(obj):
    if isinstance(obj, (pd.Timestamp,)):
        return obj.date().isoformat()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)

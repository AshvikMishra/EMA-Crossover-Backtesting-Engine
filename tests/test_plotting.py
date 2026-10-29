"""Smoke tests for plotting and the HTML report on small synthetic data."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.patches import Rectangle

from trendlab import plotting as tp
from trendlab.report import build_report, df_to_markdown, markdown_to_html

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def market():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2015-01-01", periods=600)
    r = pd.Series(rng.normal(0.0004, 0.012, len(idx)), index=idx)
    close = 100 * (1 + r).cumprod()
    fast = close.ewm(span=20, adjust=False).mean()
    slow = close.ewm(span=60, adjust=False).mean()
    exposure = (fast > slow).astype(float).shift(2).fillna(0.0)
    strat = exposure * r
    return {
        "idx": idx,
        "r": r,
        "close": close,
        "fast": fast,
        "slow": slow,
        "w": exposure,
        "s": strat,
    }


def _check(fig, path):
    assert isinstance(fig, Figure)
    assert path.exists() and path.stat().st_size > 5_000
    assert path.read_bytes()[:8] == PNG_MAGIC


def test_price_with_signals(market, tmp_path):
    p = tmp_path / "price.png"
    df = pd.DataFrame({"Close": market["close"]})
    fig = tp.price_with_signals(
        df,
        market["fast"],
        market["slow"],
        market["w"],
        "Trend rule sits out the drawdowns",
        subtitle="EMA20/60, synthetic",
        path=p,
    )
    _check(fig, p)


def test_equity_family(market, tmp_path):
    series = {"EMA20/60": market["s"], "Buy & Hold": (1 + market["r"]).cumprod() * 1e5}
    p1, p2, p3 = tmp_path / "eq.png", tmp_path / "uw.png", tmp_path / "ed.png"
    _check(tp.equity_curves(series, title="Equity", highlight="EMA20/60", path=p1), p1)
    _check(tp.underwater(series, "Drawdowns", path=p2), p2)
    fig = tp.equity_and_drawdown(series, title="Both", subtitle="synthetic", path=p3)
    _check(fig, p3)
    assert len(fig.axes) == 2


def test_series_colors_consistent():
    cmap = tp.series_colors(["Buy & Hold", "EMA50/200", "SMA50/200"])
    assert cmap["Buy & Hold"] == tp.BENCHMARK_COLOR
    assert cmap["EMA50/200"] == tp.STRATEGY_COLOR
    many = tp.series_colors([f"s{i}" for i in range(10)])
    assert len(set(list(many.values())[:8])) == 8
    assert many["s8"] == many["s9"] == tp.TOKENS["light"]["other"]


def test_bias_waterfall(tmp_path):
    steps = [("Naive", 1.8), ("Next-close fill", 1.2), ("Warm-up", 1.0), ("Costs", 0.85)]
    p = tmp_path / "wf.png"
    _check(tp.bias_waterfall(steps, "Most of the edge was bias", benchmark=1.1, path=p), p)


def test_metric_heatmap(tmp_path):
    fast, slow = [10, 20, 50, 100], [50, 100, 150, 200]
    vals = np.array([[f / s if f < s else np.nan for s in slow] for f in fast])
    df = pd.DataFrame(vals, index=pd.Index(fast, name="fast"), columns=pd.Index(slow, name="slow"))
    p1, p2 = tmp_path / "hm.png", tmp_path / "hm_div.png"
    _check(tp.metric_heatmap(df, "Sharpe surface", fmt="{:.2f}", mark=(50, 200), path=p1), p1)
    _check(tp.metric_heatmap(df - 0.3, "Excess", center=0.0, path=p2), p2)


def test_bars(tmp_path):
    df = pd.DataFrame(
        {"Strategy CAGR": [0.08, 0.05, -0.01], "B&H CAGR": [0.10, 0.04, 0.02]},
        index=["SPY", "TLT", "DBC"],
    )
    p1, p2 = tmp_path / "bar.png", tmp_path / "regime.png"
    _check(tp.bar_comparison(df, "Trend lags B&H in equities", path=p1), p1)
    reg = pd.DataFrame(
        {"Strategy": [0.02, -0.01, 0.05], "Buy & Hold": [0.01, -0.05, 0.08]},
        index=["Low vol", "Mid vol", "High vol"],
    )
    _check(tp.regime_bars(reg, "Regimes", path=p2), p2)


def test_crisis_bars(tmp_path):
    table = pd.DataFrame(
        {
            "crisis": ["GFC", "COVID", "2022 bear"],
            "bnh_return": [-0.55, -0.34, -0.25],
            "strategy_return": [-0.10, -0.12, -0.08],
            "days": [355, 23, 194],
        }
    )
    p = tmp_path / "crisis.png"
    _check(tp.crisis_bars(table, "Trend cut crisis losses", path=p), p)


def test_smile_scatter(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.normal(0.02, 0.08, 60)
    y = 0.005 + 0.4 * x + 2.0 * x**2 + rng.normal(0, 0.02, 60)
    p1, p2 = tmp_path / "smile.png", tmp_path / "smile2.png"
    _check(tp.smile_scatter(x, y, (0.005, 0.4, 2.0), "Smile", path=p1), p1)
    _check(
        tp.smile_scatter(pd.Series(x), pd.Series(y), {"alpha": 0, "beta": 1, "gamma": 2}, path=p2),
        p2,
    )


def test_rolling_hist_cost(market, tmp_path):
    roll = market["s"].rolling(126).mean() / market["s"].rolling(126).std() * np.sqrt(252)
    p1, p2, p3 = tmp_path / "roll.png", tmp_path / "hist.png", tmp_path / "cost.png"
    _check(tp.rolling_line({"EMA": roll}, "Rolling Sharpe", "Sharpe", path=p1), p1)
    null = np.random.default_rng(2).normal(0, 0.3, 2000)
    _check(tp.histogram_with_marker(null, 0.7, "Beats the null", path=p2), p2)
    cost = pd.DataFrame(
        {"SPY": [0.6, 0.5, 0.3], "BTC-USD": [1.0, 0.8, 0.5]}, index=pd.Index([0, 10, 30])
    )
    _check(tp.cost_sensitivity_line(cost, "Costs bite", ylabel="Sharpe", path=p3), p3)


def test_walk_forward(market, tmp_path):
    idx = market["idx"]
    folds = pd.DataFrame(
        {
            "test_start": idx[[100, 300, 450]],
            "test_end": idx[[299, 449, 599]],
            "fast": [20, 50, 20],
            "slow": [100, 200, 150],
        }
    )
    oos = (1 + market["s"].iloc[100:]).cumprod()
    bnh = (1 + market["r"].iloc[100:]).cumprod()
    p = tmp_path / "wf.png"
    _check(tp.walk_forward_chart(oos, bnh, folds, "OOS", path=p), p)


def test_dark_theme(market, tmp_path):
    p = tmp_path / "dark.png"
    with tp.theme(dark=True) as t:
        assert t["surface"] == "#1a1a19"
        fig = tp.equity_curves({"EMA": market["s"]}, path=p)
    assert fig.get_facecolor()[:3] == pytest.approx(to_rgb("#1a1a19"))
    _check(fig, p)


def test_no_global_side_effects():
    import matplotlib as mpl

    before = dict(mpl.rcParams)
    fig = tp.underwater(
        {"A": pd.Series([0.01, -0.02, 0.01], index=pd.bdate_range("2020", periods=3))}
    )
    assert mpl.rcParams["axes.facecolor"] == before["axes.facecolor"]
    assert isinstance(fig, Figure)


def test_build_report(tmp_path):
    png = tmp_path / "a.png"
    tp.bias_waterfall([("Naive", 0.5), ("Honest", 0.2)], "x", path=png)
    df = pd.DataFrame({"cagr": ["8.0%", "–"], "sharpe": [0.51, np.nan]}, index=["SPY", "TLT"])
    sections = [
        {
            "title": "Bias audit",
            "text": "Most of the edge was **bias**.\n\n- look-ahead\n- `same_close` fills\n",
            "figures": [("Waterfall", png)],
            "tables": [("Summary", df)],
        },
        {"title": "Out of sample <check>", "html": "<p id='raw'>raw</p>"},
    ]
    out = build_report(sections, tmp_path / "r" / "report.html", title="Trend study")
    doc = out.read_text()
    assert doc.startswith("<!doctype html>")
    assert "Bias audit" in doc and "Out of sample &lt;check&gt;" in doc
    assert "data:image/png;base64," in doc
    assert "<strong>bias</strong>" in doc and "<code>same_close</code>" in doc
    assert "<li>look-ahead</li>" in doc and "<p id='raw'>raw</p>" in doc
    assert "prefers-color-scheme:dark" in doc and 'class="table-wrap"' in doc


def test_markdown_to_html():
    h = markdown_to_html("# Head\npara *it* [l](http://x)\n\n1. one\n2. two\n\n<b>")
    assert "<h3>Head</h3>" in h
    assert "<em>it</em>" in h and '<a href="http://x">l</a>' in h
    assert "<ol><li>one</li><li>two</li></ol>" in h
    assert "&lt;b&gt;" in h


def test_df_to_markdown():
    df = pd.DataFrame(
        {"name": ["S&P | 500", "Gold"], "sharpe": [0.512, -0.1], "trades": [12, 3]},
        index=pd.Index(["SPY", "GLD"], name="ticker"),
    )
    md = df_to_markdown(df)
    lines = md.strip().splitlines()
    assert lines[0] == "| ticker | name | sharpe | trades |"
    assert lines[1] == "| :--- | :--- | ---: | ---: |"
    assert lines[2] == "| SPY | S&P \\| 500 | 0.512 | 12 |"
    assert len(lines) == 4
    assert df_to_markdown(df, index=False).splitlines()[0] == "| name | sharpe | trades |"
    assert "0.51 |" in df_to_markdown(df, float_fmt="{:.2f}")


# --------------------------------------------------------------------------- regressions


def test_returned_figure_renders_same_outside_theme(market, tmp_path):
    series = {"EMA": market["s"], "Buy & Hold": (1 + market["r"]).cumprod()}
    tp.equity_curves(series, title="t", path=tmp_path / "in.png")
    fig = tp.equity_curves(series, title="t")
    tp.save_figure(fig, tmp_path / "out.png")
    assert np.array_equal(imread(tmp_path / "in.png"), imread(tmp_path / "out.png"))


def test_all_positive_returns_are_not_mistaken_for_equity():
    idx = pd.bdate_range("2020", periods=250)
    cash = pd.Series(np.linspace(1e-4, 2e-4, 250), index=idx)
    assert tp._as_growth(cash).iloc[-1] == pytest.approx((1 + cash).prod())
    assert (tp._as_returns(cash) == cash).all()
    equity = pd.Series(np.linspace(1.0, 2.0, 250), index=idx)
    assert tp._as_growth(equity).iloc[-1] == pytest.approx(2.0)


def test_price_markers_sit_on_fill_close():
    idx = pd.bdate_range("2021", periods=40)
    close = pd.Series(np.linspace(100, 140, 40), index=idx)
    w = pd.Series(0.0, index=idx)
    w.iloc[10:20] = 1.0  # held over bars 10..19 -> bought at close 9, sold at close 19
    fig = tp.price_with_signals(close, close, close, w, markers=True)
    lines = {ln.get_label(): ln for ln in fig.axes[0].get_lines()}
    assert list(lines["Entry"].get_xdata()) == [idx[9]]
    assert list(lines["Entry"].get_ydata()) == [close.iloc[9]]
    assert list(lines["Exit"].get_xdata()) == [idx[19]]


def test_heatmap_marks_fast_slow_in_grid_pivot_layout():
    fast, slow = [10.0, 50.0], [100.0, 200.0]
    df = pd.DataFrame(
        [[0.1, 0.2], [0.3, 0.4]],
        index=pd.Index(slow, name="slow"),
        columns=pd.Index(fast, name="fast"),
    )
    fig = tp.metric_heatmap(df, mark=(50, 200))
    ax = fig.axes[0]
    rects = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(rects) == 1 and rects[0].get_xy() == (0.5, 0.5)  # row slow=200, col fast=50
    assert [t.get_text() for t in ax.get_xticklabels()] == ["10", "50"]
    plain = pd.DataFrame([[1.0, 2.0]], index=pd.Index([50], name="fast"), columns=[100, 200])
    rects = [p for p in tp.metric_heatmap(plain).axes[0].patches if isinstance(p, Rectangle)]
    assert len(rects) == 1 and rects[0].get_xy() == (0.5, -0.5)


def test_walk_forward_labels_and_aligned_benchmark(market):
    idx = market["idx"]
    folds = pd.DataFrame(
        {
            "test_start": idx[[300, 450]],
            "test_end": idx[[449, 599]],
            "chosen": ["TSMOM 12M", "Faber 10M SMA"],
            "fast": [np.nan, np.nan],
            "slow": [np.nan, np.nan],
        }
    )
    oos = market["s"].iloc[300:]
    bnh = (1 + market["r"]).cumprod()  # full history: must be clipped and rebased
    fig = tp.walk_forward_chart(oos, bnh, folds, bnh_label="Buy & hold")
    ax = fig.axes[0]
    texts = {t.get_text() for t in ax.texts}
    assert {"TSMOM 12M", "Faber 10M SMA"} <= texts and "nan/nan" not in texts
    line = next(ln for ln in ax.get_lines() if ln.get_label() == "Buy & hold")
    assert pd.Timestamp(line.get_xdata()[0]) == idx[300]
    assert line.get_ydata()[0] == pytest.approx(1.0)


def test_histogram_p_value_and_alternative():
    null = np.arange(99, dtype=float)  # 0..98
    fig = tp.histogram_with_marker(null, 89.0)
    assert any("p = 0.110" in t.get_text() for t in fig.axes[0].texts)  # (1 + 10) / 100
    with pytest.raises(ValueError):
        tp.histogram_with_marker(null, 1.0, alternative="two-sided")


def test_tables_years_missing_and_duplicate_columns():
    df = pd.DataFrame(
        {"ret": [0.1, None], "n": pd.array([12345, pd.NA], dtype="Int64")},
        index=pd.Index([2008, 2009], name="year"),
    )
    lines = df_to_markdown(df).splitlines()
    assert lines[2] == "| 2008 | 0.100 | 12,345 |"
    assert lines[3] == "| 2009 | – | – |"
    dup = pd.DataFrame([[1.0, "a"]], columns=["x", "x"])
    assert df_to_markdown(dup, index=False).splitlines()[1] == "| ---: | :--- |"

"""Experiment 8: trend following as a diversified multi-asset portfolio.

The original notebook ran one rule on one hindsight-picked stock. Trend
following as practised (and as studied by Moskowitz, Ooi & Pedersen 2012 and
Hurst, Ooi & Pedersen 2017) is a *portfolio* of many small, weakly correlated
trend bets. This experiment builds that portfolio from the ten CORE ETFs and
asks what it adds to a conventional 60/40 investor.

Pre-specified design:

* Sleeves: EMA50/200 (the notebook's rule) and, as the literature's
  alternative, 12-month time-series momentum, one per CORE ETF, long or cash.
* Weights: inverse 63-day volatility among assets whose signal is warmed up,
  reset at month ends; a variant scales the whole book to a 10% ex-post vol
  target with leverage capped at 2x (financed at the T-bill rate).
* Benchmarks: 60/40 SPY/IEF, equal-weight buy-and-hold of CORE and SPY, all
  rebalanced monthly with drift, plus an 80% 60/40 / 20% trend blend.
* Control: the same inverse-vol book with every signal switched to "always
  long" (risk-parity-lite buy-and-hold). Comparing trend against it separates
  what the *signal* adds from what diversification and the inverse-vol tilt
  towards bonds add. Its sleeves (and the TSMOM book's) are padded to the
  EMA50/200 warm-up, so every sleeve joins all three books on the same bar.
* Common window: the first date at which both SPY and IEF trend signals are
  warmed up (so the 60/40 really holds bonds and the trend book has both stock
  and bond sleeves). Later-listing ETFs (EEM, GLD, VNQ, DBC) join the static
  benchmarks as soon as they trade and the inverse-vol books once the EMA50/200
  signal is warm.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import plotting
from ..engine import evaluation_start
from ..metrics import format_table, max_drawdown, summary
from ..portfolio import (
    blend,
    correlation_matrix,
    portfolio_backtest,
    static_mix_backtest,
)
from ..report import df_to_markdown
from ..stats import bootstrap_sharpe_difference
from ..strategies import BuyAndHold, EMACrossover, Strategy, TimeSeriesMomentum
from ..universe import ASSETS, CORE
from .common import ETF_COSTS, Context, Section, costs_for

NAME = "e08_multi_asset_portfolio"
TITLE = "8. Trend as a portfolio: ten markets instead of one stock"

VOL_WINDOW = 63
PORT_VOL_TARGET = 0.10
PORT_MAX_LEV = 2.0
BLEND_W = 0.8  # weight of 60/40 in the blend
CRISIS_YEARS = ["2008", "2020", "2022"]

TREND = "Trend EMA50/200"
TSMOM = "Trend TSMOM 12M"
TREND_VT = "Trend EMA + 10% VT"
SIXTY = "60/40 benchmark"
EW = "Equal-weight B&H"
SPY = "SPY B&H"
IVBH = "Inverse-vol B&H"
BLEND = "80% 60/40 + 20% trend"

CLASS_ORDER = ["US equity", "Intl equity", "Real estate", "Bonds", "Commodities"]


@dataclass(frozen=True, repr=False)
class _JoinAfter(Strategy):
    """``base`` with its warm-up padded to ``bars``, so a sleeve joins the book
    on the same bar as a slower-warming strategy's sleeve on the same asset."""

    base: Strategy
    bars: int

    @property
    def name(self) -> str:
        return self.base.name

    @property
    def warmup(self) -> int:
        return max(self.base.warmup, self.bars)

    def _raw_target(self, prices: pd.DataFrame) -> pd.Series:
        return self.base._raw_target(prices)


def _save(ctx: Context, name: str, raw: pd.DataFrame, display: pd.DataFrame) -> None:
    ctx.save_table(name, raw)
    (ctx.table_dir / f"{name}.md").write_text(df_to_markdown(display) + "\n")


def _compound(r: pd.Series) -> float:
    return float((1.0 + r).prod() - 1.0)


def _monthly(r: pd.Series) -> pd.Series:
    return (1.0 + r).resample("ME").prod() - 1.0


def _class_exposure(weights: pd.DataFrame) -> pd.DataFrame:
    """Monthly average exposure per asset class plus the T-bill remainder."""
    cls = {t: ASSETS[t]["class"] for t in weights.columns}
    by = weights.T.groupby(cls).sum().T
    by = by[[c for c in CLASS_ORDER if c in by.columns]]
    m = by.resample("ME").mean()
    m["Cash (T-bills)"] = (1.0 - m.sum(axis=1)).clip(lower=0.0)
    return m


def _exposure_chart(exp: pd.DataFrame, title: str, subtitle: str, path) -> None:
    with plotting.theme() as t:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        cols = list(exp.columns)
        colors = [t["categorical"][i] for i in range(len(cols) - 1)] + [t["grid"]]
        ax.stackplot(
            exp.index,
            exp.to_numpy().T,
            labels=cols,
            colors=colors,
            alpha=0.95,
            lw=0,
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Share of capital")
        plotting._pct_axis(ax.yaxis)
        plotting._date_axis(ax)
        ax.margins(x=0)
        fig.legend(loc="outside lower center", ncols=len(cols))
        plotting._titles(ax, title, subtitle)
        plotting.save_figure(fig, path)


def run(ctx: Context) -> Section:
    n_boot = 300 if ctx.quick else 2000
    prices = {t: ctx.prices(t) for t in CORE}
    costs = {t: costs_for(t) for t in CORE}
    ema, tsmom = EMACrossover(50, 200), TimeSeriesMomentum(252)
    start = max(evaluation_start(prices[t], [ema, tsmom]) for t in ("SPY", "IEF"))
    # every inverse-vol book adds a sleeve on the bar the EMA50/200 sleeve joins
    tsmom_book = _JoinAfter(tsmom, ema.warmup)
    always_long = _JoinAfter(BuyAndHold(), ema.warmup)
    common = {
        "costs": costs,
        "rf_yield": ctx.rf,
        "start": start,
        "end": ctx.end,
        "calendar": "SPY",
    }
    trend_kw = {"weighting": "inverse_vol", "vol_window": VOL_WINDOW, "rebalance": "ME"}

    trend = portfolio_backtest(prices, ema, name=TREND, **trend_kw, **common)
    trend_ts = portfolio_backtest(prices, tsmom_book, name=TSMOM, **trend_kw, **common)
    trend_vt = portfolio_backtest(
        prices,
        ema,
        name=TREND_VT,
        vol_target=PORT_VOL_TARGET,
        max_leverage=PORT_MAX_LEV,
        **trend_kw,
        **common,
    )
    ivbh = portfolio_backtest(prices, always_long, name=IVBH, **trend_kw, **common)
    sixty = static_mix_backtest(prices, {"SPY": 0.6, "IEF": 0.4}, name=SIXTY, **common)
    ew = static_mix_backtest(prices, {t: 1.0 / len(CORE) for t in CORE}, name=EW, **common)
    spy = static_mix_backtest(prices, {"SPY": 1.0}, name=SPY, **common)
    rf = sixty.rf
    blend_r = blend(sixty.returns, trend.returns, BLEND_W, "ME", cost_bps=ETF_COSTS.bps)
    blend_r = blend_r.rename(BLEND)

    streams = {
        TREND: trend.returns,
        TSMOM: trend_ts.returns,
        TREND_VT: trend_vt.returns,
        SIXTY: sixty.returns,
        BLEND: blend_r,
        IVBH: ivbh.returns,
        EW: ew.returns,
        SPY: spy.returns,
    }
    results = {
        TREND: trend,
        TSMOM: trend_ts,
        TREND_VT: trend_vt,
        SIXTY: sixty,
        IVBH: ivbh,
        EW: ew,
        SPY: spy,
    }

    # ------------------------------------------------------------------ metrics
    rows = {}
    for name, r in streams.items():
        res = results.get(name)
        m = summary(
            r,
            rf=rf,
            periods=252,
            exposure=None if res is None else res.gross_exposure,
            turnover=None if res is None else res.turnover,
            benchmark=sixty.returns,
        )
        rows[name] = m
    keys = [
        "cagr",
        "vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "exposure",
        "turnover_per_year",
        "correlation",
        "beta",
    ]
    perf = pd.DataFrame(rows).T
    perf_disp = format_table(rows, keys).rename(
        columns={
            "cagr": "CAGR",
            "vol": "Vol",
            "sharpe": "Sharpe",
            "sortino": "Sortino",
            "max_drawdown": "Max DD",
            "calmar": "Calmar",
            "exposure": "Avg gross exposure",
            "turnover_per_year": "Turnover / yr",
            "correlation": "Corr. to 60/40 (daily)",
            "beta": "Beta to 60/40",
        }
    )
    perf_disp = perf_disp.map(lambda v: plotting._minus(str(v)))  # U+2212 like the figures
    perf_disp.index.name = "Portfolio"
    _save(ctx, f"{NAME}_performance", perf, perf_disp)

    # single-sleeve Sharpe ratios vs the portfolio: where does the Sharpe come from?
    sleeve_rows = {}
    for t in CORE:
        st = max(start, evaluation_start(prices[t], [ema]))
        res, bnh = ctx.pair(t, ema, start=st)
        sleeve_rows[t] = {
            "class": ASSETS[t]["class"],
            "start": res.start.date().isoformat(),
            "ema_sharpe": res.summary()["sharpe"],
            "bh_sharpe": bnh.summary()["sharpe"],
        }
    sleeves = pd.DataFrame(sleeve_rows).T
    sleeves[["ema_sharpe", "bh_sharpe"]] = sleeves[["ema_sharpe", "bh_sharpe"]].astype(float)
    sleeve_disp = pd.DataFrame(
        {
            "Class": sleeves["class"],
            "From": sleeves["start"],
            "EMA50/200 Sharpe": sleeves["ema_sharpe"].map(lambda v: plotting._minus(f"{v:.2f}")),
            "B&H Sharpe": sleeves["bh_sharpe"].map(lambda v: plotting._minus(f"{v:.2f}")),
        }
    )
    sleeve_disp.index.name = "Sleeve"
    _save(ctx, f"{NAME}_sleeves", sleeves, sleeve_disp)

    # ------------------------------------------------------------------ correlations
    corr_set = {
        "Trend EMA": trend.returns,
        "Trend TSMOM": trend_ts.returns,
        "60/40": sixty.returns,
        "80/20 blend": blend_r,
        "EW B&H": ew.returns,
        "SPY": spy.returns,
        "IEF": sixty.asset_returns["IEF"],
        "GLD": prices["GLD"]["Close"].pct_change().loc[start : ctx.end].iloc[1:],
    }
    corr = correlation_matrix(corr_set, freq="ME")
    _save(ctx, f"{NAME}_correlations", corr, corr.map(lambda v: plotting._minus(f"{v:.2f}")))
    fig_corr = ctx.fig_path(f"{NAME}_correlations")
    rho_spy = float(corr.loc["Trend EMA", "SPY"])
    rho_6040 = float(corr.loc["Trend EMA", "60/40"])
    plotting.metric_heatmap(
        corr,
        f"Monthly correlation of the trend book: {plotting._minus(f'{rho_spy:.2f}')} to SPY, "
        f"{plotting._minus(f'{rho_6040:.2f}')} to 60/40",
        fmt="{:.2f}",
        center=0.0,
        mark=None,
        subtitle=f"Pearson correlation of monthly returns, {start.year}-{ctx.end[:4]}",
        xlabel=" ",
        ylabel=" ",
        path=fig_corr,
    )

    # up/down-month behaviour vs 60/40. Correlations conditional on the sign of
    # 60/40 are mechanically lower than the unconditional one (truncation of x),
    # so compare up- and down-month *betas* (OLS slopes are not attenuated by
    # selecting on x) to ask whether the book is convex.
    m_trend, m_6040 = _monthly(trend.returns), _monthly(sixty.returns)
    m_spy, m_blend = _monthly(spy.returns), _monthly(blend_r)
    down = m_6040 < 0
    rho_down = float(m_trend[down].corr(m_6040[down]))
    rho_up = float(m_trend[~down].corr(m_6040[~down]))

    def _beta(y: pd.Series, x: pd.Series) -> float:
        return float(np.cov(y, x)[0, 1] / np.var(x, ddof=1))

    beta_all = _beta(m_trend, m_6040)
    beta_down = _beta(m_trend[down], m_6040[down])
    beta_up = _beta(m_trend[~down], m_6040[~down])

    # ------------------------------------------------------------------ worst months
    worst = m_6040.nsmallest(10).index
    worst_df = pd.DataFrame(
        {
            "60/40": m_6040[worst],
            "SPY": m_spy[worst],
            "Trend EMA": m_trend[worst],
            "80/20 blend": m_blend[worst],
        }
    )
    worst_df.index = worst_df.index.strftime("%Y-%m")
    worst_df.index.name = "Month"
    worst_disp = worst_df.apply(lambda c: c.map(lambda v: plotting._minus(f"{v:.1%}")))
    _save(ctx, f"{NAME}_worst_months", worst_df, worst_disp)
    n_worst_pos = int((worst_df["Trend EMA"] > 0).sum())
    avg_worst_6040 = float(worst_df["60/40"].mean())
    avg_worst_trend = float(worst_df["Trend EMA"].mean())
    avg_worst_blend = float(worst_df["80/20 blend"].mean())
    trend_worst_month = m_trend.idxmin()

    # ------------------------------------------------------------------ crisis years
    crisis_rows = {}
    year_streams = {
        "SPY": spy.returns,
        "IEF": sixty.asset_returns["IEF"],
        "60/40": sixty.returns,
        "Trend EMA": trend.returns,
        "Trend TSMOM": trend_ts.returns,
        "80/20 blend": blend_r,
    }
    for y in CRISIS_YEARS:
        row = {}
        for k, r in year_streams.items():
            ry = r.loc[y]
            row[k] = _compound(ry) if len(ry) else np.nan
            row[f"{k} max DD"] = max_drawdown(ry) if len(ry) else np.nan
        crisis_rows[y] = row
    crisis = pd.DataFrame(crisis_rows).T
    crisis.index.name = "Calendar year"
    ret_cols = list(year_streams)
    crisis_disp = pd.DataFrame(index=crisis.index)
    for k in ret_cols:
        crisis_disp[k] = [
            plotting._minus(f"{a:.1%} ({b:.0%})")
            for a, b in zip(crisis[k], crisis[f"{k} max DD"], strict=True)
        ]
    _save(ctx, f"{NAME}_crisis_years", crisis, crisis_disp)
    fig_crisis = ctx.fig_path(f"{NAME}_crisis_years")
    bars = crisis[["SPY", "IEF", "60/40", "Trend EMA", "80/20 blend"]].rename(
        columns={"60/40": "60/40 benchmark"}
    )
    c22 = crisis.loc["2022"]
    crisis_title = (
        "In 2022 stocks and bonds fell together; the trend book "
        + ("held up" if c22["Trend EMA"] > c22["60/40"] else "did not help")
        if c22["SPY"] < 0 and c22["IEF"] < 0
        else "Calendar-year returns in the three stress years"
    )
    if crisis.loc["2020", "Trend EMA"] < crisis.loc["2020", "60/40"]:
        crisis_title += ", but lagged in the V-shaped 2020"
    fig = plotting.regime_bars(
        bars,
        crisis_title,
        subtitle="Calendar-year total return; 60/40 = SPY/IEF rebalanced monthly",
        ylabel="Calendar-year return",
        colors={
            "SPY": plotting.CATEGORICAL[6],
            "IEF": plotting.CATEGORICAL[3],
            "Trend EMA": plotting.CATEGORICAL[0],
            "80/20 blend": plotting.CATEGORICAL[1],
        },
        path=None,
    )
    with plotting.theme():  # the library legend wraps at four entries; keep one row
        for leg in list(fig.legends):
            leg.remove()
        fig.legend(loc="outside lower center", ncols=len(bars.columns))
        plotting.save_figure(fig, fig_crisis)

    # where the stress-year results came from (arithmetic sum of daily contributions)
    eq_sleeves = [t for t in CORE if ASSETS[t]["class"] in ("US equity", "Intl equity")]
    bond_sleeves = [t for t in CORE if ASSETS[t]["class"] == "Bonds"]
    c08 = trend.contributions.loc["2008"].sum()
    bonds08, eq08 = float(c08[bond_sleeves].sum()), float(c08[eq_sleeves].sum())
    # 2020: the rebound (Apr-Dec) and how fast equity exposure came back
    reb20 = {"t": _compound(trend.returns.loc["2020-04":"2020-12"])}
    reb20["s"] = _compound(sixty.returns.loc["2020-04":"2020-12"])
    eq_w20 = trend.weights.loc["2020", eq_sleeves + ["VNQ"]].sum(axis=1).resample("ME").mean()
    eq_w20_feb, eq_w20_low = float(eq_w20.loc["2020-02"].iloc[0]), float(eq_w20.min())
    eq_w20_dec = float(eq_w20.iloc[-1])

    # 2022 attribution of the trend book (arithmetic sum of daily contributions)
    contrib22 = trend.contributions.loc["2022"].sum()
    cls = {t: ASSETS[t]["class"] for t in CORE}
    by_class22 = contrib22[CORE].groupby(cls).sum()
    cash22 = float(contrib22["cash"])
    w22 = trend.weights.loc["2022"]
    exp22 = float(w22.sum(axis=1).mean())
    commod22 = float(by_class22.get("Commodities", 0.0))
    risk22 = float(contrib22[eq_sleeves + ["VNQ"]].sum())
    risk22_q1 = float(
        trend.contributions.loc["2022-01":"2022-03", eq_sleeves + ["VNQ"]].sum().sum()
    )
    risk22_after = risk22 - risk22_q1
    # weights are held over each bar's close-to-close return, so a sleeve's last
    # 2022 bar with weight is the close at which it was sold. If none is held at
    # year end, name the sleeves that exited in the same month as the last one.
    w_risk22 = trend.weights.loc["2022", eq_sleeves + ["VNQ"]]
    held22 = w_risk22.gt(0.0).any(axis=1)
    if held22.any() and not held22.iloc[-1]:
        exit22 = held22[held22].index[-1]
        last_bar22 = {
            t: w_risk22.index[w_risk22[t].gt(0.0)][-1]
            for t in w_risk22
            if w_risk22[t].gt(0.0).any()
        }
        last_out22 = [t for t, d in last_bar22.items() if d.to_period("M") == exit22.to_period("M")]
    else:
        exit22, last_out22 = None, []
    iv_bond_w = float(ivbh.weights[bond_sleeves].sum(axis=1).mean())
    rho_spy_ief = float(corr.loc["SPY", "IEF"])
    attr = pd.concat([by_class22, pd.Series({"Cash (T-bills)": cash22})])
    attr = attr.reindex([c for c in CLASS_ORDER if c in attr.index] + ["Cash (T-bills)"])
    attr_df = pd.DataFrame(
        {"contribution": attr, "avg_weight": w22.T.groupby(cls).sum().T.mean().reindex(attr.index)}
    )
    attr_df.loc["Cash (T-bills)", "avg_weight"] = 1.0 - exp22
    attr_df.index.name = "Sleeve (2022)"
    attr_disp = pd.DataFrame(
        {
            "Avg weight": attr_df["avg_weight"].map(lambda v: f"{v:.0%}"),
            "Contribution": attr_df["contribution"].map(lambda v: plotting._minus(f"{v:+.1%}")),
        }
    )
    _save(ctx, f"{NAME}_attribution_2022", attr_df, attr_disp)

    # ------------------------------------------------------------------ figures
    fig_eq = ctx.fig_path(f"{NAME}_equity")
    s_t, s_6, s_b = (
        perf.loc[TREND, "sharpe"],
        perf.loc[SIXTY, "sharpe"],
        perf.loc[BLEND, "sharpe"],
    )
    plotting.equity_and_drawdown(
        {TREND: trend.returns, SIXTY: sixty.returns, BLEND: blend_r},
        title=(
            f"The trend book's edge is in drawdowns: max DD "
            f"{plotting._pct(perf.loc[TREND, 'max_drawdown'])} vs "
            f"{plotting._pct(perf.loc[SIXTY, 'max_drawdown'])} for 60/40"
            if perf.loc[TREND, "max_drawdown"] > perf.loc[SIXTY, "max_drawdown"]
            else f"The trend book does not reduce drawdowns: max DD "
            f"{plotting._pct(perf.loc[TREND, 'max_drawdown'])} vs "
            f"{plotting._pct(perf.loc[SIXTY, 'max_drawdown'])} for 60/40"
        ),
        subtitle=(
            f"Ten CORE ETFs, EMA50/200 sleeves, inverse-vol weights, monthly; Sharpe "
            f"{s_t:.2f} (trend), {s_6:.2f} (60/40), {s_b:.2f} (blend); {start.date()} to {ctx.end}"
        ),
        colors={TREND: plotting.CATEGORICAL[0], BLEND: plotting.CATEGORICAL[1]},
        path=fig_eq,
    )
    fig_exp = ctx.fig_path(f"{NAME}_exposure")
    exp = _class_exposure(trend.weights)
    cash_all = float(exp["Cash (T-bills)"].mean())
    cash_stress = float(pd.concat([exp.loc[y, "Cash (T-bills)"] for y in ("2008", "2022")]).mean())
    _exposure_chart(
        exp,
        f"The trend book holds {cash_all:.0%} in T-bills on average, {cash_stress:.0%} in "
        "2008 and 2022",
        "EMA50/200 trend book: monthly average capital by asset class (inverse-vol weights)",
        fig_exp,
    )

    # ------------------------------------------------------------------ tests
    bt_blend = bootstrap_sharpe_difference(blend_r, sixty.returns, rf=rf, n_boot=n_boot)
    bt_trend = bootstrap_sharpe_difference(trend.returns, sixty.returns, rf=rf, n_boot=n_boot)
    bt_iv = bootstrap_sharpe_difference(trend.returns, ivbh.returns, rf=rf, n_boot=n_boot)

    # ------------------------------------------------------------------ text
    def pct(v, d=0):
        return plotting._minus(f"{v:.{d}%}")

    def spct(v, d=1):
        return plotting._minus(f"{v:+.{d}%}")

    def f2(v):
        return plotting._minus(f"{round(float(v), 2) + 0.0:.2f}")

    p = perf
    mean_sleeve = float(sleeves["ema_sharpe"].mean())
    mean_sleeve_bh = float(sleeves["bh_sharpe"].mean())
    best_sleeve = sleeves["ema_sharpe"].idxmax()
    n_sleeve_below = int((sleeves["ema_sharpe"] < p.loc[TREND, "sharpe"]).sum())
    beats = "beats" if p.loc[TREND, "sharpe"] > p.loc[SIXTY, "sharpe"] else "trails"
    blend_dir = "raises" if s_b > s_6 else "lowers"
    stocks_bonds_22 = c22["SPY"] < 0 and c22["IEF"] < 0
    held_22 = c22["Trend EMA"] > c22["60/40"]
    s_iv = float(p.loc[IVBH, "sharpe"])
    dd_t, dd_6, dd_iv = (float(p.loc[k, "max_drawdown"]) for k in (TREND, SIXTY, IVBH))
    dd_vt, vol_vt = float(p.loc[TREND_VT, "max_drawdown"]), float(p.loc[TREND_VT, "vol"])

    if exit22 is None:
        risk22_split = (
            f"{spct(risk22_q1)} in January-March and {spct(risk22_after)} after, with some "
            "still held at year end"
            if held22.any()
            else "none were held"
        )
    else:
        names22 = ", ".join(last_out22)
        last_txt = f"the last sleeve{'s' if len(last_out22) > 1 else ''} ({names22})"
        if exit22 <= pd.Timestamp("2022-03-31"):
            risk22_split = f"all in January-March; {last_txt} exited in {exit22:%B}"
        else:
            risk22_split = (
                f"{spct(risk22_q1)} in January-March and {spct(risk22_after)} after, until "
                f"{last_txt} exited in {exit22:%B}"
            )
    iv_diff = plotting._minus(f"{bt_iv['diff']:+.2f}")
    pos_txt = "never positive" if n_worst_pos == 0 else f"positive {n_worst_pos} times"
    findings = [
        f"- **Headline:** the trend book returns {pct(p.loc[TREND, 'cagr'], 1)} a year at "
        f"{pct(p.loc[TREND, 'vol'], 1)} vol (Sharpe {f2(s_t)}, max drawdown {pct(dd_t)}) vs "
        f"{pct(p.loc[SIXTY, 'cagr'], 1)} at {pct(p.loc[SIXTY, 'vol'], 1)} for 60/40 (Sharpe "
        f"{f2(s_6)}, {pct(dd_6)}). It {beats} 60/40 on Sharpe by {abs(bt_trend['diff']):.2f} "
        f"(95% CI {f2(bt_trend['ci_low'])} to {f2(bt_trend['ci_high'])}). Scaled to a 10% vol "
        f"target ({pct(vol_vt, 1)} realised, {p.loc[TREND_VT, 'exposure']:.2f}x average gross "
        f"exposure) it earns {pct(p.loc[TREND_VT, 'cagr'], 1)} with a {pct(dd_vt)} max drawdown, "
        f"so the drawdown gap {'survives' if dd_vt > 0.75 * dd_6 else 'shrinks'} at matched risk. "
        f"TSMOM 12M sleeves give Sharpe {f2(p.loc[TSMOM, 'sharpe'])}.",
        f"- **Diversification vs signal:** the ten single-asset EMA sleeves average a Sharpe "
        f"of {f2(mean_sleeve)} ({n_sleeve_below} of 10 below the portfolio's {f2(s_t)}; best "
        f"{best_sleeve}, {f2(sleeves.loc[best_sleeve, 'ema_sharpe'])}). But the *same* "
        f"inverse-vol book held permanently long has Sharpe {f2(s_iv)} (trend minus it: "
        f"{iv_diff}, 95% CI {f2(bt_iv['ci_low'])} to "
        f"{f2(bt_iv['ci_high'])}) and a {pct(dd_iv)} max drawdown.",
        f"- **Correlation and worst months:** monthly ρ = {f2(rho_spy)} to SPY and "
        f"{f2(rho_6040)} to 60/40; beta to 60/40 is {f2(beta_down)} in 60/40's down months "
        f"and {f2(beta_up)} in its up months. In 60/40's ten worst months (average "
        f"{pct(avg_worst_6040, 1)}) the trend book averaged {pct(avg_worst_trend, 1)} and was "
        f"{pos_txt}.",
        "- **Stress years:** "
        + "; ".join(
            f"{y} trend {pct(crisis.loc[y, 'Trend EMA'], 1)} vs 60/40 "
            f"{pct(crisis.loc[y, '60/40'], 1)}"
            for y in CRISIS_YEARS
        )
        + f". In 2008 the Treasury sleeves contributed {spct(bonds08)} and the equity sleeves "
        f"{spct(eq08)}. In 2022 SPY returned {pct(c22['SPY'], 1)} and IEF "
        f"{pct(c22['IEF'], 1)}; the trend book's equity and REIT sleeves contributed "
        f"{spct(risk22)} ({risk22_split}), T-bills {spct(cash22)} and commodities {spct(commod22)}.",
        f"- **Blend:** 80% 60/40 + 20% trend moves Sharpe {f2(s_6)} → {f2(s_b)} "
        f"(one-sided p = {bt_blend['p_value']:.2f}), max drawdown {pct(dd_6)} → "
        f"{pct(p.loc[BLEND, 'max_drawdown'])}, CAGR {pct(p.loc[SIXTY, 'cagr'], 1)} → "
        f"{pct(p.loc[BLEND, 'cagr'], 1)}.",
    ]

    if stocks_bonds_22 and held_22:
        y22 = (
            "In 2022, when the bond leg of 60/40 failed as a hedge, the book was on average only "
            f"{exp22:.0%} invested and lost {pct(abs(c22['Trend EMA']), 1)} instead of "
            f"{pct(abs(c22['60/40']), 1)}; being long/cash it could only step aside, not profit "
            "from shorts as long/short managed-futures programmes can (not modelled here)."
        )
    elif stocks_bonds_22:
        y22 = (
            "In 2022 stocks and bonds fell together, and the trend book did not protect "
            "better than 60/40 in this implementation."
        )
    else:
        y22 = ""
    if n_sleeve_below >= 7 and s_iv >= s_t:
        divers = (
            f"Combining ten sleeves {'roughly doubles' if s_t > 1.6 * mean_sleeve else 'lifts'} "
            f"the typical sleeve's Sharpe ({f2(mean_sleeve)} → {f2(s_t)}): the diversification "
            "effect behind Moskowitz, Ooi & Pedersen (2012) and Hurst, Ooi & Pedersen (2017), "
            "and what the single-asset NVDA framing missed. The uncomfortable part is that "
            f"diversification does the same for buy-and-hold ({f2(mean_sleeve_bh)} → {f2(s_iv)}), "
            "largely via the inverse-vol tilt to Treasuries "
            f"({iv_bond_w:.0%} of the passive book; monthly ρ(SPY, IEF) = {f2(rho_spy_ief)}), so "
            "here the EMA signal adds no Sharpe. "
            + (
                "What it adds is drawdown control at matched risk: "
                if dd_vt > dd_iv + 0.03
                else "Nor does it clearly improve drawdowns at matched risk: "
            )
            + f"{pct(dd_vt)} for the vol-targeted book vs {pct(dd_iv)} for the passive one."
        )
    elif n_sleeve_below >= 7:
        divers = (
            f"Combining ten sleeves lifts the typical sleeve's Sharpe ({f2(mean_sleeve)} → "
            f"{f2(s_t)}) and beats the same book held passively ({f2(s_iv)}): trend's "
            "risk-adjusted return comes from many imperfectly correlated trends (Moskowitz, "
            "Ooi & Pedersen 2012; Hurst, Ooi & Pedersen 2017), which is what the single-asset "
            "NVDA framing missed."
        )
    else:
        divers = (
            "The portfolio's Sharpe is not clearly above that of the typical sleeve here, so "
            "the diversification benefit reported by Moskowitz, Ooi & Pedersen (2012) and "
            "Hurst, Ooi & Pedersen (2017) is weaker in this ten-ETF, long-only universe."
        )
    convex = beta_down < 0.75 * beta_up
    corr_line = f"Down-month beta to 60/40 ({f2(beta_down)}) is " + (
        f"well below up-month beta ({f2(beta_up)}), a hint of the option-like payoff Fung "
        "& Hsieh (2001) document for trend followers."
        if convex
        else f"no lower than up-month beta ({f2(beta_up)}): the cushioning in bad months "
        "reflects low beta and a large cash and bond allocation, not the convex payoff Fung "
        "& Hsieh (2001) describe for long/short trend followers."
    )
    c20 = crisis.loc["2020"]
    covid_line = (
        f"2020 is the failure mode: after the fast COVID crash ({trend_worst_month:%B} "
        f"{pct(m_trend.min(), 1)}, the book's worst month) equity/REIT exposure fell from "
        f"{eq_w20_feb:.0%} to {eq_w20_low:.0%} and was back only to {eq_w20_dec:.0%} by "
        f"December, so April-December returned {pct(reb20['t'], 1)} vs {pct(reb20['s'], 1)} "
        "for 60/40."
        if c20["Trend EMA"] < c20["60/40"] and f"{trend_worst_month:%Y}" == "2020"
        else ""
    )
    cagr_cost = p.loc[BLEND, "cagr"] < p.loc[SIXTY, "cagr"]
    cuts = "cuts" if p.loc[BLEND, "max_drawdown"] > dd_6 else "does not cut"
    sig_note = "" if bt_blend["p_value"] < 0.05 else "not significant, "
    blend_line = (
        f"A 20% sleeve {blend_dir} 60/40's Sharpe by {abs(s_b - s_6):.2f} ({sig_note}p = "
        f"{bt_blend['p_value']:.2f}) and {cuts} its drawdown"
        + (
            f" for {(p.loc[SIXTY, 'cagr'] - p.loc[BLEND, 'cagr']) * 100:.1f} pp of CAGR: a "
            "diversifier trade-off, not a free lunch."
            if cagr_cost
            else " without giving up CAGR in this sample."
        )
    )
    text = f"""**Question.** What does EMA50/200 trend following look like as a diversified multi-asset portfolio, and what does it add to a 60/40 investor?

**Setup.** One long/cash sleeve per CORE ETF ({", ".join(CORE)}), inverse {VOL_WINDOW}-day-vol weights reset monthly, next-close fills, 5 bp costs, cash in T-bills; benchmarks are monthly-rebalanced 60/40 SPY/IEF, equal-weight CORE and SPY, plus a control that holds the same inverse-vol book always long. Common window {start.date()} to {ctx.end}; EEM, GLD, VNQ and DBC join the static benchmarks once they trade and the trend, TSMOM and control books once their EMA50/200 signal is warm, so the three inverse-vol books always hold the same set of markets.

**Findings.**

{chr(10).join(findings)}

**Interpretation.** {divers} {corr_line} {y22} {covid_line} {blend_line}"""

    after = (
        "All benchmarks and the trend book are evaluated on the same dates. The 2x-capped "
        "vol-targeted variant borrows at the T-bill rate, which flatters it. Calendar-year "
        "cells in the stress table show return (max drawdown within the year). Single-sleeve "
        "Sharpe ratios start at each sleeve's own warm-up date (the From column). The "
        "inverse-vol control and the TSMOM book add each sleeve on the bar its EMA50/200 "
        f"signal warms up ({ema.warmup} bars after listing), although always-long and TSMOM "
        "signals would be available earlier. Stress-year attributions sum daily "
        "contributions and are approximate. p-values are one-sided (paired stationary "
        "bootstrap, H0: no improvement)."
    )

    out = {
        "start": start.date().isoformat(),
        "trend_cagr": float(p.loc[TREND, "cagr"]),
        "trend_vol": float(p.loc[TREND, "vol"]),
        "trend_sharpe": float(s_t),
        "trend_maxdd": float(p.loc[TREND, "max_drawdown"]),
        "tsmom_sharpe": float(p.loc[TSMOM, "sharpe"]),
        "trend_vt_sharpe": float(p.loc[TREND_VT, "sharpe"]),
        "sixty_forty_cagr": float(p.loc[SIXTY, "cagr"]),
        "sixty_forty_sharpe": float(s_6),
        "sixty_forty_maxdd": float(p.loc[SIXTY, "max_drawdown"]),
        "blend_sharpe": float(s_b),
        "blend_maxdd": float(p.loc[BLEND, "max_drawdown"]),
        "blend_vs_6040_p": float(bt_blend["p_value"]),
        "trend_vs_6040_sharpe_diff": float(bt_trend["diff"]),
        "trend_vs_6040_p": float(bt_trend["p_value"]),
        "corr_trend_spy_monthly": rho_spy,
        "corr_trend_6040_monthly": rho_6040,
        "corr_trend_6040_down_months": rho_down,
        "mean_single_sleeve_sharpe": mean_sleeve,
        "trend_2022": float(c22["Trend EMA"]),
        "sixty_forty_2022": float(c22["60/40"]),
        "spy_2022": float(c22["SPY"]),
        "ief_2022": float(c22["IEF"]),
        "trend_2008": float(crisis.loc["2008", "Trend EMA"]),
        "sixty_forty_2008": float(crisis.loc["2008", "60/40"]),
        "trend_2020": float(crisis.loc["2020", "Trend EMA"]),
        "sixty_forty_2020": float(crisis.loc["2020", "60/40"]),
        "worst10_6040_avg": avg_worst_6040,
        "worst10_trend_avg": avg_worst_trend,
        "worst10_blend_avg": avg_worst_blend,
        "corr_trend_6040_up_months": rho_up,
        "beta_trend_6040_down_months": beta_down,
        "beta_trend_6040_up_months": beta_up,
        "beta_trend_6040_monthly": beta_all,
        "invvol_bh_sharpe": s_iv,
        "invvol_bh_maxdd": dd_iv,
        "trend_minus_invvol_bh_sharpe": float(bt_iv["diff"]),
        "trend_vt_maxdd": dd_vt,
        "trend_vt_cagr": float(p.loc[TREND_VT, "cagr"]),
        "trend_2008_bond_contrib": bonds08,
        "trend_2022_equity_reit_contrib": risk22,
        "trend_2022_equity_reit_contrib_q1": risk22_q1,
        "trend_2022_equity_reit_contrib_after_q1": risk22_after,
        "trend_2022_equity_reit_exit": None if exit22 is None else exit22.date().isoformat(),
        "trend_2020_apr_dec": float(reb20["t"]),
        "sixty_forty_2020_apr_dec": float(reb20["s"]),
        "cash_share_avg": cash_all,
    }

    return Section(
        name=NAME,
        title=TITLE,
        text=text,
        figures=[
            ("Trend book vs 60/40 vs the 80/20 blend: growth and drawdown", fig_eq),
            ("Stress years: calendar 2008, 2020 and 2022", fig_crisis),
            ("Where the trend book's capital sits, by asset class", fig_exp),
            ("Correlation of monthly returns", fig_corr),
        ],
        tables=[
            ("Portfolio metrics (common window, Sharpe in excess of T-bills)", perf_disp),
            ("Single-asset EMA50/200 sleeves over the same window", sleeve_disp),
            ("60/40's ten worst months", worst_disp),
            ("Stress years: return (max drawdown within year)", crisis_disp),
            ("Trend book in 2022: average weight and return contribution", attr_disp),
        ],
        after=after,
        results=out,
    )

"""Experiment 1 -- the bias waterfall behind the notebook's +505%.

The original notebook (``notebooks/00_original_exploration.ipynb``) reported an
EMA50/200 crossover on NVDA from 2022-01-01 to 2025-01-01 returning +505.1%
against +346.6% for buy-and-hold. This experiment reproduces that number with
:func:`trendlab.event_engine.legacy_notebook_backtest` and then removes the
biases one at a time, cumulatively:

0. the notebook as written (EMAs seeded at the first close of the window,
   cross-only entries starting in cash, same-close fills, whole shares, $5
   tickets, idle cash earning nothing);
1. EMAs warmed up on the full prior history and the *state* form of the rule
   (long whenever EMA50 > EMA200), still same-close, frictionless, 0% cash;
2. fills at the next close;
3. 5 bp per side proportional costs;
4. idle cash earning the 3-month T-bill rate (this one helps the rule).

A 2x2 side table splits step 1 into its two ingredients (EMA seeding x entry
logic), a second side table compares execution timings, and step 5 re-runs the
honest rule on NVDA's full history and on the period after the README was
written, plus QQQ over the README window as a non-hindsight comparison.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import plotting
from ..data import per_bar_rate
from ..engine import BacktestConfig, BacktestResult, CostModel, run_backtest
from ..event_engine import legacy_notebook_backtest
from ..metrics import summary
from ..report import df_to_markdown
from ..stats import bootstrap_sharpe_difference
from ..strategies import BuyAndHold, EMACrossover
from .common import Context, Section, costs_for

NAME = "e01_bias_waterfall"
TITLE = "1. The bias waterfall: how much of +505% was real?"

TICKER = "NVDA"
WIN_START = "2022-01-01"
WIN_END = "2025-01-01"  # the notebook's end date; last bar is 2024-12-31
POST_START = "2025-01-01"
COMPARE = "QQQ"  # a tech-heavy index over the same window (less hand-picked than one stock)
BROAD = "SPY"  # the broad market over the same window
README_ALGO, README_BNH = 5.051, 3.466  # as printed in the notebook / README

STEP_LABELS = [
    "0. Original notebook",
    "1. + Warm EMAs, state rule",
    "2. + Next-close fills",
    "3. + 5 bp/side costs",
    "4. + T-bill cash",
]
SHORT_LABELS = [
    "Notebook",
    "Warm EMAs + state rule",
    "Next-close fills",
    "5 bp costs",
    "T-bill cash",
]
METRIC_KEYS = ["total_return", "cagr", "max_drawdown", "sharpe"]
METRIC_NAMES = {
    "total_return": "total",
    "cagr": "CAGR",
    "max_drawdown": "max DD",
    "sharpe": "Sharpe",
}


# --------------------------------------------------------------------------- helpers


def _stats(returns: pd.Series, rf_yield: pd.Series, exposure=None, trades=None) -> dict[str, float]:
    """Tear-sheet metrics with Sharpe always in excess of T-bills (even at 0% cash)."""
    return summary(
        returns,
        rf=per_bar_rate(returns.index, rf_yield),
        periods=252,
        exposure=exposure,
        trades=trades,
    )


def _res_stats(res: BacktestResult, rf_yield: pd.Series) -> dict[str, float]:
    return _stats(res.returns, rf_yield, exposure=res.exposure, trades=res.trades)


def _equity_returns(equity: pd.Series, capital: float) -> pd.Series:
    """Per-bar returns that compound to ``equity[-1] / capital`` (first bar vs capital)."""
    r = equity.pct_change()
    r.iloc[0] = equity.iloc[0] / capital - 1.0
    return r


def _legacy_exposure(trades: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Exposure in force during each bar (held at the previous close) from legacy fills."""
    held = pd.Series(np.nan, index=index)
    for _, tr in trades.iterrows():
        if tr["side"] == "buy" and tr["shares"] > 0:
            held.loc[tr["date"]] = 1.0
        elif tr["side"] == "sell":
            held.loc[tr["date"]] = 0.0
    return held.ffill().fillna(0.0).shift(1).fillna(0.0)


def _cross_only_state(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """The notebook's entry logic: start flat, go long on a golden cross, flat on a death cross."""
    f, s = fast.to_numpy(), slow.to_numpy()
    out = np.zeros(len(f))
    state = 0.0
    for i in range(1, len(f)):
        if f[i] > s[i] and f[i - 1] <= s[i - 1]:
            state = 1.0
        elif f[i] < s[i] and f[i - 1] >= s[i - 1]:
            state = 0.0
        out[i] = state
    return pd.Series(out, index=fast.index)


def _window_target_run(px: pd.DataFrame, target_window: pd.Series) -> BacktestResult:
    """Frictionless same-close, 0%-cash backtest of a target defined only on the window.

    A flat decision is placed on the bar before the window so the evaluation opens on
    the window's first bar, exactly like the notebook and :meth:`Context.pair`.
    """
    full = pd.Series(np.nan, index=px.index)
    full.loc[target_window.index] = target_window.to_numpy()
    before = px.index.get_loc(target_window.index[0]) - 1
    full.iloc[before] = 0.0
    cfg = BacktestConfig(
        execution="same_close", costs=CostModel.zero(), earn_cash_rate=False, end=WIN_END
    )
    return run_backtest(px, EMACrossover(), cfg, None, target=full)


def _fills(exposure: pd.Series) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    """Fill dates of entries and exits: the close *before* the first bar held/not held."""
    w = (exposure > 0).astype(int).to_numpy()
    d = np.diff(w)
    idx = exposure.index
    return list(idx[np.flatnonzero(d > 0)]), list(idx[np.flatnonzero(d < 0)])


def _exit_label(exposure: pd.Series, execution: str) -> str:
    """Date and time of the first exit fill (next-open fills happen at the following open)."""
    w = (exposure > 0).astype(int).to_numpy()
    k = np.flatnonzero(np.diff(w) < 0)
    if not len(k):
        return "–"
    if execution == "next_open":
        return f"{exposure.index[k[0] + 1].date().isoformat()} open"
    return f"{exposure.index[k[0]].date().isoformat()} close"


def _save(ctx: Context, name: str, raw: pd.DataFrame, display: pd.DataFrame) -> pd.DataFrame:
    """Raw numbers to CSV via ``ctx.save_table``; the Markdown copy gets the display version."""
    ctx.save_table(name, raw)
    (ctx.table_dir / f"{name}.md").write_text(df_to_markdown(display) + "\n")
    return display


def _fmt_rows(raw: pd.DataFrame, pct_cols, ratio_cols=(), pp_cols=(), int_cols=()) -> pd.DataFrame:
    out = raw.copy().astype(object)
    for c in raw.columns:
        for i in raw.index:
            v = raw.loc[i, c]
            if not isinstance(v, (int, float, np.floating, np.integer)) or isinstance(v, bool):
                continue
            if pd.isna(v):
                out.loc[i, c] = "–"
            elif c in pct_cols:
                out.loc[i, c] = f"{v:.1%}"
            elif c in pp_cols:
                out.loc[i, c] = f"{v * 100:+.1f} pp"
            elif c in ratio_cols:
                out.loc[i, c] = f"{v:.2f}"
            elif c in int_cols:
                out.loc[i, c] = f"{int(v)}"
    return out


def _sign_word(x: float, pos: str, neg: str) -> str:
    return pos if x >= 0 else neg


# --------------------------------------------------------------------------- figure


def _ema_2022_chart(
    px: pd.DataFrame,
    seeded: pd.DataFrame,
    warm: pd.DataFrame,
    legacy_trades: pd.DataFrame,
    warm_exposure: pd.Series,
    path,
) -> None:
    """NVDA 2021-Q4..2023-Q1: price, warm vs window-seeded EMAs, notebook and honest fills."""
    lo, hi = pd.Timestamp("2021-10-01"), pd.Timestamp("2023-03-31")
    close = px["Close"].loc[lo:hi]
    w = warm.loc[lo:hi]
    s = seeded.loc[:hi]
    start = seeded.index[0]
    with plotting.theme() as t:
        cats = t["categorical"]
        fig, ax = plt.subplots(figsize=(10, 5.6))
        ax.axvspan(lo, start, color=t["mid"], lw=0, zorder=0)
        ax.annotate(
            "warm-up history\n(unseen by notebook)",
            xy=(lo + (start - lo) / 2, 0.97),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=8.5,
            color=t["muted"],
        )
        ax.plot(close.index, close.to_numpy(), color=t["ink2"], lw=1.2, label="NVDA close")
        ax.plot(w.index, w["fast"], color=cats[1], lw=1.8, label="EMA50 (warmed up)")
        ax.plot(w.index, w["slow"], color=cats[2], lw=1.8, label="EMA200 (warmed up)")
        ax.plot(
            s.index, s["ema_fast"], color=cats[1], lw=1.4, ls=(0, (4, 2)), label="EMA50 (seeded)"
        )
        ax.plot(
            s.index, s["ema_slow"], color=cats[2], lw=1.4, ls=(0, (4, 2)), label="EMA200 (seeded)"
        )
        ring = {"mec": t["surface"], "mew": 1.5, "ms": 9, "ls": "none", "zorder": 6}
        lt = legacy_trades[pd.to_datetime(legacy_trades["date"]) <= hi]
        nb_buy = pd.to_datetime(lt.loc[lt["side"] == "buy", "date"])
        nb_sell = pd.to_datetime(lt.loc[lt["side"] == "sell", "date"])
        ax.plot(nb_buy, close.reindex(nb_buy), "^", color=cats[0], label="Notebook buy", **ring)
        ax.plot(nb_sell, close.reindex(nb_sell), "v", color=t["ink"], label="Notebook sell", **ring)
        ins, outs = _fills(warm_exposure.loc[:hi])
        hollow = {"mfc": "none", "mew": 2.0, "ms": 11, "ls": "none", "zorder": 7}
        ax.plot(ins, close.reindex(ins), "o", mec=cats[0], label="Warm rule entry", **hollow)
        ax.plot(outs, close.reindex(outs), "o", mec=t["ink"], label="Warm rule exit", **hollow)
        ax.axvline(start, color=t["axis"], lw=0.9, zorder=1)
        note = {"fontsize": 8.5, "color": t["ink2"], "va": "top"}
        top = close.max()

        def callout(day, text, y, left=True):
            dx = pd.Timedelta(days=4 if left else -4)
            ax.annotate(
                text,
                xy=(day, close.loc[day]),
                xytext=(day + dx, top * y),
                ha="left" if left else "right",
                arrowprops={
                    "arrowstyle": "-",
                    "color": t["muted"],
                    "lw": 0.8,
                    "shrinkA": 2,
                    "shrinkB": 8,
                    "relpos": (0.0 if left else 1.0, 1.0),
                },
                **note,
            )

        if len(nb_sell):
            d0 = nb_sell.min()
            callout(
                d0,
                f"{d0:%d %b}: seeded EMA50 < EMA200, the notebook\n'sells' while flat; with "
                "warmed EMAs the\nhonest rule is long from the first close",
                0.2,
            )
        if len(outs):
            d1 = outs[0]
            chg = plotting._pct(close.loc[d1] / close.loc[start] - 1)
            callout(d1, f"{d1:%d %b}: warm rule exits,\nNVDA {chg} since start", 0.38)
        if len(nb_buy):
            d2 = nb_buy.min()
            callout(d2, f"{d2:%d %b %Y}:\nfirst notebook buy", 0.2, left=False)
        ax.set_ylabel("Price (split-adjusted $)")
        ax.set_ylim(0, close.max() * 1.12)
        plotting._date_axis(ax)
        ax.legend(loc="upper center", ncols=3, fontsize=8.5, bbox_to_anchor=(0.6, 1.0))
        first_cross = nb_sell.min() if len(nb_sell) else start
        day = seeded.index.get_loc(first_cross) + 1
        tail = f" and stayed in cash until {nb_buy.min():%b %Y}" if len(nb_buy) else ""
        plotting._titles(
            ax,
            f"Seeded at the first close, the notebook's EMAs 'death-crossed' on day {day}{tail}",
            f"NVDA. Notebook EMAs start at the {start:%d %b %Y} close (dashed; first sell signal "
            f"{first_cross:%d %b %Y}).\nWarmed-up EMAs (solid) say NVDA was in an uptrend, so an "
            "honest rule starts invested and rides the first leg of the 2022 crash.",
        )
        plotting.save_figure(fig, path)


# --------------------------------------------------------------------------- experiment


def run(ctx: Context) -> Section:
    rf = ctx.rf
    px = ctx.prices(TICKER)
    window = px.loc[WIN_START:WIN_END]
    strat = EMACrossover(50, 200)
    costs = costs_for(TICKER)
    cost_bps = costs.bps

    # ---- step 0: the notebook, exactly
    lg = legacy_notebook_backtest(window)
    r0a = _equity_returns(lg["algo_equity"], 100_000.0)
    r0b = _equity_returns(lg["bnh_equity"], 100_000.0)
    exp0 = _legacy_exposure(lg["trades"], window.index)
    n_buys0 = int((lg["trades"]["side"] == "buy").sum())
    s0a = _stats(r0a, rf, exposure=exp0)
    s0a["n_trades"] = n_buys0
    s0b = _stats(r0b, rf, exposure=pd.Series(1.0, index=window.index))

    # ---- steps 1-4: cumulative corrections (vectorised engine, same window)
    base = {"start": WIN_START, "end": WIN_END}
    step_cfg = [
        {"execution": "same_close", "costs": CostModel.zero(), "earn_cash_rate": False},
        {"execution": "next_close", "costs": CostModel.zero(), "earn_cash_rate": False},
        {"execution": "next_close", "costs": costs, "earn_cash_rate": False},
        {"execution": "next_close", "costs": costs, "earn_cash_rate": True},
    ]
    pairs = [ctx.pair(TICKER, strat, **base, **kw) for kw in step_cfg]
    stats_a = [s0a] + [_res_stats(a, rf) for a, _ in pairs]
    stats_b = [s0b] + [_res_stats(b, rf) for _, b in pairs]
    honest, honest_bnh = pairs[-1]

    rows = {}
    for lab, sa, sb in zip(STEP_LABELS, stats_a, stats_b, strict=True):
        row = {f"Strategy {METRIC_NAMES[k]}": sa[k] for k in METRIC_KEYS}
        row.update({f"B&H {METRIC_NAMES[k]}": sb[k] for k in METRIC_KEYS})
        row["Excess total"] = sa["total_return"] - sb["total_return"]
        row["Time in market"] = sa["exposure"]
        row["Round trips"] = sa.get("n_trades", np.nan)
        rows[lab] = row
    wf_raw = pd.DataFrame(rows).T
    wf_raw.index.name = "Step"
    pct_cols = {c for c in wf_raw.columns if any(m in c for m in ("total", "CAGR", "max DD"))}
    pct_cols |= {"Time in market"}
    pct_cols -= {"Excess total"}
    wf_disp = _fmt_rows(
        wf_raw,
        pct_cols,
        ratio_cols={"Strategy Sharpe", "B&H Sharpe"},
        pp_cols={"Excess total"},
        int_cols={"Round trips"},
    )
    wf_disp = _save(ctx, f"{NAME}_waterfall", wf_raw, wf_disp)

    # ---- step 1 decomposed: EMA seeding x entry logic (frictionless, same close, 0% cash)
    seeded = lg["signals"]
    warm = strat.indicators(px)
    warm_w = warm.loc[window.index]
    variants = {
        ("Seeded on window", "Cross-only from cash"): _cross_only_state(
            seeded["ema_fast"], seeded["ema_slow"]
        ),
        ("Seeded on window", "State (EMA50 > EMA200)"): (
            seeded["ema_fast"] > seeded["ema_slow"]
        ).astype(float),
        ("Warmed on full history", "Cross-only from cash"): _cross_only_state(
            warm_w["fast"], warm_w["slow"]
        ),
        ("Warmed on full history", "State (EMA50 > EMA200)"): (
            warm_w["fast"] > warm_w["slow"]
        ).astype(float),
    }
    init_rows, init_res = {}, {}
    for (seed, logic), tgt in variants.items():
        res = _window_target_run(px, tgt)
        st = _res_stats(res, rf)
        ins, _ = _fills(res.exposure)
        init_res[(seed, logic)] = st
        init_rows[f"{seed} / {logic}"] = {
            "Total return": st["total_return"],
            "Max DD": st["max_drawdown"],
            "Time in market": st["exposure"],
            "First entry": ins[0].date().isoformat() if ins else "never",
        }
    init_raw = pd.DataFrame(init_rows).T
    init_raw.index.name = "EMA seeding / entry logic"
    init_disp = _fmt_rows(init_raw, {"Total return", "Max DD", "Time in market"})
    init_disp = _save(ctx, f"{NAME}_initialisation", init_raw, init_disp)
    sc_cross = init_res[("Seeded on window", "Cross-only from cash")]["total_return"]
    sc_state = init_res[("Seeded on window", "State (EMA50 > EMA200)")]["total_return"]
    wm_cross = init_res[("Warmed on full history", "Cross-only from cash")]["total_return"]
    wm_state = init_res[("Warmed on full history", "State (EMA50 > EMA200)")]["total_return"]

    # ---- execution-timing side table (frictionless and honest settings)
    ex_rows = {}
    ex_tot = {}
    for ex in ("same_close", "next_open", "next_close"):
        fa, _ = ctx.pair(
            TICKER, strat, **base, execution=ex, costs=CostModel.zero(), earn_cash_rate=False
        )
        ha, hb = ctx.pair(TICKER, strat, **base, execution=ex)
        sh, sb = _res_stats(ha, rf), _res_stats(hb, rf)
        ex_tot[ex] = sh["total_return"]
        ex_rows[ex.replace("_", " ")] = {
            "Total (frictionless)": _res_stats(fa, rf)["total_return"],
            "Total (5 bp + T-bills)": sh["total_return"],
            "CAGR": sh["cagr"],
            "Max DD": sh["max_drawdown"],
            "Sharpe": sh["sharpe"],
            "B&H total": sb["total_return"],
            "First exit fill": _exit_label(ha.exposure, ex),
        }
    ex_raw = pd.DataFrame(ex_rows).T
    ex_raw.index.name = "Execution"
    ex_disp = _fmt_rows(
        ex_raw,
        {"Total (frictionless)", "Total (5 bp + T-bills)", "CAGR", "Max DD", "B&H total"},
        ratio_cols={"Sharpe"},
    )
    ex_disp = _save(ctx, f"{NAME}_execution", ex_raw, ex_disp)

    # ---- step 5: out of period / out of sample (honest settings)
    full_a, full_b = ctx.pair(TICKER, strat)
    post_a, post_b = ctx.pair(TICKER, strat, start=POST_START)
    cmp_a, cmp_b = ctx.pair(COMPARE, strat, **base)
    spy_a, spy_b = ctx.pair(BROAD, strat, **base)
    spy_full = ctx.backtest("SPY", BuyAndHold(), start=full_a.start)
    oop = {
        f"{TICKER} README window": (honest, honest_bnh),
        f"{TICKER} full history": (full_a, full_b),
        f"{TICKER} after README": (post_a, post_b),
        f"{COMPARE} README window": (cmp_a, cmp_b),
        f"{BROAD} README window": (spy_a, spy_b),
    }
    oop_rows, oop_stats = {}, {}
    for lab, (a, b) in oop.items():
        sa, sb = _res_stats(a, rf), _res_stats(b, rf)
        oop_stats[lab] = (sa, sb)
        row = {"Start": sa["start"], "End": sa["end"], "Years": sa["years"]}
        row.update({f"Strategy {METRIC_NAMES[k]}": sa[k] for k in METRIC_KEYS})
        row.update({f"B&H {METRIC_NAMES[k]}": sb[k] for k in METRIC_KEYS})
        row["Round trips"] = sa["n_trades"]
        oop_rows[lab] = row
    oop_raw = pd.DataFrame(oop_rows).T
    oop_raw.index.name = "Sample"
    oop_pct = {c for c in oop_raw.columns if any(m in c for m in ("total", "CAGR", "max DD"))}
    oop_disp = _fmt_rows(
        oop_raw,
        oop_pct,
        ratio_cols={"Strategy Sharpe", "B&H Sharpe", "Years"},
        int_cols={"Round trips"},
    )
    oop_disp = _save(ctx, f"{NAME}_out_of_period", oop_raw, oop_disp)
    fa_s, fb_s = oop_stats[f"{TICKER} full history"]
    pa_s, pb_s = oop_stats[f"{TICKER} after README"]
    ca_s, cb_s = oop_stats[f"{COMPARE} README window"]
    sa_s, sb_s = oop_stats[f"{BROAD} README window"]
    spy_s = _res_stats(spy_full, rf)
    post_in, post_out = _fills(post_a.exposure)

    # ---- is the Sharpe difference distinguishable from zero? (paired stationary bootstrap)
    n_boot = 500 if ctx.quick else 5000

    def sr_diff(a: BacktestResult, b: BacktestResult) -> dict:
        out = bootstrap_sharpe_difference(
            a.returns, b.returns, rf=per_bar_rate(a.returns.index, rf), n_boot=n_boot, seed=0
        )
        out.pop("samples")
        return out

    bs_win = sr_diff(honest, honest_bnh)
    bs_full = sr_diff(full_a, full_b)

    # ---- figures
    tot_a = [s["total_return"] for s in stats_a]
    tot_b = [s["total_return"] for s in stats_b]
    f_wf = ctx.fig_path(f"{NAME}_waterfall")
    gap4 = tot_a[-1] - tot_b[-1]
    with plotting.theme():
        fig = plotting.bias_waterfall(
            list(zip(SHORT_LABELS, tot_a, strict=True)),
            title=(
                f"Bias-free, NVDA's EMA50/200 drops from {tot_a[0]:.0%} to {tot_a[-1]:.0%}, "
                f"{_sign_word(gap4, 'above', 'below')} buy-and-hold's {tot_b[-1]:.0%}"
            ),
            subtitle=(
                f"Total return {window.index[0]:%b %Y}–{window.index[-1]:%b %Y}, each step cumulative. "
                f"Grey line: buy-and-hold under the step-4 settings ({tot_b[-1]:.0%})."
            ),
            benchmark=tot_b[-1],
            benchmark_label="Buy & hold",
            final_label=None,
            ylabel="Total return",
        )
        # widen to the right so the buy-and-hold label clears the last bar's value label
        ax = fig.axes[0]
        ax.set_xlim(-0.7, len(SHORT_LABELS) + 0.55)
        plotting.save_figure(fig, f_wf)

    f_ema = ctx.fig_path(f"{NAME}_ema_seeding")
    _ema_2022_chart(px, seeded, warm, lg["trades"], pairs[0][0].exposure, f_ema)

    f_eq = ctx.fig_path(f"{NAME}_honest_equity")
    plotting.equity_and_drawdown(
        {f"EMA50/200, step 4 ({tot_a[-1]:.0%})": honest.returns, "Buy & Hold": honest_bnh.returns},
        log=True,
        title=(
            f"After the corrections the rule {_sign_word(gap4, 'beats', 'trails')} buy-and-hold "
            f"by {abs(gap4) * 100:.0f} pp but cuts the drawdown "
            f"from {abs(stats_b[-1]['max_drawdown']):.0%} to "
            f"{abs(stats_a[-1]['max_drawdown']):.0%}"
            if stats_a[-1]["max_drawdown"] > stats_b[-1]["max_drawdown"]
            else f"After the corrections the rule {_sign_word(gap4, 'beats', 'trails')} "
            f"buy-and-hold by {abs(gap4) * 100:.0f} pp"
        ),
        subtitle=(
            f"NVDA {honest.start:%d %b %Y}–{honest.end:%d %b %Y}: warmed EMAs, next-close fills, "
            f"{cost_bps:.0f} bp/side, idle cash in T-bills."
        ),
        path=f_eq,
    )

    f_full = ctx.fig_path(f"{NAME}_full_history")
    plotting.equity_and_drawdown(
        {"EMA50/200": full_a.returns, "Buy & Hold": full_b.returns},
        log=True,
        title=(
            f"Over NVDA's full history the rule {_sign_word(fa_s['cagr'] - fb_s['cagr'], 'beats', 'trails')} "
            f"buy-and-hold on CAGR ({fa_s['cagr']:.1%} vs {fb_s['cagr']:.1%})"
        ),
        subtitle=(
            f"NVDA {full_a.start:%b %Y}–{full_a.end:%b %Y}, step-4 settings. Max drawdown "
            f"{plotting._pct(fa_s['max_drawdown'])} vs {plotting._pct(fb_s['max_drawdown'])}; "
            f"{int(fa_s['n_trades'])} round trips."
        ),
        path=f_full,
    )

    # ---- derived numbers for the text
    d = np.diff(tot_a)
    start_spread = float(warm_w["fast"].iloc[0] / warm_w["slow"].iloc[0] - 1.0)
    _, w_outs = _fills(pairs[0][0].exposure)
    first_exit = w_outs[0] if w_outs else None
    close = window["Close"]
    loss_to_exit = float(close.loc[first_exit] / close.iloc[0] - 1.0) if first_exit else np.nan
    nb_sells = pd.to_datetime(lg["trades"].loc[lg["trades"]["side"] == "sell", "date"])
    nb_buys = pd.to_datetime(lg["trades"].loc[lg["trades"]["side"] == "buy", "date"])
    nb_first_sell = nb_sells.min() if len(nb_sells) else None
    nb_first_buy = nb_buys.min() if len(nb_buys) else None
    timing_spread = max(ex_tot.values()) - min(ex_tot.values())
    idle_mask = (honest.exposure == 0).to_numpy().copy()
    idle_mask[0] = False  # opening bar
    idle_r = per_bar_rate(honest.returns.index, rf).to_numpy()[idle_mask]
    idle_rf = float(np.prod(1.0 + idle_r) ** (252 / max(len(idle_r), 1)) - 1.0)
    idle = 1.0 - stats_a[-1]["exposure"]
    reproduces = abs(tot_a[0] - README_ALGO) < 0.01 and abs(tot_b[0] - README_BNH) < 0.01
    full_gap = fa_s["cagr"] - fb_s["cagr"]
    post_gap = pa_s["total_return"] - pb_s["total_return"]
    cmp_gap = ca_s["total_return"] - cb_s["total_return"]

    spy_gap = sa_s["total_return"] - sb_s["total_return"]
    fee_effect = tot_a[0] - sc_cross  # whole shares + $5 tickets vs the frictionless replica

    def pp(x: float) -> str:
        return f"{x * 100:+.1f} pp"

    def ci(bs: dict) -> str:
        return f"[{bs['ci_low']:+.2f}, {bs['ci_high']:+.2f}]"

    exit_txt = (
        f"and only exits on {first_exit:%d %b %Y}, after NVDA had fallen {abs(loss_to_exit):.0%}"
        if first_exit is not None and loss_to_exit < 0
        else (
            f"and exits on {first_exit:%d %b %Y}" if first_exit is not None else "and never exits"
        )
    )
    seeded_txt = (
        f"'death-crossed' on {nb_first_sell:%d %b %Y} (a sell while already flat) and first "
        f"bought on {nb_first_buy:%d %b %Y}"
        if nb_first_sell is not None and nb_first_buy is not None
        else "kept the notebook out of the market"
    )
    post_reentry = next((x for x in post_in if post_out and x > post_out[0]), None)
    if post_out and post_reentry is not None:
        pc = px["Close"]
        rebuy = float(pc.loc[post_reentry] / pc.loc[post_out[0]] - 1.0)
        post_txt = (
            f"sold on {post_out[0]:%d %b %Y} and bought back on {post_reentry:%d %b %Y} "
            f"{abs(rebuy):.0%} {'higher' if rebuy >= 0 else 'lower'}"
        )
    else:
        rebuy = np.nan
        post_txt = f"{int(pa_s['n_trades'])} round trips"
    same_dd = abs(pa_s["max_drawdown"] - pb_s["max_drawdown"]) < 0.005
    post_dd_txt = (
        f"the same {pb_s['max_drawdown']:.0%} drawdown as buy-and-hold"
        if same_dd
        else f"max drawdown {pa_s['max_drawdown']:.0%} vs {pb_s['max_drawdown']:.0%}"
    )
    both_ns = all(b["ci_low"] <= 0 <= b["ci_high"] for b in (bs_win, bs_full))

    text = f"""**Question.** How much of the notebook's +{README_ALGO:.1%} for an EMA50/200 crossover on NVDA (vs +{README_BNH:.1%} buy-and-hold, 2022-2024) survives once each backtesting bias is removed?

**Setup.** Step 0 replays the notebook (`legacy_notebook_backtest`); steps 1-4 remove one bias at a time, cumulatively, on the same {window.index[0]:%d %b %Y} to {window.index[-1]:%d %b %Y} window, with buy-and-hold under identical settings and Sharpe always in excess of T-bills. Step 5 re-runs the step-4 rule on NVDA's full history, after the README, and on {COMPARE} and {BROAD}.

**Findings.**

- The replica {"reproduces" if reproduces else "does not exactly reproduce"} the README: {tot_a[0]:.1%} vs {tot_b[0]:.1%}, from {n_buys0} purchase{"" if n_buys0 == 1 else "s"} in three years.
- Warm EMAs plus the *state* rule (long while EMA50 > EMA200) is the decisive correction: {tot_a[1]:.1%} ({pp(d[0])}). On {window.index[0]:%d %b %Y} the warmed EMA50 was {start_spread:.0%} above EMA200, so the rule starts invested {exit_txt}. The notebook's EMAs, seeded at the first close, {seeded_txt}. Seeded EMAs alone ({sc_state:.0%}) or cross-only entries alone ({wm_cross:.0%}) each keep the rule out of the 2022 crash; only both corrections together give {wm_state:.0%}.
- Next-close fills {"add" if d[1] >= 0 else "cost"} {abs(d[1]) * 100:.1f} pp. Under step-4 settings same-close, next-open and next-close give {ex_tot["same_close"]:.0%}, {ex_tot["next_open"]:.0%} and {ex_tot["next_close"]:.0%}: with {int(stats_a[-1]["n_trades"])} round trips, a handful of fill prices move the result by {timing_spread * 100:.0f} pp.
- {cost_bps:.0f} bp/side costs take {abs(d[2]) * 100:.1f} pp; T-bill interest on idle cash *helps* by {pp(d[3])} (in cash {idle:.0%} of the time at {idle_rf:.1%} a year).
- Honest result: {tot_a[-1]:.1%} vs {tot_b[-1]:.1%} ({pp(gap4)}), max drawdown {stats_a[-1]["max_drawdown"]:.1%} vs {stats_b[-1]["max_drawdown"]:.1%}, Sharpe {stats_a[-1]["sharpe"]:.2f} vs {stats_b[-1]["sharpe"]:.2f} (bootstrap 95% CI of the difference {ci(bs_win)}).
- Full history ({fa_s["start"][:4]}-{fa_s["end"][:4]}, {int(fa_s["n_trades"])} round trips): the rule {_sign_word(full_gap, "beats", "trails")} on CAGR, {fa_s["cagr"]:.1%} vs {fb_s["cagr"]:.1%}, max drawdown {fa_s["max_drawdown"]:.0%} vs {fb_s["max_drawdown"]:.0%}, Sharpe {fa_s["sharpe"]:.2f} vs {fb_s["sharpe"]:.2f} (CI {ci(bs_full)}). After the README: {pa_s["total_return"]:.1%} vs {pb_s["total_return"]:.1%}; the rule {post_txt}, with {post_dd_txt}."""

    prior = px["Close"].loc[: window.index[0]].iloc[-252:]
    peak_day = prior.idxmax()
    peak_gap = float(window["Close"].iloc[0] / prior.max() - 1.0)
    shock_2025 = bool(post_out) and pd.Timestamp("2025-03-01") <= post_out[0] <= pd.Timestamp(
        "2025-05-31"
    )

    def vs(gap: float) -> str:
        return "roughly matched" if abs(gap) < 0.02 else _sign_word(gap, "beat", "trailed")

    shallower = (
        ca_s["max_drawdown"] > cb_s["max_drawdown"] and sa_s["max_drawdown"] > sb_s["max_drawdown"]
    )
    regime_txt = (
        f"over 2022-2024 it {vs(cmp_gap)} buy-and-hold on {COMPARE} ({pp(cmp_gap)}) and "
        f"{vs(spy_gap)} it on {BROAD} ({pp(spy_gap)})"
        + (", with shallower drawdowns on both" if shallower else "")
        + f", while on NVDA after the README it {vs(post_gap)} buy-and-hold by "
        f"{abs(post_gap) * 100:.0f} pp"
        + (
            f", selling into the short April 2025 sell-off and buying back {rebuy:.0%} higher"
            if shock_2025 and post_gap < 0 and rebuy > 0
            else ""
        )
    )
    if both_ns:
        sig_txt = (
            "Neither Sharpe difference is distinguishable from zero (both bootstrap CIs contain "
            "it): "
        )
    else:
        sig_txt = "Even where a Sharpe difference clears the bootstrap test, "
    interp = (
        f"**Interpretation.** The +{README_ALGO:.0%} was an initialisation artefact, not an edge. "
        f"The backtest opens in cash {(window.index[0] - peak_day).days} days after NVDA's "
        f"{peak_day:%b %Y} high ({abs(peak_gap):.0%} below it), and seeded EMAs and cross-only "
        "entries each keep it there through the 2022 bear market. Corrected, the "
        f"{pp(tot_a[0] - tot_b[0])} lead over buy-and-hold "
        + (
            f"becomes a {abs(gap4) * 100:.1f} pp shortfall. "
            if gap4 < 0
            else f"shrinks to {pp(gap4)}. "
        )
        + (
            "What is left is the drawdown control Faber (2007) documents for moving-average timing"
            if stats_a[-1]["max_drawdown"] > stats_b[-1]["max_drawdown"]
            else "Not even the drawdown control Faber (2007) documents for moving-average timing "
            "survives"
        )
        + (", here bought with lower return. " if gap4 < 0 else ". ")
        + "Zakamulin (2014) finds "
        "that such rules' gains are concentrated in a few long bear markets and depend heavily "
        f"on the sample, and that is the pattern here: {regime_txt}. "
        + sig_txt
        + "the evidence for a trend premium (Moskowitz, Ooi & Pedersen 2012; Hurst, Ooi & "
        "Pedersen 2017) comes from diversified portfolios of dozens of futures markets, which "
        "one stock and a few round trips cannot test. Finally, NVDA was picked with hindsight "
        f"(buy-and-hold CAGR {fb_s['cagr']:.0%} since {full_a.start:%Y} vs {spy_s['cagr']:.0%} "
        f"for {BROAD}): that choice is an unrecorded trial which none of the statistics above "
        "account for (Sullivan, Timmermann & White 1999; Bailey & Lopez de Prado 2014)."
    )
    text = text + "\n\n" + interp

    results = {
        "original_algo_return": tot_a[0],
        "original_bnh_return": tot_b[0],
        "original_algo_mdd": stats_a[0]["max_drawdown"],
        "original_bnh_mdd": stats_b[0]["max_drawdown"],
        "original_algo_sharpe": stats_a[0]["sharpe"],
        "original_bnh_sharpe": stats_b[0]["sharpe"],
        "reproduces_readme": bool(reproduces),
        "step1_warm_state_return": tot_a[1],
        "step2_next_close_return": tot_a[2],
        "step3_costs_return": tot_a[3],
        "step4_tbill_return": tot_a[4],
        "init_seeded_cross_return": sc_cross,
        "init_seeded_state_return": sc_state,
        "init_warm_cross_return": wm_cross,
        "init_warm_state_return": wm_state,
        "warm_ema_spread_at_start": start_spread,
        "warm_first_exit": first_exit.date().isoformat() if first_exit is not None else "",
        "nvda_return_to_first_exit": loss_to_exit,
        "exec_same_close_return": ex_tot["same_close"],
        "exec_next_open_return": ex_tot["next_open"],
        "exec_next_close_return": ex_tot["next_close"],
        "exec_timing_spread": timing_spread,
        "corrected_algo_return": tot_a[-1],
        "corrected_bnh_return": tot_b[-1],
        "corrected_algo_cagr": stats_a[-1]["cagr"],
        "corrected_bnh_cagr": stats_b[-1]["cagr"],
        "corrected_algo_mdd": stats_a[-1]["max_drawdown"],
        "corrected_bnh_mdd": stats_b[-1]["max_drawdown"],
        "corrected_algo_sharpe": stats_a[-1]["sharpe"],
        "corrected_bnh_sharpe": stats_b[-1]["sharpe"],
        "corrected_round_trips": int(stats_a[-1]["n_trades"]),
        "full_history_start": fa_s["start"],
        "full_history_end": fa_s["end"],
        "full_history_algo_cagr": fa_s["cagr"],
        "full_history_bnh_cagr": fb_s["cagr"],
        "full_history_algo_sharpe": fa_s["sharpe"],
        "full_history_bnh_sharpe": fb_s["sharpe"],
        "full_history_algo_mdd": fa_s["max_drawdown"],
        "full_history_bnh_mdd": fb_s["max_drawdown"],
        "full_history_round_trips": int(fa_s["n_trades"]),
        "full_history_spy_bnh_cagr": spy_s["cagr"],
        "post_readme_start": pa_s["start"],
        "post_readme_algo_return": pa_s["total_return"],
        "post_readme_bnh_return": pb_s["total_return"],
        "post_readme_algo_mdd": pa_s["max_drawdown"],
        "post_readme_bnh_mdd": pb_s["max_drawdown"],
        "qqq_window_algo_return": ca_s["total_return"],
        "qqq_window_bnh_return": cb_s["total_return"],
        "spy_window_algo_return": sa_s["total_return"],
        "spy_window_bnh_return": sb_s["total_return"],
        "corrected_sharpe_diff": bs_win["diff"],
        "corrected_sharpe_diff_ci_low": bs_win["ci_low"],
        "corrected_sharpe_diff_ci_high": bs_win["ci_high"],
        "corrected_sharpe_diff_p": bs_win["p_value"],
        "full_history_sharpe_diff": bs_full["diff"],
        "full_history_sharpe_diff_ci_low": bs_full["ci_low"],
        "full_history_sharpe_diff_ci_high": bs_full["ci_high"],
        "full_history_sharpe_diff_p": bs_full["p_value"],
        "legacy_fee_integer_effect": fee_effect,
    }
    results = {
        k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in results.items()
    }

    return Section(
        name=NAME,
        title=TITLE,
        text=text,
        figures=[
            ("Total return of the EMA50/200 rule as each bias is removed (cumulative).", f_wf),
            (
                "NVDA around the 2022 bear market: EMAs seeded at the window's first close "
                "(dashed, the notebook) vs warmed up on the prior history (solid).",
                f_ema,
            ),
            ("Step 4 (honest) strategy vs buy-and-hold on the README window.", f_eq),
            ("The same honest rule over NVDA's full history.", f_full),
        ],
        tables=[
            ("Bias waterfall, both legs (Sharpe in excess of T-bills at every step).", wf_disp),
            (
                "Step 1 decomposed: EMA seeding x entry logic (same-close, frictionless, 0% cash).",
                init_disp,
            ),
            ("Execution timing (step-4 settings unless stated).", ex_disp),
            ("Out-of-period and out-of-sample checks (step-4 settings).", oop_disp),
        ],
        after=(
            "Caveat: Yahoo adjusted prices are not point-in-time (they are rewritten after every "
            "split and dividend), so the cached series differs slightly from the one the notebook "
            f"downloaded; the replica is within {abs(tot_a[0] - README_ALGO) * 100:.2f} pp "
            f"(strategy) and {abs(tot_b[0] - README_BNH) * 100:.2f} pp (buy-and-hold) of the "
            "README figures, which are quoted to one decimal."
        ),
        results=results,
    )

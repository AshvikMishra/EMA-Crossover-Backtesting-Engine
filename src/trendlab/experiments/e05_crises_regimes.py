"""Experiment 5: crisis alpha or crisis lag?

Trend following is marketed as "crisis alpha" (Hurst, Ooi & Pedersen 2017) with
a long-straddle-like payoff (Fung & Hsieh 2001). This experiment checks the
claim for the EMA50/200 rule on SPY, QQQ and Bitcoin:

* named stress windows (:data:`trendlab.universe.CRISES` plus two crypto
  crashes defined here) via :func:`trendlab.regimes.crisis_table`, extended with
  a *peak-to-recovery* view -- the in-window number flatters a rule that exits
  at the trough, because the cost of re-entering late is paid after the window;
* performance by trend, VIX and drawdown regime;
* up/down capture and the quarterly convexity "smile";
* the rolling 3-year Sharpe difference against buy-and-hold.
"""

from __future__ import annotations

import math
import re

import matplotlib.ticker
import numpy as np
import pandas as pd

from .. import plotting, regimes
from ..engine import BacktestResult
from ..strategies import EMACrossover
from ..universe import CRISES
from .common import Context, Section

NAME = "e05_crises_regimes"
TITLE = "5. Crisis alpha or crisis lag? Stress periods, regimes and convexity"

# Crypto-specific peak-to-trough windows of BTC-USD (daily closes).
CRYPTO_CRISES: dict[str, tuple[str, str]] = {
    "2018 crypto winter": ("2017-12-17", "2018-12-15"),
    "2021-22 crypto crash": ("2021-11-08", "2022-11-21"),
}
# For BTC the equity-market windows that are subsumed by a crypto crash (Q4 2018,
# the 2022 bear) are dropped so no drawdown is counted twice; the two macro
# shocks that hit every risk asset at once are kept.
BTC_CRISES: dict[str, tuple[str, str]] = {
    "COVID crash": CRISES["COVID crash"],
    "2025 tariff shock": CRISES["2025 tariff shock"],
    **CRYPTO_CRISES,
}
CRISIS_SETS: dict[str, dict[str, tuple[str, str]]] = {
    "SPY": CRISES,
    "QQQ": CRISES,
    "BTC-USD": BTC_CRISES,
}
ROLL_YEARS = 3
# A peak-to-trough window longer than ~6 calendar months is a "slow bear". Calendar
# days, not bars, so 7-day crypto bars and 5-day ETF bars are classified alike.
SLOW_DAYS = 182
SHORT = {
    "Global financial crisis": "GFC",
    "2011 euro / US downgrade": "2011 euro/downgrade",
    "2015-16 China / oil": "2015-16 China/oil",
}


# --------------------------------------------------------------------------- helpers


def _cycle_view(
    s: BacktestResult,
    b: BacktestResult,
    table: pd.DataFrame,
    crises: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Peak-to-recovery comparison for each crisis row of ``table``.

    ``recovery`` is the first common bar after the trough at which buy-and-hold
    equity is back to its level at the crisis start (or the last bar, flagged by
    ``recovered=False``). If the next crisis in ``crises`` starts before that,
    the cycle is cut at that crisis's start close (``capped=True``) so no bar is
    charged to two episodes (e.g. SPY's GFC recovery runs into 2011, BTC's 2018
    winter into COVID). ``cycle_strat`` is the strategy's compounded return
    from the crisis start to the cycle end -- buy-and-hold's is ~0 when
    uncapped -- so ``cycle_edge`` is the rule's edge over the drawdown-and-
    recovery cycle, including the rebound it missed by re-entering late.
    ``post_trough_exits`` counts switches to cash after the trough but before
    the cycle end (selling into the rebound).
    """
    common = s.equity.index.intersection(b.equity.index)
    es, eb = s.equity.reindex(common), b.equity.reindex(common)
    inv = s.exposure.reindex(common).fillna(0.0).to_numpy() > 1e-12
    starts = sorted(pd.Timestamp(v[0]) for v in crises.values())
    rows = {}
    for name, row in table.iterrows():
        p0 = int(common.searchsorted(row["start"], side="right")) - 1
        pt = int(common.get_loc(row["trough"]))
        later = np.flatnonzero(eb.iloc[pt:].to_numpy() >= eb.iloc[p0])
        recovered = len(later) > 0
        pr = pt + int(later[0]) if recovered else len(common) - 1
        nxt = [d for d in starts if d > row["end"]]
        capped = False
        if nxt:
            pc = int(common.searchsorted(nxt[0], side="right")) - 1
            if pt < pc < pr:
                pr, capped = pc, True
        exits = int(np.sum(inv[pt:pr] & ~inv[pt + 1 : pr + 1]))
        rows[name] = {
            "recovery": common[pr],
            "recovered": recovered and not capped,
            "capped": capped,
            "cycle_bars": pr - p0,
            "cycle_bench": float(eb.iloc[pr] / eb.iloc[p0] - 1.0),
            "cycle_strat": float(es.iloc[pr] / es.iloc[p0] - 1.0),
            "post_trough_exits": exits,
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out["cycle_edge"] = out["cycle_strat"] - out["cycle_bench"]
    return out


def _pct(v: float, d: int = 1) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{d}%}"


def _pp(v: float) -> str:
    return f"{v * 100:+.1f} pp"


def _date(v) -> str:
    return "–" if pd.isna(v) else pd.Timestamp(v).date().isoformat()


def _num(v, fmt="{:.0f}") -> str:
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else fmt.format(v)


def _crisis_display(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["B&H"] = df["bench_return"].map(_pct)
    out["EMA50/200"] = df["strat_return"].map(_pct)
    out["Protection"] = df["protection"].map(_pp)
    out["EMA max DD"] = df["strat_max_dd"].map(_pct)
    out["Exit (first bar out)"] = df["exit_date"].map(_date)
    out["Re-entry lag (bars)"] = df["reentry_lag_bars"].map(_num)
    out["Missed rebound"] = df["missed_rebound"].map(_pct)
    out["Cycle end (B&H recovery)"] = [
        _date(r) + (" (next crisis)" if cap else "" if ok else " (not yet)")
        for r, ok, cap in zip(df["recovery"], df["recovered"], df["capped"], strict=True)
    ]
    out["Exits after trough"] = df["post_trough_exits"].map(_num)
    out["Peak-to-recovery edge"] = df["cycle_edge"].map(_pp)
    out.index = [f"{a} · {c}" for a, c in df.index]
    out.index.name = "Asset · crisis"
    return out


def _shade_crises(fig, crises: dict[str, tuple[str, str]], lo, hi) -> None:
    ax = fig.axes[0]
    t = plotting.TOKENS["light"]
    for s, e in crises.values():
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        if e < lo or s > hi:
            continue
        ax.axvspan(max(s, lo), min(e, hi), color=t["grid"], alpha=0.7, lw=0, zorder=0)


# --------------------------------------------------------------------------- run


def run(ctx: Context) -> Section:
    strat = EMACrossover(50, 200)
    # a few vectorised backtests (~3 s), so quick mode runs the same analysis
    tickers = list(CRISIS_SETS)
    pairs = {tk: ctx.pair(tk, strat) for tk in tickers}

    # ---- crisis windows ----------------------------------------------------
    parts = []
    for tk, (s, b) in pairs.items():
        ct = regimes.crisis_table(s, b, CRISIS_SETS[tk])
        ct = ct.join(_cycle_view(s, b, ct, CRISIS_SETS[tk]))
        ct["days"] = (ct["end"] - ct["start"]).dt.days
        ct.index = pd.MultiIndex.from_product([[tk], ct.index], names=["asset", "crisis"])
        parts.append(ct)
    crises = pd.concat(parts)
    gaps = ["protection", "cycle_edge"]
    crises[gaps] = crises[gaps].astype(float).round(9) + 0.0  # no "-0.0" for flat legs
    ctx.save_table(f"{NAME}_crises", crises)
    crises_disp = _crisis_display(crises)

    spy_s, spy_b = pairs["SPY"]
    spy_c = crises.loc["SPY"]
    top2 = spy_c["protection"].nlargest(2)
    rest_mean = spy_c["protection"].drop(top2.index).mean()
    bars_title = (
        f"Protection is concentrated in two crises: {top2.index[0]} and {top2.index[1]}"
        if top2.min() > 3 * max(rest_mean, 0.0)
        else "Protection is spread across SPY's crises"
    )
    plotting.crisis_bars(
        spy_c[["strat_return", "bench_return"]],
        bars_title,
        subtitle=f"SPY, EMA50/200 vs buy & hold, return from peak to trough "
        f"(next-close fills, 5 bp/side, {spy_s.start.year}-{spy_s.end.year})",
        strategy_col="strat_return",
        bnh_col="bench_return",
        labels=("EMA50/200", "Buy & hold"),
        path=ctx.fig_path(f"{NAME}_crisis_bars"),
    )
    edge = crises[["protection", "cycle_edge"]].copy()
    edge.index = [f"{a} · {c}" for a, c in edge.index]
    edge.columns = ["Peak to trough (in window)", "Peak to B&H recovery or next crisis"]
    n_flip = int(((crises["protection"] > 0.005) & (crises["cycle_edge"] < -0.005)).sum())
    pal = plotting.TOKENS["light"]["categorical"]
    fig = plotting.bar_comparison(
        edge,
        f"Counting the late re-entry, {n_flip} in-window wins become losses",
        subtitle="EMA50/200 minus buy & hold, inside the crisis window vs over the full "
        "drawdown-and-recovery cycle",
        xlabel="EMA50/200 minus buy & hold (percentage points)",
        fmt=lambda v: f"{v * 100:+.1f}",
        colors=dict(zip(edge.columns, pal[:2], strict=True)),
    )
    ax = fig.axes[0]
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo - 0.08 * (hi - lo), hi)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda v, _: "0" if abs(v) < 1e-9 else f"{v * 100:+.0f}".replace("-", "−")
        )
    )
    plotting.save_figure(fig, ctx.fig_path(f"{NAME}_crisis_cycle"))

    # ---- regimes (SPY) -----------------------------------------------------
    labels = regimes.label_regimes(ctx.prices("SPY"), ctx.prices("^VIX"))
    labels = labels.reindex(spy_s.returns.index)
    dims = {"trend": "Trend (SPY vs SMA200)", "vol": "VIX tercile", "drawdown": "SPY drawdown"}
    reg_rows, bar_rows = [], {}
    for col, dim in dims.items():
        lab = labels[col].rename(col)
        perf = regimes.performance_by_regime(
            {"EMA50/200": spy_s.returns, "Buy & hold": spy_b.returns}, lab, rf=spy_s.rf
        )
        expo = spy_s.exposure.groupby(lab, observed=True).mean()
        for reg in perf.index:
            reg_rows.append(
                {
                    "dimension": dim,
                    "regime": str(reg),
                    "share": perf.loc[reg, ("EMA50/200", "share")],
                    "strat_exposure": float(expo.loc[reg]),
                    "strat_ann_return": perf.loc[reg, ("EMA50/200", "ann_return")],
                    "bench_ann_return": perf.loc[reg, ("Buy & hold", "ann_return")],
                    "strat_ann_vol": perf.loc[reg, ("EMA50/200", "ann_vol")],
                    "bench_ann_vol": perf.loc[reg, ("Buy & hold", "ann_vol")],
                    "strat_sharpe": perf.loc[reg, ("EMA50/200", "sharpe")],
                    "bench_sharpe": perf.loc[reg, ("Buy & hold", "sharpe")],
                }
            )
            short = {"trend": "Trend", "vol": "VIX", "drawdown": "DD"}[col]
            bar_rows[f"{short}: {reg}"] = {
                "EMA50/200": perf.loc[reg, ("EMA50/200", "sharpe")],
                "Buy & hold": perf.loc[reg, ("Buy & hold", "sharpe")],
            }
    reg = pd.DataFrame(reg_rows).set_index(["dimension", "regime"])
    ctx.save_table(f"{NAME}_regimes", reg)
    reg_disp = pd.DataFrame(index=[f"{d} · {r}" for d, r in reg.index])
    reg_disp.index.name = "Regime"
    reg_disp["Share of days"] = [_pct(v, 0) for v in reg["share"]]
    reg_disp["EMA exposure"] = [_pct(v, 0) for v in reg["strat_exposure"]]
    reg_disp["EMA ann. return"] = [_pct(v) for v in reg["strat_ann_return"]]
    reg_disp["B&H ann. return"] = [_pct(v) for v in reg["bench_ann_return"]]
    reg_disp["EMA vol"] = [_pct(v) for v in reg["strat_ann_vol"]]
    reg_disp["B&H vol"] = [_pct(v) for v in reg["bench_ann_vol"]]
    reg_disp["EMA Sharpe"] = [f"{v:.2f}" for v in reg["strat_sharpe"]]
    reg_disp["B&H Sharpe"] = [f"{v:.2f}" for v in reg["bench_sharpe"]]

    bars = pd.DataFrame(bar_rows).T
    n_better = int((bars["EMA50/200"] > bars["Buy & hold"]).sum())
    plotting.regime_bars(
        bars,
        f"EMA50/200 has the higher Sharpe in {n_better} of {len(bars)} regimes",
        subtitle="SPY, annualised Sharpe within each regime; labels known at the prior "
        "close (VIX terciles on an expanding history)",
        percent=False,
        fmt="{:.2f}",
        ylabel="Sharpe ratio (excess of T-bills)",
        path=ctx.fig_path(f"{NAME}_regime_bars"),
    )

    # ---- capture ratios and convexity --------------------------------------
    conv_rows = {}
    smiles = {}
    for tk, (s, b) in pairs.items():
        cap = regimes.capture_ratios(s.returns, b.returns, "ME")
        sm = regimes.convexity_smile(s.returns, b.returns, "QE", rf=s.rf)
        smiles[tk] = sm
        conv_rows[tk] = {
            "upside_capture": cap["upside_capture"],
            "downside_capture": cap["downside_capture"],
            "capture_ratio": cap["capture_ratio"],
            "smile_a": float(sm["coef"]["a"]),
            "smile_b": float(sm["coef"]["b"]),
            "smile_c": float(sm["coef"]["c"]),
            "t_c": sm["t_c"],
            "r2": sm["r2"],
            "quarters": sm["n"],
        }
    conv = pd.DataFrame(conv_rows).T
    conv.index.name = "asset"
    ctx.save_table(f"{NAME}_convexity", conv)
    conv_disp = pd.DataFrame(index=conv.index)
    conv_disp["Upside capture (monthly)"] = conv["upside_capture"].map(lambda v: _pct(v, 0))
    conv_disp["Downside capture (monthly)"] = conv["downside_capture"].map(lambda v: _pct(v, 0))
    conv_disp["Capture ratio"] = conv["capture_ratio"].map("{:.2f}".format)
    conv_disp["Beta b (quarterly)"] = conv["smile_b"].map("{:.2f}".format)
    conv_disp["Smile c"] = conv["smile_c"].map("{:+.2f}".format)
    conv_disp["t(c)"] = conv["t_c"].map("{:+.2f}".format)
    conv_disp["R²"] = conv["r2"].map("{:.2f}".format)
    conv_disp["Quarters"] = conv["quarters"].map(lambda v: f"{int(v)}")

    sm = smiles["SPY"]
    c_spy, t_spy = float(sm["coef"]["c"]), float(sm["t_c"])
    if t_spy > 2:
        smile_title = "SPY: a convex trend smile"
    elif t_spy < -2:
        smile_title = "SPY: a concave payoff, the opposite of a trend smile"
    else:
        smile_title = "SPY: no statistically meaningful trend smile in quarterly returns"
    plotting.smile_scatter(
        sm["data"]["x"],
        sm["data"]["y"],
        sm["coef"],
        smile_title,
        subtitle=f"Quarterly excess returns over T-bills, {sm['n']} quarters; "
        f"quadratic coefficient c = {c_spy:+.2f} (t = {t_spy:+.2f})",
        xlabel="SPY buy & hold quarterly excess return",
        ylabel="EMA50/200 quarterly excess return",
        path=ctx.fig_path(f"{NAME}_smile_scatter"),
    )

    # ---- rolling 3-year Sharpe difference ----------------------------------
    win = 252 * ROLL_YEARS
    rs_s = regimes.rolling_sharpe(spy_s.returns, win, rf=spy_s.rf)
    rs_b = regimes.rolling_sharpe(spy_b.returns, win, rf=spy_b.rf)
    diff = (rs_s - rs_b).dropna()
    share_pos = float((diff > 1e-3).mean())
    share_neg = float((diff < -1e-3).mean())
    roll_title = (
        "The rule trails in most 3-year windows and wins a few by a lot"
        if share_neg > share_pos
        else "The rule leads in most 3-year windows"
    )
    label = "EMA50/200 minus passive"
    fig = plotting.rolling_line(
        {label: diff},
        roll_title,
        ylabel="Δ trailing 3-year Sharpe",
        subtitle=f"SPY, trailing {ROLL_YEARS}-year Sharpe of EMA50/200 minus that of buy & hold; "
        "shaded: crisis windows; flat at zero = fully invested throughout",
        colors={label: pal[0]},
    )
    _shade_crises(fig, CRISES, diff.index[0], diff.index[-1])
    plotting.save_figure(fig, ctx.fig_path(f"{NAME}_rolling_line"))

    # ---- headline numbers --------------------------------------------------
    sum_s, sum_b = spy_s.summary(spy_b), spy_b.summary()
    slow = crises[crises["days"] > SLOW_DAYS]
    fast = crises[crises["days"] <= SLOW_DAYS]
    helped = crises[crises["cycle_edge"] > 0.02]
    hurt = crises[crises["cycle_edge"] < -0.02]
    fast_spy = spy_c[spy_c["days"] <= SLOW_DAYS]
    helped_spy = spy_c[spy_c["cycle_edge"] > 0.02]
    hurt_spy = spy_c[spy_c["cycle_edge"] < -0.02]
    bear = reg.loc[(dims["trend"], "bear")]
    high = reg.loc[(dims["vol"], "high")]
    ahead = sum_s["sharpe"] > sum_b["sharpe"]
    pos = {d: i for i, d in enumerate(spy_s.returns.index)}

    def short(c: str) -> str:
        return SHORT.get(c, c)

    bullets = []
    if len(helped_spy):
        slow_spy = spy_c[spy_c["days"] > SLOW_DAYS]
        n_slow_paid = int((slow_spy["cycle_edge"] > 0.02).sum())
        if not bool((helped_spy["days"] > SLOW_DAYS).all()):
            head = "Where it paid"
        elif n_slow_paid == len(slow_spy):
            head = "Every SPY episode longer than six months paid"
        else:
            head = f"Only {n_slow_paid} of {len(slow_spy)} SPY episodes longer than six months paid"

        def to_end(r) -> str:
            if r["capped"]:
                return f"to {pd.Timestamp(r['recovery']):%b %Y}, when the next crisis began"
            return "to recovery"

        bullets.append(
            f"- **{head}.** SPY "
            + "; ".join(
                f"{short(c)} {_pct(r['strat_return'], 0)} vs {_pct(r['bench_return'], 0)} "
                f"({_pp(r['cycle_edge'])} {to_end(r)})"
                for c, r in helped_spy.iterrows()
            )
            + f". Episodes longer than six months (all assets) averaged "
            f"{_pp(slow['protection'].mean())} in the window and {_pp(slow['cycle_edge'].mean())} "
            f"to recovery; shorter ones {_pp(fast['protection'].mean())} and "
            f"{_pp(fast['cycle_edge'].mean())}."
        )
    if len(fast_spy):
        line = (
            f"- **Fast crashes {'cost' if fast_spy['cycle_edge'].mean() < 0 else 'still paid'}.** "
            "SPY to recovery: "
            + "; ".join(
                f"{short(c)} {_pp(r['cycle_edge'])}"
                + (f" (missed {_pct(r['missed_rebound'], 0)})" if r["missed_rebound"] > 0 else "")
                for c, r in fast_spy.iterrows()
            )
            + "."
        )
        if "COVID crash" in spy_c.index and pd.notna(spy_c.loc["COVID crash", "exit_date"]):
            cv = spy_c.loc["COVID crash"]
            gap = pos[cv["exit_date"]] - pos[cv["trough"]]
            when = (
                "the trough day itself"
                if gap == 0
                else f"{abs(gap)} bars {'after' if gap > 0 else 'before'} the trough"
            )
            line += (
                f" In COVID its first day in cash was {when}, and it was back in "
                f"{_num(cv['reentry_lag_bars'])} bars after the low"
            )
            tf = spy_c.loc["2025 tariff shock"] if "2025 tariff shock" in spy_c.index else None
            if tf is not None and pd.isna(tf["exit_date"]) and tf["post_trough_exits"] > 0:
                line += "; in 2025 it held through the low and sold into the rebound"
            line += "."
        bullets.append(line)
    if "2022 inflation bear" in spy_c.index:
        r22 = spy_c.loc["2022 inflation bear"]
        bullets.append(
            f"- **In-window numbers {'flatter' if crises['protection'].mean() > crises['cycle_edge'].mean() else 'understate'} the rule.** SPY 2022: "
            f"{_pp(r22['protection'])} in the window, {_pp(r22['cycle_edge'])} to recovery "
            f"(re-entry {_num(r22['reentry_lag_bars'])} bars after the low). Over all "
            f"{len(crises)} episodes it helped (> +2 pp) in {len(helped)} "
            f"({', '.join(f'{a} {short(c)}' for a, c in helped.index)}) and hurt (< −2 pp) "
            f"in {len(hurt)}."
        )
    bullets.append(
        f"- **Regimes.** Below SPY's SMA200 (close to the rule's own signal) exposure was "
        f"{_pct(bear['strat_exposure'], 0)}, yet buy & hold still "
        f"{'earned' if bear['bench_ann_return'] > 0 else 'lost'} "
        f"{_pct(abs(bear['bench_ann_return']))} a year there (Sharpe "
        f"{bear['bench_sharpe']:.2f} vs {bear['strat_sharpe']:.2f} for the rule): "
        "below-trend days include the sharpest rebounds. In the top VIX tercile exposure was "
        f"{_pct(high['strat_exposure'], 0)}, "
        f"earning {_pct(high['strat_ann_return'])} a year vs {_pct(high['bench_ann_return'])} at "
        f"{_pct(high['strat_ann_vol'])} vs {_pct(high['bench_ann_vol'])} vol. Full sample: Sharpe "
        f"{sum_s['sharpe']:.2f} vs {sum_b['sharpe']:.2f}, max drawdown "
        f"{_pct(sum_s['max_drawdown'])} vs {_pct(sum_b['max_drawdown'])}."
    )
    others = [tk for tk in conv.index if tk != "SPY"]
    other_smiles = "; ".join(
        f"{tk} {conv.loc[tk, 'smile_c']:+.2f} (t = {conv.loc[tk, 't_c']:+.2f})" for tk in others
    )
    smile_head = "No smile on SPY" if c_spy < 0 or t_spy < 2 else "A smile on SPY"
    bullets.append(
        f"- **{smile_head}.** Monthly capture {_pct(conv.loc['SPY', 'upside_capture'], 0)} up / "
        f"{_pct(conv.loc['SPY', 'downside_capture'], 0)} down; quarterly c = {c_spy:+.2f} "
        f"(t = {t_spy:+.2f})" + (f"; {other_smiles}." if others else ".")
    )
    big = spy_c[(spy_c["days"] > SLOW_DAYS) & (spy_c["cycle_edge"] > 0.02)]
    lead = diff[diff > 1e-3]
    in_big = pd.Series(False, index=lead.index)
    for _, r in big.iterrows():
        # a trailing window ending at d covers (d - ROLL_YEARS, d]
        in_big |= (lead.index >= r["start"]) & (
            lead.index - pd.DateOffset(years=ROLL_YEARS) <= r["end"]
        )
    share_big = float(in_big.mean()) if len(lead) else math.nan
    big_names = " and ".join(short(c) for c in big.index)
    conc = (
        f"; {_pct(share_big, 0)} of the leading windows contain the {big_names}"
        if len(big) and len(lead)
        else ""
    )
    bullets.append(
        f"- **{'Concentrated' if share_neg > share_pos else 'Broad'} payoff.** The trailing "
        f"3-year Sharpe led buy & hold in {_pct(share_pos, 0)} of windows and trailed in "
        f"{_pct(share_neg, 0)} (median {diff.median():+.2f}){conc}, so the "
        f"{'higher' if ahead else 'lower'} full-sample Sharpe rests on "
        f"{'those episodes' if len(big) and share_big > 0.5 else 'a broad set of windows'}."
    )

    # interpretation pieces chosen from the data
    slow_helped = [short(c) for c, r in helped_spy.iterrows() if r["days"] > SLOW_DAYS]
    fast_hurt = [short(c) for c, r in hurt_spy.iterrows() if r["days"] <= SLOW_DAYS]
    slow_hurt = [short(c) for c, r in hurt_spy.iterrows() if r["days"] > SLOW_DAYS]
    convex = [tk for tk in conv.index if conv.loc[tk, "t_c"] > 2]
    concave = [tk for tk in conv.index if conv.loc[tk, "t_c"] < -2]
    flat = [tk for tk in conv.index if abs(conv.loc[tk, "t_c"]) <= 2]
    parts = []
    if concave:
        parts.append(f"{' and '.join(concave)} {'is' if len(concave) == 1 else 'are'} concave")
    if flat:
        parts.append(
            f"{' and '.join(flat)} {'shows' if len(flat) == 1 else 'show'} no significant curvature"
        )
    if convex:
        parts.append(
            f"{' and '.join(convex)} {'shows' if len(convex) == 1 else 'show'} a smile, "
            "earned in deep bear quarters spent in cash"
        )
    interp = (
        "**Interpretation.** A moving-average crossover is a low-pass filter, so it lags by "
        "construction (Levine & Pedersen 2016)."
    )
    if slow_helped:
        interp += (
            f" The lag is harmless in long, deep bears ({', '.join(slow_helped)}) that give it "
            "months to step aside."
        )
    if fast_hurt:
        interp += (
            f" It is costly in V-shaped shocks ({', '.join(fast_hurt)}), where the exit comes "
            "near (or after) the low and the re-entry after the rebound"
            + (
                f", and in shallow corrections ({', '.join(slow_hurt)}) too small to pay for the "
                "round trip."
                if slow_hurt
                else "."
            )
        )
    interp += (
        " The crisis alpha and straddle-like payoff of Fung & Hsieh (2001) and Hurst, Ooi & "
        "Pedersen (2017) belong to diversified long-short futures books; a long-or-cash rule "
        "cannot profit from a crash and gives up rebound quarters"
        + (f" ({'; '.join(parts)}; plain OLS t-statistics, so indicative only)." if parts else ".")
        + (" Its value hinges on the sample containing a slow, deep bear." if slow_helped else "")
    )

    text = (
        '**Question.** Does EMA50/200 earn the "crisis alpha" trend following is sold on, and '
        "what does it pay for it when crashes are fast?\n\n"
        f"**Setup.** EMA50/200 vs buy & hold on SPY (from {spy_s.start.date()})"
        + ", QQQ and BTC-USD"
        + ", next-close fills, default costs. Crisis windows are peak to trough "
        "(`universe.CRISES`, plus the 2018 and 2021-22 crypto crashes for BTC); because a rule "
        "that exits near the low looks fine inside the window, each crisis is also measured "
        "from the peak until buy & hold regained its pre-crisis level (or the next crisis "
        "began). Regime labels (SPY vs "
        "SMA200, VIX terciles, SPY drawdown) are known at the prior close.\n\n"
        "**Findings.**\n" + "\n".join(bullets) + "\n\n" + interp + "\n"
    )
    after = (
        "Protection compounds returns strictly inside each peak-to-trough window; the "
        "peak-to-recovery edge extends the window until buy & hold regains its starting level, "
        "so it charges the rule for the rebound it missed; it is cut at the start of the next "
        "listed crisis so no period is counted twice"
        + (
            " ("
            + "; ".join(
                f"{a} {short(c)} stops at {pd.Timestamp(r['recovery']):%B %Y}"
                for (a, c), r in crises[crises["capped"]].iterrows()
            )
            + ")"
            if bool(crises["capped"].any())
            else ""
        )
        + ". Long uncapped cycles (SPY 2000-06) still include unrelated whipsaws after the "
        "crisis. QQQ's "
        "dot-com bust is not covered: after the EMA200 warm-up its backtest starts in 2001. "
        "Quarterly convexity t-statistics are plain OLS and only indicative. The crisis list "
        "and the two crypto windows are dated peak to trough with hindsight, which is fine for "
        "attribution but would be look-ahead in a trading rule."
    )

    def key(*parts: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", "_".join(parts).lower().replace("-usd", "")).strip("_")

    results = {
        "spy_start": spy_s.start.date().isoformat(),
        "spy_sharpe_ema": sum_s["sharpe"],
        "spy_sharpe_bnh": sum_b["sharpe"],
        "spy_maxdd_ema": sum_s["max_drawdown"],
        "spy_maxdd_bnh": sum_b["max_drawdown"],
        "slow_mean_protection": float(slow["protection"].mean()),
        "fast_mean_protection": float(fast["protection"].mean()),
        "slow_mean_cycle_edge": float(slow["cycle_edge"].mean()),
        "fast_mean_cycle_edge": float(fast["cycle_edge"].mean()),
        "n_crises": int(len(crises)),
        "n_helped": int(len(helped)),
        "n_hurt": int(len(hurt)),
        "spy_upside_capture": float(conv.loc["SPY", "upside_capture"]),
        "spy_downside_capture": float(conv.loc["SPY", "downside_capture"]),
        "spy_smile_c": c_spy,
        "spy_smile_t": t_spy,
        "spy_rolling3y_share_ahead": share_pos,
        "spy_rolling3y_share_behind": share_neg,
        "spy_rolling3y_median_diff": float(diff.median()),
        "spy_bear_regime_exposure": float(bear["strat_exposure"]),
        "spy_high_vix_exposure": float(high["strat_exposure"]),
    }
    for (a, c), r in crises.iterrows():
        results[key(a, c, "protection")] = round(float(r["protection"]), 6)
        results[key(a, c, "cycle_edge")] = round(float(r["cycle_edge"]), 6)
        if pd.notna(r["missed_rebound"]):
            results[key(a, c, "missed_rebound")] = float(r["missed_rebound"])
    for tk in others:
        results[key(tk, "smile_c")] = float(conv.loc[tk, "smile_c"])
        results[key(tk, "smile_t")] = float(conv.loc[tk, "t_c"])

    figures = [
        (
            "SPY peak-to-trough returns, EMA50/200 vs buy & hold",
            ctx.fig_path(f"{NAME}_crisis_bars"),
        ),
        (
            "Strategy minus buy & hold: in-window vs peak-to-recovery",
            ctx.fig_path(f"{NAME}_crisis_cycle"),
        ),
        ("SPY Sharpe by regime", ctx.fig_path(f"{NAME}_regime_bars")),
        ("Quarterly convexity of EMA50/200 against SPY", ctx.fig_path(f"{NAME}_smile_scatter")),
        ("Rolling 3-year Sharpe difference on SPY", ctx.fig_path(f"{NAME}_rolling_line")),
    ]
    tables = [
        ("Crisis windows: timing, missed rebound and peak-to-recovery edge", crises_disp),
        ("SPY performance by regime", reg_disp),
        ("Capture ratios and quarterly convexity", conv_disp),
    ]
    return Section(
        name=NAME,
        title=TITLE,
        text=text,
        figures=figures,
        tables=tables,
        after=after,
        results=results,
    )

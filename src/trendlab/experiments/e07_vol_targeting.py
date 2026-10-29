"""Experiment 7: does volatility targeting help, and where?

Pre-specified design (fixed before looking at results):

* Overlay: :class:`~trendlab.strategies.VolTarget` with a 63-bar (about three
  months) trailing realised-vol estimate, rebalanced daily, applied to
  buy-and-hold and to the EMA50/200 rule.
* Target: 10% a year for every ETF (the conventional institutional risk budget
  used by Harvey et al. 2018). Bitcoin's realised vol is 60-80%, so a 10%
  target would turn it into a permanent ~15% position and the comparison would
  be about de-risking, not timing; it gets 40%, roughly half its long-run vol,
  which keeps the average exposure in the same 0.5-0.8 range as the equity ETFs.
* Leverage caps: 1x (long-only, no borrowing -- what a retail account can do)
  and 2x (closer to the unconstrained scaling studied in the literature;
  financed at the T-bill rate, which flatters it).
* All six variants of an asset are evaluated on one common window (the first
  bar at which EMA50/200 and the vol estimate are both warmed up).

The mechanism is measured separately on monthly data: persistence of realised
vol (vol clustering), the contemporaneous return / vol-change correlation (the
leverage effect) and the correlation of this month's realised vol with *next*
month's excess return (does high vol predict high returns? If not, scaling
down in high vol raises the Sharpe ratio -- Moreira & Muir 2017).
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from .. import plotting
from ..engine import evaluation_start
from ..metrics import format_table
from ..report import df_to_markdown
from ..stats import bootstrap_sharpe_difference
from ..strategies import BuyAndHold, EMACrossover, VolTarget, scaled
from ..universe import ASSETS
from .common import Context, Section

NAME = "e07_vol_targeting"
TITLE = "7. Volatility targeting: where does it help, and why?"

ASSETS_VT = ["SPY", "QQQ", "EFA", "TLT", "IEF", "GLD", "BTC-USD"]
RISK_ASSETS = ["SPY", "QQQ", "EFA"]  # Harvey et al. (2018): equities / credit
DEFENSIVE = ["TLT", "IEF", "GLD"]  # bonds, gold: no leverage effect expected
TARGET_VOL = 0.10
TARGET_VOL_CRYPTO = 0.40
WINDOW = 63
CAPS = (1.0, 2.0)

VARIANTS = ["B&H", "B&H + VT", "B&H + VT 2x", "EMA50/200", "EMA + VT", "EMA + VT 2x"]
DISPLAY_VARIANTS = ["B&H", "B&H + VT", "EMA50/200", "EMA + VT"]


def target_for(ticker: str) -> float:
    return TARGET_VOL_CRYPTO if ASSETS.get(ticker, {}).get("class") == "Crypto" else TARGET_VOL


def _strategies(ticker: str, window: int = WINDOW) -> dict:
    tv = target_for(ticker)
    bh, ema = BuyAndHold(), EMACrossover(50, 200)
    return {
        "B&H": bh,
        "B&H + VT": VolTarget(bh, tv, window, CAPS[0]),
        "B&H + VT 2x": VolTarget(bh, tv, window, CAPS[1]),
        "EMA50/200": ema,
        "EMA + VT": VolTarget(ema, tv, window, CAPS[0]),
        "EMA + VT 2x": VolTarget(ema, tv, window, CAPS[1]),
    }


def _save(ctx: Context, name: str, raw: pd.DataFrame, display: pd.DataFrame) -> None:
    """Raw numbers to CSV, the formatted copy to Markdown."""
    ctx.save_table(name, raw)
    (ctx.table_dir / f"{name}.md").write_text(df_to_markdown(display) + "\n")


def mechanism(asset_returns: pd.Series, rf: pd.Series, periods: int) -> dict:
    """Monthly vol clustering, leverage effect and vol -> next-month-return predictability."""
    df = pd.DataFrame({"r": asset_returns, "rf": rf}).iloc[1:]
    grp = df.groupby(df.index.to_period("M"))
    monthly = pd.DataFrame(
        {
            "ret": grp["r"].apply(lambda x: float((1 + x).prod() - 1)),
            "rf": grp["rf"].apply(lambda x: float((1 + x).prod() - 1)),
            "rv": grp["r"].std() * math.sqrt(periods),
            "n": grp["r"].count(),
        }
    )
    monthly = monthly[monthly["n"] >= 10]
    ex = monthly["ret"] - monthly["rf"]
    rv = monthly["rv"]
    nxt = ex.shift(-1)
    ok = nxt.notna()
    n = int(ok.sum())
    rho_pred = float(np.corrcoef(rv[ok], nxt[ok])[0, 1])
    # Sharpe of the next month conditional on this month's vol (top vs bottom tercile)
    q_lo, q_hi = rv[ok].quantile([1 / 3, 2 / 3])
    hi, lo = nxt[ok][rv[ok] >= q_hi], nxt[ok][rv[ok] <= q_lo]
    sr_hi = float(hi.mean() / hi.std() * math.sqrt(12))
    sr_lo = float(lo.mean() / lo.std() * math.sqrt(12))
    return {
        "months": n,
        "vol_persistence": float(rv.autocorr(1)),
        "leverage_effect": float(ex.corr(rv.diff())),
        "corr_vol_next_ret": rho_pred,
        "t_vol_next_ret": rho_pred * math.sqrt(n - 2) / math.sqrt(max(1e-12, 1 - rho_pred**2)),
        "sharpe_after_low_vol": sr_lo,
        "sharpe_after_high_vol": sr_hi,
    }


def _exposure_chart(
    spy_vol: pd.Series, exposure: pd.Series, ema_out: pd.Series, target: float, path
):
    lows = []
    for yr in (2008, 2020):
        e = exposure[exposure.index.year == yr]
        if len(e):
            lows.append(f"{e.min():.0%} in {e.idxmin():%b %Y}")
    title = (
        "When SPY volatility spikes the overlay de-risks: exposure fell to " + " and ".join(lows)
        if lows
        else "When SPY volatility spikes, the overlay scales exposure down"
    )
    """SPY trailing vol (top) against the exposure the overlay holds (bottom)."""
    with plotting.theme() as t:
        fig, (top, bot) = plt.subplots(
            2, 1, figsize=(10, 6.6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.0]}
        )
        top.plot(spy_vol.index, spy_vol.to_numpy(), color=t["benchmark"], lw=1.4, label="SPY")
        top.axhline(
            target, color=t["categorical"][1], lw=1.4, ls="--", label=f"{target:.0%} target"
        )
        top.set_ylabel(f"{WINDOW}-day realised vol")
        top.set_ylim(0, None)
        plotting._pct_axis(top.yaxis)
        top.tick_params(axis="x", length=0)
        top.legend(loc="upper right", ncols=2)
        # shade spans where the EMA50/200 rule is out of the market
        out = ema_out.astype(bool).to_numpy()
        idx = ema_out.index
        edges = np.flatnonzero(np.diff(np.concatenate([[0], out.astype(int), [0]])))
        for k, (i0, i1) in enumerate(zip(edges[::2], edges[1::2], strict=True)):
            bot.axvspan(
                idx[i0],
                idx[min(i1, len(idx) - 1)],
                color=t["grid"],
                alpha=0.8,
                lw=0,
                label="EMA50/200 out of market" if k == 0 else None,
            )
        bot.plot(
            exposure.index,
            exposure.to_numpy(),
            color=t["categorical"][0],
            lw=1.2,
            label="Buy & hold + VT exposure",
        )
        bot.axhline(1.0, color=t["axis"], lw=0.8)
        bot.set_ylim(0, 1.12)
        bot.set_ylabel("Exposure held")
        plotting._pct_axis(bot.yaxis)
        bot.legend(loc="lower left", ncols=2)
        plotting._date_axis(bot)
        plotting._titles(
            top,
            title,
            f"SPY {WINDOW}-day realised vol and the exposure of a {target:.0%}-target overlay "
            "(1x cap); grey spans: EMA50/200 in cash",
        )
        return plotting.save_figure(fig, path)


def _mechanism_scatter(df: pd.DataFrame, title: str, path):
    """Predictive corr(vol_t, r_t+1) against the Sharpe gain from vol targeting B&H."""
    with plotting.theme() as t:
        fig, ax = plt.subplots(figsize=(8, 5.4))
        ax.axhline(0.0, color=t["axis"], lw=0.8, zorder=1)
        ax.axvline(0.0, color=t["axis"], lw=0.8, zorder=1)
        for tk, row in df.iterrows():
            risk = tk in RISK_ASSETS
            c = t["categorical"][0] if risk else t["categorical"][1]
            if tk == "BTC-USD":
                c = t["categorical"][3]
            ax.scatter(row["x"], row["y"], s=70, color=c, edgecolor=t["surface"], lw=1.5, zorder=3)
            ax.annotate(
                tk,
                xy=(row["x"], row["y"]),
                xytext=(7, 4),
                textcoords="offset points",
                fontsize=9,
                color=t["ink"],
            )
        for label, c in (
            ("Equities", t["categorical"][0]),
            ("Bonds / gold", t["categorical"][1]),
            ("Bitcoin", t["categorical"][3]),
        ):
            ax.scatter([], [], s=50, color=c, label=label)
        ax.legend(loc="upper right")
        ax.grid(axis="both")
        ax.set_xlabel("corr(realised vol this month, excess return next month)")
        ax.set_ylabel("Sharpe change, B&H + VT minus B&H")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: plotting._minus(f"{v:.2f}")))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: plotting._minus(f"{v:+.2f}")))
        ax.yaxis.set_major_locator(MaxNLocator(7, steps=[1, 2, 5, 10]))
        ax.xaxis.set_major_locator(MaxNLocator(8, steps=[1, 2, 5, 10]))
        xs, ys = df["x"].to_numpy(), df["y"].to_numpy()
        pad_x = 0.15 * (xs.max() - xs.min() + 1e-3)
        pad_y = 0.2 * (ys.max() - ys.min() + 1e-3)
        ax.set_xlim(xs.min() - pad_x, xs.max() + 2 * pad_x)
        ax.set_ylim(min(ys.min(), 0) - pad_y, ys.max() + pad_y)
        plotting._titles(
            ax,
            title,
            "Monthly predictive correlation vs Sharpe gain of the 1x-capped overlay; seven assets, so suggestive only",
        )
        return plotting.save_figure(fig, path)


def run(ctx: Context) -> Section:
    n_boot = 300 if ctx.quick else 2000
    rows: dict[str, dict] = {}
    results_by: dict[str, dict] = {}
    mech: dict[str, dict] = {}
    tests: dict[str, dict] = {}
    spy_parts = None

    for tk in ASSETS_VT:
        px = ctx.prices(tk)
        strats = _strategies(tk)
        start = evaluation_start(px, list(strats.values()))
        res = {k: ctx.backtest(tk, s, start=start) for k, s in strats.items()}
        results_by[tk] = res
        for k, r in res.items():
            m = r.summary()
            rows[f"{tk} · {k}"] = {
                "asset": tk,
                "variant": k,
                "start": m["start"],
                "cagr": m["cagr"],
                "vol": m["vol"],
                "sharpe": m["sharpe"],
                "max_drawdown": m["max_drawdown"],
                "exposure": m["exposure"],
                "turnover_per_year": m["turnover_per_year"],
            }
        bh = res["B&H"]
        ppy = bh.periods_per_year
        mech[tk] = mechanism(bh.asset_returns, bh.rf, ppy)
        for base, vt in (("B&H", "B&H + VT"), ("EMA50/200", "EMA + VT")):
            bt = bootstrap_sharpe_difference(
                res[vt].returns, res[base].returns, rf=bh.rf, n_boot=n_boot, periods=ppy
            )
            tests[f"{tk} · {base}"] = {k: v for k, v in bt.items() if k != "samples"}
        if tk == "SPY":
            rets = bh.asset_returns.iloc[1:]
            spy_parts = {
                "vol": rets.rolling(WINDOW, min_periods=WINDOW // 2).std() * math.sqrt(ppy),
                "exp": res["B&H + VT"].exposure.iloc[1:],
                "ema_out": res["EMA50/200"].exposure.iloc[1:] <= 0,
            }

    perf = pd.DataFrame(rows).T
    num_cols = ["cagr", "vol", "sharpe", "max_drawdown", "exposure", "turnover_per_year"]
    perf[num_cols] = perf[num_cols].astype(float)
    disp_rows = {
        f"{r.asset} · {r.variant}": {c: getattr(r, c) for c in num_cols}
        for r in perf.itertuples()
        if r.variant in DISPLAY_VARIANTS
    }
    perf_disp = format_table(disp_rows, num_cols).rename(
        columns={
            "cagr": "CAGR",
            "vol": "Vol",
            "sharpe": "Sharpe",
            "max_drawdown": "Max DD",
            "exposure": "Avg exposure",
            "turnover_per_year": "Turnover / yr",
        }
    )
    perf_disp = perf_disp.map(lambda v: plotting._minus(str(v)))  # U+2212 like the figures
    perf_disp.index.name = "Asset · strategy"
    _save(ctx, f"{NAME}_performance", perf, perf_disp)

    def s(tk, v, key="sharpe"):
        return float(perf.loc[f"{tk} · {v}", key])

    # Sharpe change from the overlay, per asset and variant
    dsr = pd.DataFrame(
        {
            "B&H + VT (1x cap)": [s(t, "B&H + VT") - s(t, "B&H") for t in ASSETS_VT],
            "B&H + VT (2x cap)": [s(t, "B&H + VT 2x") - s(t, "B&H") for t in ASSETS_VT],
            "EMA + VT (1x cap)": [s(t, "EMA + VT") - s(t, "EMA50/200") for t in ASSETS_VT],
            "EMA + VT (2x cap)": [s(t, "EMA + VT 2x") - s(t, "EMA50/200") for t in ASSETS_VT],
        },
        index=ASSETS_VT,
    )
    fig_bars = ctx.fig_path(f"{NAME}_sharpe_change")
    n_up_any = int((dsr["B&H + VT (1x cap)"] > 0).sum())
    n_sig_any = sum(tests[f"{t} · B&H"]["p_value"] < 0.05 for t in ASSETS_VT)
    sig_tag = "none significantly" if n_sig_any == 0 else f"{n_sig_any} significantly"
    plotting.bar_comparison(
        dsr,
        f"Vol targeting raises buy-and-hold Sharpe in {n_up_any} of {len(ASSETS_VT)} assets, "
        f"by at most {dsr['B&H + VT (1x cap)'].max():.2f}; {sig_tag}",
        subtitle=(
            f"Change in Sharpe ratio from a {TARGET_VOL:.0%} vol-target overlay "
            f"({TARGET_VOL_CRYPTO:.0%} for BTC), {WINDOW}-day estimate, same window per asset"
        ),
        percent=False,
        fmt=lambda v: "0.00" if round(v, 2) == 0 else f"{v:+.2f}",
        xlabel="Sharpe(with overlay) minus Sharpe(without)",
        colors={
            "B&H + VT (1x cap)": plotting.CATEGORICAL[0],
            "B&H + VT (2x cap)": plotting.SEQUENTIAL[3],
            "EMA + VT (1x cap)": plotting.CATEGORICAL[1],
            "EMA + VT (2x cap)": "#f5b08f",
        },
        path=fig_bars,
    )

    # mechanism table
    mech_df = pd.DataFrame(mech).T
    mech_df["dsharpe_bh_vt"] = dsr["B&H + VT (1x cap)"]
    mech_df["dsharpe_ema_vt"] = dsr["EMA + VT (1x cap)"]
    mech_df["p_bh_vt"] = [tests[f"{t} · B&H"]["p_value"] for t in ASSETS_VT]
    mech_df["p_ema_vt"] = [tests[f"{t} · EMA50/200"]["p_value"] for t in ASSETS_VT]
    mech_df["ci_low_bh_vt"] = [tests[f"{t} · B&H"]["ci_low"] for t in ASSETS_VT]
    mech_df["ci_high_bh_vt"] = [tests[f"{t} · B&H"]["ci_high"] for t in ASSETS_VT]
    mech_df = mech_df.astype(float)
    mech_df["months"] = mech_df["months"].astype(int)
    mech_disp = pd.DataFrame(index=ASSETS_VT)
    mech_disp["Months"] = mech_df["months"]
    mech_disp["Vol persistence ρ(σₜ, σₜ₊₁)"] = mech_df["vol_persistence"].map("{:.2f}".format)
    mech_disp["Leverage effect ρ(rₜ, Δσₜ)"] = mech_df["leverage_effect"].map("{:.2f}".format)
    mech_disp["ρ(σₜ, rₜ₊₁)"] = mech_df["corr_vol_next_ret"].map("{:.2f}".format)
    mech_disp["Next-month SR after low / high vol"] = [
        f"{a:.2f} / {b:.2f}"
        for a, b in zip(
            mech_df["sharpe_after_low_vol"], mech_df["sharpe_after_high_vol"], strict=True
        )
    ]
    mech_disp["ΔSharpe B&H+VT [95% CI]"] = [
        f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        for d, lo, hi in zip(
            mech_df["dsharpe_bh_vt"], mech_df["ci_low_bh_vt"], mech_df["ci_high_bh_vt"], strict=True
        )
    ]
    mech_disp["p (B&H)"] = mech_df["p_bh_vt"].map("{:.3f}".format)
    mech_disp["ΔSharpe EMA+VT"] = mech_df["dsharpe_ema_vt"].map("{:+.2f}".format)
    mech_disp["p (EMA)"] = mech_df["p_ema_vt"].map("{:.3f}".format)
    for col in mech_disp.columns[1:]:
        mech_disp[col] = mech_disp[col].map(plotting._minus)
    mech_disp.index.name = "Asset"
    _save(ctx, f"{NAME}_mechanism", mech_df, mech_disp)

    fig_scatter = ctx.fig_path(f"{NAME}_mechanism")
    xc = float(np.corrcoef(mech_df["corr_vol_next_ret"], mech_df["dsharpe_bh_vt"])[0, 1])
    scatter_title = (
        f"Sharpe gains are larger where high vol does not predict high returns (ρ = {xc:.2f})"
        if xc < -0.3
        else f"Little cross-asset link between vol predictability and the gain (ρ = {xc:.2f})"
    )
    _mechanism_scatter(
        pd.DataFrame({"x": mech_df["corr_vol_next_ret"], "y": mech_df["dsharpe_bh_vt"]}),
        plotting._minus(scatter_title),
        fig_scatter,
    )

    fig_spy = ctx.fig_path(f"{NAME}_spy_exposure")
    _exposure_chart(
        spy_parts["vol"], spy_parts["exp"], spy_parts["ema_out"], target_for("SPY"), fig_spy
    )

    # robustness of the window choice (not used to pick anything)
    windows = [63] if ctx.quick else [21, 63, 126]
    win_rows = {}
    for w in windows:
        d = {}
        for tk in ASSETS_VT:
            px = ctx.prices(tk)
            vt = VolTarget(BuyAndHold(), target_for(tk), w, CAPS[0])
            st = evaluation_start(px, list(_strategies(tk).values()) + [vt])
            a = ctx.backtest(tk, vt, start=st).summary()["sharpe"]
            b = ctx.backtest(tk, BuyAndHold(), start=st).summary()["sharpe"]
            d[tk] = a - b
        win_rows[f"{w}-day"] = d
    win_df = pd.DataFrame(win_rows).T
    win_df.index.name = "Vol window"
    win_disp = win_df.apply(lambda c: c.map(lambda v: plotting._minus(f"{v:+.2f}")))
    _save(ctx, f"{NAME}_window_robustness", win_df, win_disp)

    # ------------------------------------------------------------------ text
    risk_d = dsr.loc[RISK_ASSETS, "B&H + VT (1x cap)"]
    def_d = dsr.loc[DEFENSIVE, "B&H + VT (1x cap)"]
    btc_d = float(dsr.loc["BTC-USD", "B&H + VT (1x cap)"])
    ema_d = dsr["EMA + VT (1x cap)"]
    n_up = int((dsr["B&H + VT (1x cap)"] > 0).sum())
    n_sig = int((mech_df["p_bh_vt"] < 0.05).sum())
    n_sig_ema = int((mech_df["p_ema_vt"] < 0.05).sum())
    risk_rho = mech_df.loc[RISK_ASSETS, "corr_vol_next_ret"].mean()
    def_rho = mech_df.loc[DEFENSIVE, "corr_vol_next_ret"].mean()
    risk_lev = mech_df.loc[RISK_ASSETS, "leverage_effect"].mean()
    def_lev = mech_df.loc[DEFENSIVE, "leverage_effect"].mean()
    sr_drop = mech_df["sharpe_after_high_vol"] < mech_df["sharpe_after_low_vol"]
    n_drop_risk = int(sr_drop.loc[RISK_ASSETS].sum())
    n_drop_def = int(sr_drop.loc[DEFENSIVE].sum())
    spy_dd_bh, spy_dd_vt = s("SPY", "B&H", "max_drawdown"), s("SPY", "B&H + VT", "max_drawdown")
    spy_cagr_bh, spy_cagr_vt = s("SPY", "B&H", "cagr"), s("SPY", "B&H + VT", "cagr")
    spy_exp = s("SPY", "B&H + VT", "exposure")
    spy_to = s("SPY", "B&H + VT", "turnover_per_year")
    spy_px = ctx.prices("SPY")
    spy_start = evaluation_start(spy_px, list(_strategies("SPY").values()))
    const = ctx.backtest("SPY", scaled(BuyAndHold(), spy_exp), start=spy_start).summary()
    lev2_gain = float(
        (dsr["B&H + VT (2x cap)"] - dsr["B&H + VT (1x cap)"]).loc[RISK_ASSETS].abs().max()
    )
    ief_2x = float(dsr.loc["IEF", "B&H + VT (2x cap)"])
    if risk_d.min() > def_d.max():
        split = "clean"
    elif risk_d.mean() > def_d.mean():
        split = "weak"
    else:
        split = "none"
    win_mean = win_df[RISK_ASSETS].mean(axis=1)
    win_range = (float(win_mean.min()), float(win_mean.max())) if len(win_df) > 1 else None

    def pm(v, d=2):
        return plotting._minus(f"{round(float(v), d) + 0.0:+.{d}f}")

    def pct(v, d=0):
        return plotting._minus(f"{v:.{d}%}")

    timing_dd = spy_dd_vt - const["max_drawdown"]
    size_dd = const["max_drawdown"] - spy_dd_bh
    timing_share = timing_dd / (timing_dd + size_dd) if timing_dd + size_dd > 0 else float("nan")
    split_pp = f"{timing_dd * 100:.0f} pp from timing vs {size_dd * 100:.0f} pp from holding less"
    if 0.4 <= timing_share <= 0.6:
        dd_attrib = f"timing and simply holding less share the drawdown reduction ({split_pp})"
    elif timing_share > 0.6:
        dd_attrib = f"most of the drawdown reduction comes from timing ({split_pp})"
    else:
        dd_attrib = f"most of the drawdown reduction comes simply from holding less ({split_pp})"
    odd = [
        f"{t} changes by {pm(risk_d[t])} despite a strong leverage effect"
        for t in RISK_ASSETS
        if risk_d[t] <= 0.01 and mech_df.loc[t, "leverage_effect"] < -0.2
    ]
    odd += [
        f"{t} improves by {pm(def_d[t])} without one"
        for t in DEFENSIVE
        if def_d[t] >= 0.05 and abs(mech_df.loc[t, "leverage_effect"]) < 0.15
    ]
    odd_line = (", and " + " while ".join(odd)) if odd else ""
    dd_gain = [s(t, "B&H + VT", "max_drawdown") - s(t, "B&H", "max_drawdown") for t in RISK_ASSETS]
    shape_line = (
        f"The robust benefit is to the *shape* of the equity return distribution (max drawdown "
        f"{min(dd_gain) * 100:.0f} to {max(dd_gain) * 100:.0f} pp shallower) rather than to "
        f"the Sharpe ratio."
        if min(dd_gain) > 0.05
        else "Drawdown improvements are also uneven across equity markets."
    )
    ema_vs_bh = (
        f"below the {pm(dsr['B&H + VT (1x cap)'].mean())} on buy-and-hold: the crossover "
        "already sits out many high-vol bear markets, so the overlays overlap"
        if ema_d.mean() < dsr["B&H + VT (1x cap)"].mean()
        else f"no smaller than the {pm(dsr['B&H + VT (1x cap)'].mean())} on buy-and-hold"
    )
    ief_note = (
        f"IEF's vol sits below the target, so at 1x the overlay is nearly a no-op (and "
        f"{pm(ief_2x)} when allowed to lever to 2x)"
        if s("IEF", "B&H + VT", "exposure") > 0.95
        else f"IEF changes by {pm(ief_2x)} when allowed to lever to 2x"
    )
    findings = [
        f"- **Equities:** buy-and-hold Sharpe changes by "
        f"{', '.join(f'{pm(risk_d[t])} ({t})' for t in RISK_ASSETS)}. On SPY, Sharpe goes "
        f"{s('SPY', 'B&H'):.2f} → {s('SPY', 'B&H + VT'):.2f}, max drawdown {pct(spy_dd_bh)} → "
        f"{pct(spy_dd_vt)}, CAGR {pct(spy_cagr_bh, 1)} → {pct(spy_cagr_vt, 1)} (average "
        f"exposure {pct(spy_exp)}). A *constant* {pct(spy_exp)} position had a "
        f"{pct(const['max_drawdown'])} drawdown, so {dd_attrib}.",
        f"- **Bonds, gold, Bitcoin:** {', '.join(f'{pm(def_d[t])} ({t})' for t in DEFENSIVE)}"
        f" (average {pm(def_d.mean())} vs {pm(risk_d.mean())} for equities); {ief_note}. "
        f"Bitcoin: {pm(btc_d)}.",
        f"- **Significance:** Sharpe rises in {n_up} of {len(ASSETS_VT)} assets; "
        f"{n_sig or 'none'} of the buy-and-hold changes and {n_sig_ema or 'none'} of the EMA "
        f"ones are significant at 5% (one-sided paired stationary bootstrap, {n_boot:,} "
        "resamples, H0: no improvement).",
        f"- **Mechanism:** the leverage effect ρ(rₜ, Δσₜ) averages {pm(risk_lev)} for equity "
        f"ETFs vs {pm(def_lev)} for bonds/gold. Vol barely predicts next month's *mean* "
        f"(ρ = {pm(risk_rho)} vs {pm(def_rho)}), but next-month Sharpe is lower after "
        f"high-vol months in {n_drop_risk}/{len(RISK_ASSETS)} equity and "
        f"{n_drop_def}/{len(DEFENSIVE)} bond/gold ETFs.",
        f"- **On top of EMA50/200:** the Sharpe change averages {pm(ema_d.mean())} (range "
        f"{pm(ema_d.min())} to {pm(ema_d.max())}), {ema_vs_bh}.",
    ]
    if win_range is not None:
        findings.append(
            f"- **Robustness:** the mean equity gain is {pm(win_range[0])} to "
            f"{pm(win_range[1])} across 21/63/126-day windows; the 2x cap moves it by at most "
            f"{lev2_gain:.2f}; costs of daily rebalancing are included (SPY turnover "
            f"{spy_to:.1f}x a year)."
        )

    verdict = {
        "clean": "The data reproduce the Harvey et al. (2018) split cleanly",
        "weak": "The data reproduce the Harvey et al. (2018) split only weakly",
        "none": "The data do not reproduce the Harvey et al. (2018) split",
    }[split]
    mech_line = (
        "The mechanism is visible where it should be: equity returns fall as vol rises "
        "(the leverage effect) and vol clusters, so a trailing estimate is a usable forecast "
        "and cutting exposure in high-vol months removes risk that is not paid for with "
        "proportionally higher expected returns (Moreira & Muir 2017). Bonds and gold show no "
        "such asymmetry."
        if risk_lev < -0.2 and abs(def_lev) < 0.15
        else "The leverage-effect asymmetry between equities and bonds/gold is not clear in "
        "this sample, so there is little mechanism to point to."
    )
    sig_line = (
        "none of the seven Sharpe changes is statistically distinguishable from zero"
        if n_sig == 0
        else f"only {n_sig} of the seven Sharpe changes is distinguishable from zero"
    )
    text = f"""**Question.** Does scaling exposure to a constant volatility target improve risk-adjusted returns, and is the gain confined to risk assets as Harvey et al. (2018) report?

**Setup.** `VolTarget` overlay with a pre-specified {TARGET_VOL:.0%} annual target ({TARGET_VOL_CRYPTO:.0%} for Bitcoin, against its {s("BTC-USD", "B&H", "vol"):.0%} realised vol, so the overlay times BTC rather than just shrinking it), {WINDOW}-day realised vol, leverage capped at 1x (2x as a variant), daily rebalancing with costs, on buy-and-hold and EMA50/200; all variants of an asset share one window.

**Findings.**

{chr(10).join(findings)}

**Interpretation.** {verdict}. {mech_line} But the Sharpe effect is small and noisy at the single-asset level: {sig_line}{odd_line}. {shape_line} Vol targeting is a risk-management tool, not a source of alpha, and it does not rescue a single-asset trend rule."""

    after = (
        "The window-robustness table is a sensitivity check only; the 63-day window was fixed "
        "before running it. The 2x variant borrows at the T-bill rate, which understates the "
        "cost of leverage for a real account."
    )

    results = {
        "target_vol": TARGET_VOL,
        "target_vol_crypto": TARGET_VOL_CRYPTO,
        "spy_sharpe_bh": s("SPY", "B&H"),
        "spy_sharpe_bh_vt": s("SPY", "B&H + VT"),
        "spy_maxdd_bh": spy_dd_bh,
        "spy_maxdd_bh_vt": spy_dd_vt,
        "spy_sharpe_ema": s("SPY", "EMA50/200"),
        "spy_sharpe_ema_vt": s("SPY", "EMA + VT"),
        "mean_dsharpe_equities": float(risk_d.mean()),
        "mean_dsharpe_bonds_gold": float(def_d.mean()),
        "dsharpe_btc": btc_d,
        "n_assets_sharpe_up": n_up,
        "n_significant_bh_vt": n_sig,
        "n_significant_ema_vt": n_sig_ema,
        "leverage_effect_equities": float(risk_lev),
        "leverage_effect_bonds_gold": float(def_lev),
        "corr_vol_next_ret_equities": float(risk_rho),
        "corr_vol_next_ret_bonds_gold": float(def_rho),
        "harvey_split": split,
        "cross_asset_corr_pred_rho_vs_dsharpe": xc,
        "spy_maxdd_constant_exposure": float(const["max_drawdown"]),
        "spy_dd_reduction_timing_share": float(timing_share),
    }
    for tk in ASSETS_VT:
        results[f"dsharpe_bh_vt_{tk}"] = float(dsr.loc[tk, "B&H + VT (1x cap)"])
        results[f"dsharpe_ema_vt_{tk}"] = float(dsr.loc[tk, "EMA + VT (1x cap)"])

    return Section(
        name=NAME,
        title=TITLE,
        text=text,
        figures=[
            ("Change in Sharpe ratio from the vol-target overlay, by asset", fig_bars),
            ("SPY: trailing volatility and the overlay's exposure", fig_spy),
            ("Mechanism: vol-return predictability vs Sharpe gain", fig_scatter),
        ],
        tables=[
            ("Buy-and-hold and EMA50/200 with and without the 1x vol-target overlay", perf_disp),
            ("Mechanism and significance (monthly data, paired bootstrap)", mech_disp),
            ("Sharpe change of B&H + VT by vol-estimation window", win_disp),
        ],
        after=after,
        results=results,
    )

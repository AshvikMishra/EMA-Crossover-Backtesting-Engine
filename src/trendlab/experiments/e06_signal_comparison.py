"""Experiment 6: a horse race of trend signals -- and are they the same bet?

Seven textbook long/flat trend rules plus buy-and-hold, on SPY over its full
history and on each of the ten CORE ETFs, all evaluated on a common window per
asset (the latest warm-up among the rules) with the default research settings.

The "trend signature" section follows Levine & Pedersen (2016, "Which Trend Is
Your Friend?", *Financial Analysts Journal* 72(3)): every moving-average rule is
the sign of a weighted sum of past price changes. For an EMA with
``q = 1 - 2 / (span + 1)`` the weight of price level ``P_{t-k}`` is
``(1 - q) q^k``; writing ``P_{t-k} = P_t - sum_{j<k} dP_{t-j}`` and using that both
kernels sum to one gives

    EMA_fast(P)_t - EMA_slow(P)_t = sum_j c_j dP_{t-j},
    c_j = q_slow^(j+1) - q_fast^(j+1),

and in general ``c_j = sum_{k>j} (w_slow(k) - w_fast(k))`` for any pair of
price-averaging kernels. The identity is exact in dollar price changes (the
crossover's sign is reproduced bar for bar) and approximate in log returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import metrics, plotting, regimes, stats
from ..engine import BacktestResult, evaluation_start
from ..strategies import (
    BuyAndHold,
    DonchianBreakout,
    EMACrossover,
    FaberSMA,
    SMACrossover,
    Strategy,
    TimeSeriesMomentum,
)
from ..universe import CORE, CRISES
from .common import Context, Section

NAME = "e06_signal_comparison"
TITLE = "6. Signal horse race: EMA, SMA, Faber, TSMOM, Donchian -- one bet or many?"

BNH = "Buy & hold"
WHIPSAW_BARS = 63  # a losing round trip shorter than ~3 months counts as a whipsaw
EMA = "EMA50/200"


def signals() -> dict[str, Strategy]:
    """Display name -> rule; fixed textbook parameters, none tuned here."""
    return {
        EMA: EMACrossover(50, 200),
        "SMA50/200": SMACrossover(50, 200),
        "EMA50/200 ±2% band": EMACrossover(50, 200, band=0.02),
        "Faber 10M SMA": FaberSMA(10),
        "TSMOM 12M": TimeSeriesMomentum(252, 0, monthly=True),
        "TSMOM 12-1": TimeSeriesMomentum(252, 21, monthly=True),
        "Donchian 100/50": DonchianBreakout(100, 50),
        BNH: BuyAndHold(),
    }


SHORT = {
    EMA: "EMA",
    "SMA50/200": "SMA",
    "EMA50/200 ±2% band": "EMA band",
    "Faber 10M SMA": "Faber",
    "TSMOM 12M": "TSMOM 12",
    "TSMOM 12-1": "TSMOM 12-1",
    "Donchian 100/50": "Donchian",
}


# --------------------------------------------------------------------------- kernels


def trend_kernels(max_lag: int = 3000) -> pd.DataFrame:
    """Implied weights on past daily price changes (lag 0 = today), each scaled to sum to 1.

    Donchian is path-dependent (a max/min channel with memory), so it has no
    linear kernel and is omitted. Faber's monthly SMA is approximated by a daily
    210-bar SMA of the close compared with the close itself.
    """
    j = np.arange(max_lag, dtype=float)

    def ema_tail(span: int) -> np.ndarray:
        q = 1.0 - 2.0 / (span + 1.0)
        return q ** (j + 1.0)

    def sma_tail(n: int) -> np.ndarray:
        return np.clip((n - 1.0 - j) / n, 0.0, None)

    k = {
        EMA: ema_tail(200) - ema_tail(50),
        "SMA50/200": sma_tail(200) - sma_tail(50),
        "Faber 10M SMA": sma_tail(210),  # price itself has an all-zero tail
        "TSMOM 12M": (j < 252).astype(float),
        "TSMOM 12-1": ((j >= 21) & (j < 252)).astype(float),
    }
    df = pd.DataFrame(k, index=pd.Index(np.arange(max_lag), name="lag"))
    return df / df.sum()


def kernel_stats(k: pd.DataFrame) -> pd.DataFrame:
    lag = k.index.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "mean_lag": (k.mul(lag, axis=0)).sum(),
            "peak_lag": k.idxmax(),
            "weight_last_month": k.loc[:20].sum(),
            "weight_beyond_1y": k.loc[252:].sum(),
        }
    )


def kernel_check(prices: pd.DataFrame, strategy: Strategy, kernel: np.ndarray) -> dict:
    """Share of warmed-up bars where sign(sum c_j x_{t-j}) equals the rule's state."""
    target = strategy.target_position(prices)
    ok = target.notna().to_numpy()
    out = {}
    close = prices["Close"].astype(float)
    for label, x in {
        "dollar": close.diff().fillna(0.0).to_numpy(),
        "log": np.log(close).diff().fillna(0.0).to_numpy(),
    }.items():
        s = np.convolve(x, kernel)[: len(x)]
        out[label] = float(((s > 0).astype(float)[ok] == target.to_numpy()[ok]).mean())
    return out


# --------------------------------------------------------------------------- figures


def _kernel_figure(k: pd.DataFrame, ks: pd.DataFrame, path) -> None:
    with plotting.theme() as t:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4.8))
        names = list(k.columns)
        cmap = plotting.series_colors(names)
        show = k.loc[:400]
        for n in names:
            lw = 2.6 if n == EMA else 1.8
            ax.plot(
                show.index,
                show[n] * 100,
                color=cmap[n],
                lw=lw,
                label=f"{n} (mean lag {ks.loc[n, 'mean_lag']:.0f}d)",
                zorder=4 if n == EMA else 3,
            )
        ax.axvline(252, color=t["axis"], lw=1.0, ls="--", zorder=1)
        ax.annotate(
            "1 year",
            xy=(252, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(4, -4),
            textcoords="offset points",
            va="top",
            fontsize=8.5,
            color=t["ink2"],
        )
        ax.set_xlim(0, 400)
        ax.set_ylim(0, None)
        ax.set_xlabel("Lag of the daily price change (trading days)")
        ax.set_ylabel("Weight (% of total)")
        ax.legend(loc="upper right")
        e = ks.loc[EMA]
        plotting._titles(
            ax,
            f"EMA50/200 is a hump-shaped momentum kernel with a {e['mean_lag']:.0f}-day mean lag",
            f"Implied weights on past price changes (Levine & Pedersen 2016); "
            f"{e['weight_beyond_1y']:.0%} of the EMA weight lies beyond one year",
        )
        plotting.save_figure(fig, path)


def _median_sharpe_figure(sh: pd.DataFrame, path) -> None:
    """Bar = cross-asset median Sharpe per signal; dots = individual assets."""
    with plotting.theme() as t:
        import matplotlib.pyplot as plt

        med = sh.median().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(9, 5.2))
        y = np.arange(len(med))
        colors = [t["benchmark"] if n == BNH else t["categorical"][0] for n in med.index]
        ax.barh(y, med.to_numpy(), height=0.55, color=colors, zorder=2)
        rng = np.random.default_rng(0)
        for i, n in enumerate(med.index):
            vals = sh[n].to_numpy()
            jitter = rng.uniform(-0.18, 0.18, len(vals))
            ax.scatter(
                vals,
                i + jitter,
                s=18,
                color=t["ink2"],
                alpha=0.55,
                lw=0,
                zorder=3,
                label="Single ETF" if i == 0 else None,
            )
            ax.plot(
                [med[n]] * 2,
                [i - 0.3, i + 0.3],
                color=t["ink"],
                lw=2,
                zorder=4,
                label="Median" if i == 0 else None,
            )
        ax.axvline(0, color=t["axis"], lw=0.8)
        ax.set_yticks(y, [f"{n}  {med[n]:.2f}" for n in med.index])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.set_xlabel("Sharpe ratio (excess of T-bills)")
        ax.margins(x=0.1)
        fig.legend(loc="outside lower center", ncols=2)
        best = med.drop(BNH).idxmax()
        lead = med[BNH] >= med[best]
        plotting._titles(
            ax,
            "Across ten ETFs no trend rule beats buy & hold's median Sharpe"
            if lead
            else f"Across ten ETFs {best} has the highest median Sharpe",
            f"Bar: median over the {sh.shape[0]} CORE ETFs; dots: individual ETFs; "
            "common window per ETF, next-close fills, 5 bp/side",
        )
        plotting.save_figure(fig, path)


# --------------------------------------------------------------------------- run


def _run_asset(ctx: Context, ticker: str, sigs: dict[str, Strategy]) -> dict[str, BacktestResult]:
    start = evaluation_start(ctx.prices(ticker), list(sigs.values()), ctx.config(ticker).start)
    return {name: ctx.backtest(ticker, s, start=start) for name, s in sigs.items()}


def _agreement(e: pd.DataFrame) -> pd.DataFrame:
    """Share of bars on which each pair of rules holds the same exposure."""
    a = e.to_numpy()
    out = (a[:, :, None] == a[:, None, :]).mean(axis=0)
    return pd.DataFrame(out, index=e.columns, columns=e.columns)


def run(ctx: Context) -> Section:
    sigs = signals()
    trend_names = [n for n in sigs if n != BNH]
    tickers = ["SPY", "QQQ", "IEF", "GLD"] if ctx.quick else list(CORE)
    n_boot = 300 if ctx.quick else 2000

    per_asset: dict[str, dict[str, BacktestResult]] = {}
    for tk in tickers:
        per_asset[tk] = _run_asset(ctx, tk, sigs)
    spy = per_asset["SPY"]
    bnh = spy[BNH]

    # ---- SPY metrics table -------------------------------------------------
    summ, whip = {}, {}
    for n, r in spy.items():
        summ[n] = r.summary(None if n == BNH else bnh)
        whip[n] = regimes.whipsaw_stats(r.trades, WHIPSAW_BARS)
    raw = pd.DataFrame(summ).T
    w = pd.DataFrame(whip).T
    raw["n_losing"] = w["n_losing"]
    raw["n_whipsaws"] = w["n_whipsaws"]
    raw["whipsaw_loss"] = w["whipsaw_loss"]
    raw.index.name = "signal"
    boot = {}
    for n in trend_names:
        if n == EMA:
            continue
        b = stats.bootstrap_sharpe_difference(
            spy[n].returns, spy[EMA].returns, rf=spy[EMA].rf, n_boot=n_boot, seed=0
        )
        boot[n] = b
    raw["sharpe_diff_vs_ema"] = [boot[n]["diff"] if n in boot else np.nan for n in raw.index]
    raw["diff_ci_low"] = [boot[n]["ci_low"] if n in boot else np.nan for n in raw.index]
    raw["diff_ci_high"] = [boot[n]["ci_high"] if n in boot else np.nan for n in raw.index]
    ctx.save_table(f"{NAME}_spy_metrics", raw)

    keys = [
        "cagr",
        "vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "exposure",
        "beta",
        "n_trades",
        "trades_per_year",
    ]
    disp = metrics.format_table({n: summ[n] for n in sigs}, keys)
    disp = disp.rename(
        columns={
            "cagr": "CAGR",
            "vol": "Vol",
            "sharpe": "Sharpe",
            "sortino": "Sortino",
            "max_drawdown": "Max DD",
            "calmar": "Calmar",
            "exposure": "Exposure",
            "beta": "Beta",
            "n_trades": "Trades",
            "trades_per_year": "Trades / yr",
        }
    )
    disp["Losing trips"] = [f"{int(v)}" for v in raw["n_losing"]]
    disp["Whipsaws (< 3m)"] = [f"{int(v)}" for v in raw["n_whipsaws"]]
    disp["Whipsaw drag"] = [f"{v:.1%}" for v in raw["whipsaw_loss"]]
    disp["ΔSharpe vs EMA [95% CI]"] = [
        f"{boot[n]['diff']:+.2f} [{boot[n]['ci_low']:+.2f}, {boot[n]['ci_high']:+.2f}]"
        if n in boot
        else "–"
        for n in raw.index
    ]
    disp.index.name = "Signal (SPY)"

    # ---- cross-asset -------------------------------------------------------
    rows = []
    for tk, res in per_asset.items():
        b = res[BNH]
        for n, r in res.items():
            s = r.summary()
            ws = regimes.whipsaw_stats(r.trades, WHIPSAW_BARS)
            rows.append(
                {
                    "asset": tk,
                    "signal": n,
                    "start": s["start"],
                    "sharpe": s["sharpe"],
                    "cagr": s["cagr"],
                    "max_drawdown": s["max_drawdown"],
                    "exposure": s["exposure"],
                    "trades_per_year": s["trades_per_year"],
                    "n_whipsaws": ws["n_whipsaws"],
                    "sharpe_minus_bnh": s["sharpe"] - b.summary()["sharpe"] if n != BNH else 0.0,
                }
            )
    xa = pd.DataFrame(rows).set_index(["asset", "signal"])
    ctx.save_table(f"{NAME}_cross_asset", xa)
    sh = xa["sharpe"].unstack("signal")[list(sigs)]
    dd = xa["max_drawdown"].unstack("signal")[list(sigs)]
    cg = xa["cagr"].unstack("signal")[list(sigs)]
    tpy = xa["trades_per_year"].unstack("signal")[list(sigs)]
    beats = (sh.drop(columns=BNH).gt(sh[BNH], axis=0)).sum()

    # same-bet diagnostics per asset, then averaged
    ret_corrs, exp_corrs, agrees = [], [], []
    for res in per_asset.values():
        R = pd.DataFrame({n: r.returns for n, r in res.items()})
        E = pd.DataFrame({n: r.exposure for n, r in res.items() if n != BNH})
        ret_corrs.append(R.corr())
        exp_corrs.append(E.corr())
        agrees.append(_agreement(E))
    spy_ret_corr, spy_exp_corr, spy_agree = ret_corrs[0], exp_corrs[0], agrees[0]
    avg_exp_corr = sum(exp_corrs) / len(exp_corrs)
    avg_agree = sum(agrees) / len(agrees)

    xsum = pd.DataFrame(
        {
            "median_sharpe": sh.median(),
            "median_cagr": cg.median(),
            "median_max_drawdown": dd.median(),
            "median_trades_per_year": tpy.median(),
            "assets_beating_bnh_sharpe": beats.reindex(sh.columns),
            "avg_exposure_corr_with_ema": avg_exp_corr[EMA].reindex(sh.columns),
            "avg_agreement_with_ema": avg_agree[EMA].reindex(sh.columns),
        }
    )
    xsum.index.name = "signal"
    ctx.save_table(f"{NAME}_cross_asset_summary", xsum)
    xdisp = pd.DataFrame(index=xsum.index)
    xdisp["Median Sharpe"] = xsum["median_sharpe"].map("{:.2f}".format)
    xdisp["Median CAGR"] = xsum["median_cagr"].map("{:.1%}".format)
    xdisp["Median max DD"] = xsum["median_max_drawdown"].map("{:.1%}".format)
    xdisp["Median trades / yr"] = xsum["median_trades_per_year"].map("{:.1f}".format)
    xdisp[f"ETFs with Sharpe > B&H (of {len(tickers)})"] = [
        "–" if n == BNH else f"{int(v)}" for n, v in xsum["assets_beating_bnh_sharpe"].items()
    ]
    xdisp["Exposure corr. with EMA (avg)"] = [
        "–" if pd.isna(v) else f"{v:.2f}" for v in xsum["avg_exposure_corr_with_ema"]
    ]
    xdisp["Days in same state as EMA (avg)"] = [
        "–" if pd.isna(v) else f"{v:.0%}" for v in xsum["avg_agreement_with_ema"]
    ]
    xdisp.index.name = "Signal"

    corr_tbl = spy_ret_corr.rename(index=SHORT, columns=SHORT)
    expc_tbl = spy_exp_corr.rename(index=SHORT, columns=SHORT)
    ctx.save_table(f"{NAME}_spy_return_corr", corr_tbl)
    ctx.save_table(f"{NAME}_spy_exposure_corr", expc_tbl)
    ctx.save_table(f"{NAME}_spy_agreement", spy_agree.rename(index=SHORT, columns=SHORT))
    ctx.save_table(f"{NAME}_avg_exposure_corr", avg_exp_corr.rename(index=SHORT, columns=SHORT))
    corr_disp = pd.DataFrame(
        {
            c: [
                f"{spy_ret_corr.loc[r, c]:.2f} / "
                + ("–" if r == BNH or c == BNH else f"{spy_exp_corr.loc[r, c]:.2f}")
                for r in spy_ret_corr.index
            ]
            for c in spy_ret_corr.columns
        },
        index=spy_ret_corr.index,
    ).rename(index=SHORT, columns=SHORT)
    corr_disp = corr_disp.rename(index={BNH: "B&H"}, columns={BNH: "B&H"})
    corr_disp.index.name = "SPY: returns / exposure"

    # ---- trend signature ---------------------------------------------------
    k = trend_kernels()
    ks = kernel_stats(k)
    ctx.save_table(f"{NAME}_kernel_stats", ks)
    ks_disp = pd.DataFrame(
        {
            "Mean lag (days)": ks["mean_lag"].map("{:.0f}".format),
            "Peak lag (days)": ks["peak_lag"].map(lambda v: f"{int(v)}"),
            "Weight on last month": ks["weight_last_month"].map("{:.0%}".format),
            "Weight beyond 1 year": ks["weight_beyond_1y"].map("{:.0%}".format),
        }
    )
    ks_disp.index.name = "Signal kernel"
    check = kernel_check(ctx.prices("SPY"), sigs[EMA], k[EMA].to_numpy())

    # ---- figures -------------------------------------------------------------
    fin = pd.Series({n: float((1.0 + r.returns).prod()) for n, r in spy.items()})
    best_spy = raw.loc[trend_names, "sharpe"].idxmax()
    spread = float(raw.loc[trend_names, "sharpe"].max() - raw.loc[trend_names, "sharpe"].min())
    plotting.equity_curves(
        {n: r.returns for n, r in spy.items()},
        log=True,
        title=(
            f"Buy & hold ends richest ({fin[BNH]:.0f}×); EMA50/200 ends at {fin[EMA]:.0f}×"
            if fin.idxmax() == BNH
            else f"{fin.idxmax()} ends richest ({fin.max():.0f}×), buy & hold {fin[BNH]:.0f}×, "
            f"EMA50/200 {fin[EMA]:.0f}×"
        ),
        subtitle=f"Growth of $1 from {bnh.start.date()}, next-close fills, 5 bp/side, "
        "T-bills when out",
        highlight=[EMA] + ([fin.idxmax()] if fin.idxmax() not in (EMA, BNH) else []),
        path=ctx.fig_path(f"{NAME}_equity"),
    )
    exp_off = spy_exp_corr.to_numpy()[~np.eye(len(spy_exp_corr), dtype=bool)]
    med_corr = float(np.median(exp_off))
    heat = spy_exp_corr.rename(index=SHORT, columns=SHORT)
    heat.index.name, heat.columns.name = "", ""
    tri = np.tril(np.ones(heat.shape, dtype=bool))
    lo_corr = float(heat.where(~np.eye(len(heat), dtype=bool)).min().min())
    fig = plotting.metric_heatmap(
        heat.where(tri),
        "Mostly the same bet: exposure correlations on SPY"
        if med_corr > 0.6
        else "Different bets: exposure correlations on SPY",
        fmt="{:.2f}",
        mark=None,
        subtitle=f"Correlation of daily 0/1 exposures, {bnh.start.year}-{bnh.end.year}; "
        f"median pair {med_corr:.2f}, lowest {lo_corr:.2f}",
        cbar_label="Correlation",
        xlabel=" ",
        ylabel=" ",
    )
    ax = fig.axes[0]
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")
    plotting.save_figure(fig, ctx.fig_path(f"{NAME}_exposure_corr"))
    _median_sharpe_figure(sh, ctx.fig_path(f"{NAME}_median_sharpe"))
    _kernel_figure(k, ks, ctx.fig_path(f"{NAME}_trend_signature"))

    # ---- text ----------------------------------------------------------------
    sp = raw["sharpe"]
    md = raw["max_drawdown"]
    n_sig = sum(1 for n in boot if boot[n]["ci_low"] > 0 or boot[n]["ci_high"] < 0)
    ci_w = np.median([boot[n]["ci_high"] - boot[n]["ci_low"] for n in boot])
    med_sh = xsum["median_sharpe"]
    best_x = med_sh.drop(BNH).idxmax()
    worst_x = med_sh.drop(BNH).idxmin()
    ma_like = [n for n in trend_names if n != "Donchian 100/50"]
    ma_ret = spy_ret_corr.loc[ma_like, ma_like].to_numpy()[~np.eye(len(ma_like), dtype=bool)]
    don_ret = spy_ret_corr.loc["Donchian 100/50", ma_like]
    ma_exp = spy_exp_corr.loc[ma_like, ma_like].to_numpy()[~np.eye(len(ma_like), dtype=bool)]
    band_tr = raw.loc["EMA50/200 ±2% band", "n_trades"]
    ema_tr = raw.loc[EMA, "n_trades"]
    least_trades = raw.loc[trend_names, "trades_per_year"].idxmin()
    most_whip = raw.loc[trend_names, "n_whipsaws"].astype(float).idxmax()
    trade_bit = f"the ±2% band cuts SPY round trips from {ema_tr:.0f} to {band_tr:.0f}" + (
        "" if least_trades == "EMA50/200 ±2% band" else f"; {least_trades} trades least"
    )
    noise_head = "Noise, not signal choice" if n_sig == 0 else "Some differences are significant"
    e_ks, t_ks = ks.loc[EMA], ks.loc["TSMOM 12M"]
    beats_ema = int(beats[EMA])
    ema_rank = int(sp[trend_names].rank(ascending=False)[EMA])
    cv0, cv1 = (pd.Timestamp(d) for d in CRISES["COVID crash"])
    dd_trough = {n: metrics.drawdown_series(spy[n].returns).idxmin() for n in trend_names}
    covid_dd = [n for n, d in dd_trough.items() if cv0 <= d <= cv1]
    covid_full = [n for n in covid_dd if float(spy[n].exposure.loc[cv0:cv1].min()) > 0.999]
    covid_bit = (
        f" {len(covid_dd)} of the {len(trend_names)} rules had their worst drawdown in the "
        "COVID crash"
        + (", which they sat through fully invested." if len(covid_full) == len(covid_dd) else ".")
        if len(covid_dd) >= 2
        else ""
    )
    ci_cmp = "wider than" if ci_w > spread else "narrower than"
    lag_gap = abs(e_ks["mean_lag"] - t_ks["mean_lag"])
    rho_ts = float(spy_exp_corr.loc[EMA, "TSMOM 12M"])

    text = f"""**Question.** Is EMA50/200 a special choice, or one member of a family of trend rules that all make the same bet?

**Setup.** Seven textbook long/flat rules with fixed, untuned parameters (EMA and SMA 50/200, EMA with a ±2% band, Faber's 10-month SMA, 12-month and 12-1 TSMOM, Donchian 100/50) and buy & hold, on SPY from {bnh.start.date()} and on {len(tickers)} CORE ETFs, each on a common window after the slowest warm-up. Sharpe differences use a paired stationary bootstrap ({n_boot} draws, 20-day blocks).

**Findings.**
- **SPY:** Sharpe ranges from {sp[trend_names].min():.2f} to {sp[trend_names].max():.2f} across the rules (best {best_spy}; EMA50/200 {sp[EMA]:.2f}, rank {ema_rank} of {len(trend_names)}) vs {sp[BNH]:.2f} for buy & hold; max drawdowns {md[trend_names].max():.0%} to {md[trend_names].min():.0%} vs {md[BNH]:.0%}.{covid_bit}
- **{noise_head}:** {n_sig} of {len(boot)} Sharpe differences vs EMA50/200 have a 95% CI excluding zero; the median CI is {ci_w:.2f} wide, {ci_cmp} the whole {spread:.2f} spread between the rules.
- **Across ETFs:** median Sharpe is {med_sh[BNH]:.2f} for buy & hold vs {med_sh[worst_x]:.2f} ({worst_x}) to {med_sh[best_x]:.2f} ({best_x}) for the rules; EMA50/200 beats buy & hold's Sharpe in {beats_ema} of {len(tickers)} ETFs. Median max drawdown falls from {xsum.loc[BNH, "median_max_drawdown"]:.0%} to between {xsum.loc[trend_names, "median_max_drawdown"].max():.0%} and {xsum.loc[trend_names, "median_max_drawdown"].min():.0%}.
- **Same bet:** on SPY the MA and TSMOM rules' daily returns correlate {ma_ret.min():.2f}-{ma_ret.max():.2f} and their 0/1 exposures {ma_exp.min():.2f}-{ma_exp.max():.2f}; across ETFs the average rule is in the same state as EMA50/200 on {avg_agree[EMA].drop(EMA).mean():.0%} of days. Donchian is the outlier (return correlation {don_ret.min():.2f}-{don_ret.max():.2f}).
- **Trading:** {trade_bit}; {most_whip} has the most whipsaws (losing trips under 3 months: {int(raw.loc[most_whip, "n_whipsaws"])}, compounding to {raw.loc[most_whip, "whipsaw_loss"]:.0%}) vs {int(raw.loc[EMA, "n_whipsaws"])} for EMA50/200.
- **Trend signature:** EMA50/200 equals the sign of a weighted sum of past price changes, with weights peaking at lag {e_ks["peak_lag"]:.0f}, a mean lag of {e_ks["mean_lag"]:.0f} days (TSMOM 12M: {t_ks["mean_lag"]:.0f}) and {e_ks["weight_beyond_1y"]:.0%} of its weight beyond a year. The kernel reproduces the rule's state on {check["dollar"]:.1%} of SPY days ({check["log"]:.1%} with log returns).
"""
    noise = n_sig == 0 and ci_w > spread
    snoop = (
        "With highly correlated candidates and confidence intervals wider than their "
        'differences, picking "the best" rule from this table would be data snooping in the '
        "sense of Sullivan, Timmermann & White (1999): this is one trend bet, and the choice "
        "among its variants is mostly about turnover and whipsaw tolerance, not expected return."
        if noise and med_corr > 0.6
        else "Some variants differ measurably, but with this many related candidates any "
        "winner should be judged against a data-snooping benchmark (Sullivan, Timmermann & "
        "White 1999)."
    )
    bnh_best = med_sh[BNH] >= med_sh.drop(BNH).max()
    xasset_line = (
        f"Across the {len(tickers)} ETFs the family's value is risk reduction (median max "
        f"drawdown {xsum.loc[EMA, 'median_max_drawdown']:.0%} for EMA50/200 vs "
        f"{xsum.loc[BNH, 'median_max_drawdown']:.0%}), "
        + (
            "not a higher Sharpe: no variant beats buy & hold's median."
            if bnh_best
            else f"with {best_x} edging buy & hold's median Sharpe."
        )
    )
    text += f"""
**Interpretation.** Levine & Pedersen (2016) show that moving-average crossovers and time-series momentum (Moskowitz, Ooi & Pedersen 2012) are both linear filters on past returns and are closely related once their horizons are matched; they differ mainly in the shape of the weights. The analytic kernel makes this concrete: EMA50/200 has {"almost the same mean lag as" if lag_gap < 10 else "a different mean lag from"} 12-month TSMOM ({e_ks["mean_lag"]:.0f} vs {t_ks["mean_lag"]:.0f} days), but its weights are hump-shaped (peak at lag {e_ks["peak_lag"]:.0f}) with a long tail instead of a flat 12-month box{"; the shared horizon is why their SPY exposures correlate " + f"{rho_ts:.2f}" if lag_gap < 10 else ""}. Faber's (2007) 10-month rule weights recent changes most (mean lag {ks.loc["Faber 10M SMA", "mean_lag"]:.0f} days) and, being checked only at month ends, trades {"more" if raw.loc["Faber 10M SMA", "n_trades"] > ema_tr else "less"} often than EMA50/200 on SPY ({raw.loc["Faber 10M SMA", "n_trades"]:.0f} vs {ema_tr:.0f} round trips). Zakamulin (2014) likewise finds the real-life performance of MA and momentum timing rules far more modest than commonly reported. {snoop} {xasset_line}
"""
    after = (
        "Correlation table: daily strategy-return correlation / daily exposure correlation "
        "(buy & hold has constant exposure, so its exposure correlation is undefined). Kernel "
        "weights are on daily price changes and scaled to sum to one; Faber's monthly rule is "
        "approximated by a daily 210-bar SMA, and Donchian, being path-dependent, has no "
        "linear kernel."
    )

    results = {
        "spy_start": bnh.start.date().isoformat(),
        "spy_sharpe_bnh": float(sp[BNH]),
        "spy_sharpe_spread": spread,
        "spy_best_signal": best_spy,
        "n_significant_vs_ema": int(n_sig),
        "median_ci_width": float(ci_w),
        "xasset_median_sharpe_bnh": float(med_sh[BNH]),
        "xasset_best_signal": best_x,
        "xasset_best_median_sharpe": float(med_sh[best_x]),
        "ema_beats_bnh_assets": beats_ema,
        "n_assets": len(tickers),
        "spy_ma_return_corr_min": float(ma_ret.min()),
        "spy_ma_return_corr_max": float(ma_ret.max()),
        "spy_exposure_corr_min": float(exp_off.min()),
        "spy_ma_exposure_corr_min": float(ma_exp.min()),
        "avg_agreement_with_ema": float(avg_agree[EMA].drop(EMA).mean()),
        "ema_kernel_mean_lag": float(e_ks["mean_lag"]),
        "ema_kernel_peak_lag": float(e_ks["peak_lag"]),
        "ema_kernel_weight_beyond_1y": float(e_ks["weight_beyond_1y"]),
        "tsmom12_kernel_mean_lag": float(t_ks["mean_lag"]),
        "ema_kernel_sign_match_dollar": check["dollar"],
        "ema_kernel_sign_match_log": check["log"],
    }
    for n in sigs:
        key = SHORT.get(n, "bnh").lower().replace(" ", "_").replace("-", "_")
        results[f"spy_sharpe_{key}"] = float(sp[n])
        results[f"spy_maxdd_{key}"] = float(md[n])
        results[f"xasset_median_sharpe_{key}"] = float(med_sh[n])

    figures = [
        ("SPY growth of $1 for every signal (log scale)", ctx.fig_path(f"{NAME}_equity")),
        ("Correlation of daily exposures on SPY", ctx.fig_path(f"{NAME}_exposure_corr")),
        ("Cross-asset median Sharpe per signal", ctx.fig_path(f"{NAME}_median_sharpe")),
        (
            "Trend signature: implied weights on past price changes",
            ctx.fig_path(f"{NAME}_trend_signature"),
        ),
    ]
    tables = [
        ("SPY: metrics per signal", disp),
        (f"Across {len(tickers)} CORE ETFs", xdisp),
        ("SPY correlation of daily returns / exposures", corr_disp),
        ("Trend signatures", ks_disp),
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

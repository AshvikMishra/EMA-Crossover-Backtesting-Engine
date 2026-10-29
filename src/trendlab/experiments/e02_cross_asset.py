"""Experiment 2: the same EMA50/200 rule on every asset in the fixed universe.

The original notebook reported one hindsight-picked stock over three years.
Here the identical rule (no re-tuning) runs on one liquid ETF per asset class,
plus Bitcoin and NVDA, over each asset's full history with the default
research settings (next-close fills, 5/15 bp per side, T-bill cash). Each
asset is compared with buy-and-hold over the same window and with an
*exposure-matched static mix* that holds the rule's average exposure in the
asset and the rest in T-bills.

The static mix's excess return is ``e * (asset - rf)``, so its Sharpe ratio
equals buy-and-hold's by construction (Sharpe is leverage invariant). It is
the fair benchmark for *CAGR and drawdown*: a rule that is out of the market
part of the time mechanically has a shallower drawdown, and the question is
whether timing adds anything beyond holding less of the asset.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator

from .. import metrics, plotting
from ..plotting import _date_axis, _growth_axis, _minus, _pct_axis, _titles, save_figure, theme
from ..regimes import capture_ratios, crisis_table
from ..report import df_to_markdown
from ..stats import bootstrap_sharpe_difference
from ..strategies import EMACrossover
from ..universe import ASSETS, CORE
from .common import Context, Section, costs_for

NAME = "e02_cross_asset"
TITLE = "2. One rule, twelve markets: EMA50/200 across asset classes"

STRATEGY = EMACrossover(50, 200)
STRAT_LABEL = "EMA50/200"
MIX_LABEL = "Exposure-matched mix"
BNH_LABEL = "Buy & hold"
BOND_PEAK = "2020-07-31"  # long Treasuries peaked in early August 2020
SLOW_BEARS = ("Dot-com bust", "Global financial crisis")
FAST_BEAR = "COVID crash"
N_BOOT, N_BOOT_QUICK = 2000, 200  # paired stationary bootstrap of the Sharpe difference


def _mix_returns(res) -> tuple[pd.Series, float]:
    """Static mix: mean exposure in the asset, the rest in T-bills (daily rebalanced, no costs)."""
    e = float(res.exposure.mean())
    r = e * res.asset_returns + (1.0 - e) * res.rf
    r.iloc[0] = 0.0
    return r.rename(MIX_LABEL), e


def _asset_row(ctx: Context, ticker: str) -> tuple[dict, dict]:
    s, b = ctx.pair(ticker, STRATEGY)
    p = s.periods_per_year
    ss, bs = s.summary(b), b.summary()
    mix, e = _mix_returns(s)
    cap = capture_ratios(s.returns, b.returns, "ME")
    boot = bootstrap_sharpe_difference(
        s.returns,
        b.returns,
        rf=s.rf,
        n_boot=N_BOOT_QUICK if ctx.quick else N_BOOT,
        periods=p,
    )
    row = {
        "class": ASSETS[ticker]["class"],
        "start": s.start.date().isoformat(),
        "years": ss["years"],
        "cagr": ss["cagr"],
        "vol": ss["vol"],
        "sharpe": ss["sharpe"],
        "max_drawdown": ss["max_drawdown"],
        "calmar": ss["calmar"],
        "bnh_cagr": bs["cagr"],
        "bnh_vol": bs["vol"],
        "bnh_sharpe": bs["sharpe"],
        "bnh_max_drawdown": bs["max_drawdown"],
        "exposure": e,
        "trades_per_year": ss["trades_per_year"],
        "hit_rate": ss["hit_rate"],
        "mix_cagr": metrics.cagr(mix, p),
        "mix_vol": metrics.ann_vol(mix, p),
        "mix_sharpe": metrics.sharpe(mix, s.rf, p),
        "mix_max_drawdown": metrics.max_drawdown(mix),
        "beta": ss["beta"],
        "alpha": ss["alpha"],
        "alpha_t": ss["alpha_t"],
        "upside_capture": cap["upside_capture"],
        "downside_capture": cap["downside_capture"],
        "sharpe_diff_ci_low": boot["ci_low"],
        "sharpe_diff_ci_high": boot["ci_high"],
    }
    return row, {"res": s, "bnh": b}


def _display(df: pd.DataFrame) -> pd.DataFrame:
    def pct(v, d=1):
        return "–" if pd.isna(v) else f"{v:.{d}%}"

    def num(v):
        return "–" if pd.isna(v) else f"{v:.2f}"

    out = pd.DataFrame(index=df.index)
    out.index.name = "Asset"
    out["Class"] = df["class"]
    out["Start"] = df["start"]
    out["Years"] = df["years"].map(lambda v: f"{v:.1f}")
    out["CAGR"] = df["cagr"].map(pct)
    out["Vol"] = df["vol"].map(pct)
    out["Sharpe"] = df["sharpe"].map(num)
    out["MaxDD"] = df["max_drawdown"].map(pct)
    out["Calmar"] = df["calmar"].map(num)
    out["B&H CAGR"] = df["bnh_cagr"].map(pct)
    out["B&H Sharpe"] = df["bnh_sharpe"].map(num)
    out["B&H MaxDD"] = df["bnh_max_drawdown"].map(pct)
    out["Exposure"] = df["exposure"].map(lambda v: pct(v, 0))
    out["Trades/yr"] = df["trades_per_year"].map(num)
    out["Hit rate"] = df["hit_rate"].map(lambda v: pct(v, 0))
    out["Matched CAGR"] = df["mix_cagr"].map(pct)
    out["Matched MaxDD"] = df["mix_max_drawdown"].map(pct)
    out["Matched Sharpe"] = df["mix_sharpe"].map(num)
    out["Sharpe gap 95% CI"] = [
        _minus(f"{lo:+.2f} to {hi:+.2f}")
        for lo, hi in zip(df["sharpe_diff_ci_low"], df["sharpe_diff_ci_high"], strict=True)
    ]
    return out


def _save(ctx: Context, name: str, raw: pd.DataFrame, display: pd.DataFrame) -> None:
    """Raw numbers to CSV; the formatted copy replaces the Markdown version."""
    ctx.save_table(name, raw)
    (ctx.table_dir / f"{name}.md").write_text(df_to_markdown(display) + "\n")


def _small_multiples(curves: dict, df: pd.DataFrame, path) -> None:
    """Grid (4 per row) of growth-of-$1 (log) curves, strategy vs buy-and-hold, per asset."""
    with theme() as t:
        n = len(curves)
        ncols = 4
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.1 * nrows + 1.2))
        axes = np.atleast_1d(axes).ravel()
        for ax, (ticker, c) in zip(axes, curves.items(), strict=False):
            bnh = c["bnh"].equity / c["bnh"].equity.iloc[0]
            strat = c["res"].equity / c["res"].equity.iloc[0]
            ax.plot(bnh.index, bnh.to_numpy(), color=plotting.BENCHMARK_COLOR, lw=1.3)
            ax.plot(strat.index, strat.to_numpy(), color=plotting.STRATEGY_COLOR, lw=1.5)
            ax.axhline(1.0, color=t["axis"], lw=0.8, zorder=1)
            ax.set_yscale("log")
            _growth_axis(ax)
            lo, hi = ax.get_ylim()
            ticks = [v for v in ax.yaxis.get_major_locator().tick_values(lo, hi) if lo <= v <= hi]
            if len(ticks) < 4:  # a 1-2-5 log ladder can leave too few labels
                ax.yaxis.set_major_locator(LogLocator(10, (1.0, 1.5, 2.0, 3.0, 5.0, 7.0)))
            _date_axis(ax)
            ax.tick_params(axis="both", labelsize=8)
            r = df.loc[ticker]
            ax.set_title(
                f"{ticker} · {ASSETS[ticker]['name']}",
                loc="left",
                fontsize=10,
                fontweight="bold",
                color=t["ink"],
                pad=16,
            )
            ax.annotate(
                _minus(
                    f"Rule vs B&H: Sharpe {r['sharpe']:.2f} vs {r['bnh_sharpe']:.2f} · "
                    f"MaxDD {r['max_drawdown']:.0%} vs {r['bnh_max_drawdown']:.0%}"
                ),
                xy=(0, 1),
                xycoords="axes fraction",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=8,
                color=t["ink2"],
                va="bottom",
            )
        for ax in axes[n:]:
            ax.set_visible(False)
        handles = [
            plt.Line2D([], [], color=plotting.STRATEGY_COLOR, lw=2, label=STRAT_LABEL),
            plt.Line2D([], [], color=plotting.BENCHMARK_COLOR, lw=2, label=BNH_LABEL),
        ]
        fig.legend(handles=handles, loc="outside lower center", ncols=2)
        fig.suptitle(
            "Growth of $1 (log scale): EMA50/200 vs buy & hold in every market",
            x=0.01,
            ha="left",
            fontsize=13,
            fontweight="bold",
            color=t["ink"],
        )
        save_figure(fig, path)


def _tradeoff_scatter(df: pd.DataFrame, title: str, path) -> None:
    """CAGR give-up (x) against drawdown reduction (y), one labelled dot per asset."""
    with theme() as t:
        fig, ax = plt.subplots(figsize=(9, 5.8))
        x = (df["cagr"] - df["bnh_cagr"]).to_numpy(dtype=float)
        y = (df["max_drawdown"] - df["bnh_max_drawdown"]).to_numpy(dtype=float)
        classes = list(dict.fromkeys(df["class"]))
        cmap = plotting.series_colors(classes)
        ax.axhline(0.0, color=t["axis"], lw=0.8, zorder=1)
        ax.axvline(0.0, color=t["axis"], lw=0.8, zorder=1)
        for cls in classes:
            m = (df["class"] == cls).to_numpy()
            ax.scatter(
                x[m],
                y[m],
                s=64,
                color=cmap[cls],
                edgecolor=t["surface"],
                linewidths=1.5,
                zorder=3,
                label=cls,
            )
        for xi, yi, name in zip(x, y, df.index, strict=False):
            ax.annotate(
                name,
                xy=(xi, yi),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=8.5,
                color=t["ink2"],
            )
        ax.grid(axis="both")
        _pct_axis(ax.xaxis)
        _pct_axis(ax.yaxis)
        ax.margins(0.15)
        ax.set_xlabel("CAGR, rule minus buy & hold (pp per year)")
        ax.set_ylabel("Max drawdown, rule minus buy & hold (pp; > 0 = shallower)")
        fig.legend(loc="outside right upper")
        _titles(
            ax,
            title,
            "One dot per asset, full history, next-close fills, costs included",
        )
        save_figure(fig, path)


def _bond_split(s, b) -> dict:
    p = s.periods_per_year
    out = {}
    peak = pd.Timestamp(BOND_PEAK)
    for label, m in (("pre", s.returns.index <= peak), ("post", s.returns.index > peak)):
        rs, rb, rf = s.returns[m], b.returns[m], s.rf[m]
        out[label] = {
            "sharpe": metrics.sharpe(rs, rf, p),
            "bnh_sharpe": metrics.sharpe(rb, rf, p),
            "cagr": metrics.cagr(rs, p),
            "bnh_cagr": metrics.cagr(rb, p),
            "max_drawdown": metrics.max_drawdown(rs),
            "bnh_max_drawdown": metrics.max_drawdown(rb),
        }
    return out


def _join(items: list[str]) -> str:
    if not items:
        return "none"
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def run(ctx: Context) -> Section:
    tickers = list(ASSETS)
    rows, curves = {}, {}
    for t in tickers:
        rows[t], curves[t] = _asset_row(ctx, t)
    df = pd.DataFrame(rows).T
    num_cols = [c for c in df.columns if c not in ("class", "start")]
    df[num_cols] = df[num_cols].astype(float)
    df.index.name = "ticker"

    display = _display(df)
    _save(ctx, f"{NAME}_assets", df, display)

    # ---- headline statistics
    n = len(df)
    sharpe_diff = df["sharpe"] - df["bnh_sharpe"]
    cagr_diff = df["cagr"] - df["bnh_cagr"]
    dd_diff = df["max_drawdown"] - df["bnh_max_drawdown"]
    sharpe_wins = int((sharpe_diff > 0).sum())
    cagr_wins = int((cagr_diff > 0).sum())
    dd_wins = int((dd_diff > 0).sum())
    calmar_wins = int((df["calmar"] > df["bnh_max_drawdown"].abs().rdiv(df["bnh_cagr"])).sum())
    mix_cagr_wins = int((df["cagr"] > df["mix_cagr"]).sum())
    mix_dd_wins = int((df["max_drawdown"] > df["mix_max_drawdown"]).sum())
    mix_sharpe_gap = float((df["mix_sharpe"] - df["bnh_sharpe"]).abs().max())
    med = {
        "sharpe_diff": float(sharpe_diff.median()),
        "cagr_diff": float(cagr_diff.median()),
        "dd_diff": float(dd_diff.median()),
        "mix_cagr_diff": float((df["cagr"] - df["mix_cagr"]).median()),
        "mix_dd_diff": float((df["max_drawdown"] - df["mix_max_drawdown"]).median()),
        "exposure": float(df["exposure"].median()),
        "trades": float(df["trades_per_year"].median()),
        "hit": float(df["hit_rate"].median()),
        "up": float(df["upside_capture"].median()),
        "down": float(df["downside_capture"].median()),
        "beta": float(df["beta"].median()),
    }
    ci_pos = int((df["sharpe_diff_ci_low"] > 0).sum())
    ci_neg = int((df["sharpe_diff_ci_high"] < 0).sum())
    ci_halfwidth = float(((df["sharpe_diff_ci_high"] - df["sharpe_diff_ci_low"]) / 2).median())
    core = [t for t in CORE if t in df.index]
    core_sharpe_wins = int((sharpe_diff[core] > 0).sum())
    core_cagr_wins = int((cagr_diff[core] > 0).sum())
    picks = [t for t in df.index if t not in core]
    picks_sharpe_wins = int((sharpe_diff[picks] > 0).sum())
    by_class = sharpe_diff.groupby(df["class"], sort=False).mean()
    class_n = df["class"].value_counts()

    def cls_label(c, v):
        return f"{c} ({v:+.2f}{', one asset' if class_n[c] == 1 else ''})"

    pos_cls = [cls_label(c, v) for c, v in by_class.items() if v > 0]
    neg_cls = [cls_label(c, v) for c, v in by_class.items() if v <= 0]

    # crisis behaviour of SPY: slow bears vs a V-shaped crash
    spy = crisis_table(curves["SPY"]["res"], curves["SPY"]["bnh"])
    slow_prot = [float(spy.loc[c, "protection"]) for c in SLOW_BEARS if c in spy.index]
    fast = spy.loc[FAST_BEAR] if FAST_BEAR in spy.index else None

    bonds = {t: _bond_split(curves[t]["res"], curves[t]["bnh"]) for t in ("TLT", "IEF")}
    tlt_pre, tlt_post = bonds["TLT"]["pre"], bonds["TLT"]["post"]

    # ---- figures
    sharpe_df = df[["sharpe", "bnh_sharpe"]].rename(
        columns={"sharpe": STRAT_LABEL, "bnh_sharpe": BNH_LABEL}
    )
    plotting.bar_comparison(
        sharpe_df,
        f"Sharpe: the rule beats buy & hold in {sharpe_wins} of {n} markets; "
        + _minus(f"median gap {med['sharpe_diff']:+.3f}"),
        subtitle=f"Full history per asset, excess over T-bills, next-close fills, costs included; "
        f"the 95% bootstrap CI of the gap excludes zero in {ci_pos + ci_neg} of {n}",
        percent=False,
        xlabel="Sharpe ratio",
        path=ctx.fig_path(f"{NAME}_sharpe"),
    )
    dd_df = df[["max_drawdown", "mix_max_drawdown", "bnh_max_drawdown"]].rename(
        columns={
            "max_drawdown": STRAT_LABEL,
            "mix_max_drawdown": MIX_LABEL,
            "bnh_max_drawdown": BNH_LABEL,
        }
    )
    plotting.bar_comparison(
        dd_df,
        f"Drawdowns: shallower than buy & hold in {dd_wins} of {n}, "
        f"than the matched mix in {mix_dd_wins} of {n}",
        subtitle="Worst peak-to-trough loss; matched mix = rule's average exposure in the "
        "asset, rest in T-bills",
        percent=True,
        fmt=lambda v: f"{v:.0%}",
        xlabel="Maximum drawdown",
        colors={MIX_LABEL: plotting.CATEGORICAL[1]},
        path=ctx.fig_path(f"{NAME}_maxdd"),
    )
    dd_word = "shallower" if med["dd_diff"] > 0 else "deeper"
    cagr_word = "less" if med["cagr_diff"] < 0 else "more"
    _tradeoff_scatter(
        df,
        f"Median trade-off: {abs(med['dd_diff']) * 100:.0f} pp {dd_word} drawdown for "
        f"{abs(med['cagr_diff']) * 100:.1f} pp {cagr_word} CAGR a year",
        ctx.fig_path(f"{NAME}_tradeoff"),
    )
    _small_multiples(curves, df, ctx.fig_path(f"{NAME}_equity_grid"))

    # ---- text
    def pick(cond: bool, yes: str, no: str) -> str:
        return yes if cond else no

    if med["hit"] < 0.4:
        hit_txt = "so most round trips are small whipsaw losses and a few long trends pay for them"
    elif med["hit"] < 0.6:
        hit_txt = (
            "so roughly half the round trips are whipsaw losses and the long trends pay for them"
        )
    else:
        hit_txt = (
            "so the typical round trip is a winner; the losers are whipsaws around the 200-day"
        )
    cap_txt = pick(
        med["up"] > med["down"] + 0.05,
        "the rule keeps noticeably more of the up months than of the down months",
        "up and down months are cut by similar amounts: at a monthly horizon the rule "
        "looks like a smaller position, and the option-like payoff only appears over "
        "multi-month bear markets",
    )
    slow_txt = " and ".join(f"{p * 100:+.0f}" for p in slow_prot) + " percentage points"
    if fast is not None:
        late = pd.notna(fast["exit_date"]) and fast["exit_date"] >= fast["trough"]
        fast_txt = (
            f"in the {FAST_BEAR} it lost {-fast['strat_return']:.0%} vs "
            f"{-fast['bench_return']:.0%} for buy-and-hold"
            + pick(
                late,
                f", exited only at the trough ({fast['exit_date']:%d %b %Y})",
                "",
            )
            + f" and then missed a {fast['missed_rebound']:.0%} rebound before re-entering "
            f"on {fast['reentry_date']:%d %b %Y}"
        )
    else:
        fast_txt = "the COVID window is not covered"
    bond_txt = pick(
        tlt_pre["sharpe"] < tlt_pre["bnh_sharpe"],
        f"Before the August 2020 bond peak the rule **trailed** buy-and-hold TLT "
        f"(Sharpe {tlt_pre['sharpe']:.2f} vs {tlt_pre['bnh_sharpe']:.2f}); "
        f"afterwards its drawdown was {tlt_post['max_drawdown']:.0%} vs "
        f"{tlt_post['bnh_max_drawdown']:.0%}.",
        f"Before the August 2020 bond peak the rule matched or beat buy-and-hold TLT "
        f"(Sharpe {tlt_pre['sharpe']:.2f} vs {tlt_pre['bnh_sharpe']:.2f}); "
        f"afterwards its drawdown was {tlt_post['max_drawdown']:.0%} vs "
        f"{tlt_post['bnh_max_drawdown']:.0%}.",
    )
    verdict = pick(
        sharpe_wins > n / 2 and med["sharpe_diff"] > 0.05,
        "a modest but broad risk-adjusted improvement",
        "no reliable risk-adjusted improvement: the Sharpe comparison is close to a coin flip",
    )
    faber_txt = pick(
        med["cagr_diff"] < 0,
        "The drawdown reduction matches Faber (2007); the return give-up is less flattering "
        "than his long-history results with a monthly 10-month average",
        "The drawdown reduction and the comparable returns match Faber (2007)",
    )
    only = pick(cagr_wins <= n / 2, "only ", "")
    if ci_pos + ci_neg == 0:
        boot_txt = (
            f"A paired stationary bootstrap's 95% interval for the Sharpe gap covers zero in "
            f"**every** market (median half-width {ci_halfwidth:.2f})."
        )
    else:
        boot_txt = (
            f"A paired stationary bootstrap's 95% interval for the Sharpe gap excludes zero in "
            f"{ci_pos} market{'s' if ci_pos != 1 else ''} in the rule's favour and {ci_neg} "
            f"against (median half-width {ci_halfwidth:.2f})."
        )
    cost_etf, cost_btc = costs_for("SPY").bps, costs_for("BTC-USD").bps
    timing_txt = pick(
        mix_dd_wins > n / 2,
        "the rule still has the shallower drawdown in most markets, so the timing, not "
        "just the lower average exposure, cuts the tails",
        "most of the drawdown reduction is reproduced by simply holding less of the asset",
    )

    text = f"""**Question.** Is the EMA50/200 result a property of NVDA in 2022-2025, or does the same untuned rule add value across asset classes?

**Setup.** The identical rule runs on all {n} assets over each one's full history (warm-up on prior data, next-close fills, {cost_etf:g} bp per side, {cost_btc:g} bp for Bitcoin, T-bill cash), against buy-and-hold over the same window and an *exposure-matched mix* holding the rule's average exposure in the asset and the rest in T-bills (its Sharpe equals buy-and-hold's by construction, max gap {mix_sharpe_gap:.3f}, so it is the yardstick for CAGR and drawdown).

**Findings.**

- Sharpe is higher than buy-and-hold in **{sharpe_wins} of {n}** markets (median difference {med["sharpe_diff"]:+.3f}); CAGR is higher in {only}**{cagr_wins} of {n}** (median {med["cagr_diff"]:+.1%} a year). {boot_txt} Excluding the two hindsight picks (NVDA, Bitcoin), the rule wins on Sharpe in {core_sharpe_wins} of {len(core)} ETFs and on CAGR in {core_cagr_wins} of {len(core)}; the picks win {picks_sharpe_wins} of {len(picks)} on Sharpe.
- Maximum drawdown is shallower in **{dd_wins} of {n}** markets, by a median {med["dd_diff"] * 100:.0f} percentage points. Against the exposure-matched mix (median exposure {med["exposure"]:.0%}) the rule has a shallower drawdown in **{mix_dd_wins} of {n}** (median {med["mix_dd_diff"] * 100:+.0f} pp) and a higher CAGR in {mix_cagr_wins} of {n}.
- By class, the mean Sharpe gain is positive for {_join(pos_cls)}; zero or negative for {_join(neg_cls)}.
- The rule is slow: a median {med["trades"]:.1f} round trips a year and a {med["hit"]:.0%} hit rate, {hit_txt}.
- On SPY it gave {slow_txt} of protection in the 2000-02 and 2007-09 bears, but {fast_txt}. {bond_txt}

**Interpretation.** A long/flat trend rule behaves like a dynamically replicated protective put: it sells after a sustained fall and buys back after a recovery, so it truncates slow bear markets and pays a premium in whipsaws and in missed V-shaped rebounds (Fung & Hsieh 2001 model trend followers as lookback straddles). That premium shows up as the CAGR give-up; the protection shows up as drawdowns. Slow, grinding declines (US equities in 2000-02 and 2007-09, Treasuries in 2022) give the 200-day average time to react; fast crashes (COVID) and assets in a secular uptrend (Treasuries before 2020) make time out of the market close to pure opportunity cost. Monthly, {cap_txt} (median upside capture {med["up"]:.0%}, downside {med["down"]:.0%}). Against the exposure-matched mix, {timing_txt}. Overall the evidence is {verdict}. {faber_txt}. Time-series momentum is broad but modest per asset and earns its keep mainly through diversification across many markets and through shorting downtrends, which this long/flat single-asset rule does not do (Moskowitz, Ooi & Pedersen 2012; Hurst, Ooi & Pedersen 2017).
"""

    after = (
        f"Histories run {df['years'].min():.0f}-{df['years'].max():.0f} years, so a single "
        f"Sharpe ratio has a standard error of roughly 1/sqrt(years) = "
        f"{1 / np.sqrt(df['years'].max()):.2f}-{1 / np.sqrt(df['years'].min()):.2f}; the paired "
        f"bootstrap puts the standard error of the *gap* near {ci_halfwidth / 1.96:.2f} "
        "(narrower, because the rule and buy-and-hold share most of their days), still far "
        f"larger than the median gap. The {n} markets are "
        "also correlated (SPY, QQQ, IWM, VNQ and NVDA all load on US stocks), so the counts are not "
        "independent tests. Selection caveats: the ETFs are the long-lived survivors of their "
        "categories, and NVDA and Bitcoin are included only because they are famous winners. "
        "The matched mix rebalances daily without costs, a slight advantage to the benchmark."
    )

    results = {
        "n_assets": n,
        "sharpe_beats_bnh": sharpe_wins,
        "cagr_beats_bnh": cagr_wins,
        "maxdd_shallower_than_bnh": dd_wins,
        "calmar_beats_bnh": calmar_wins,
        "maxdd_shallower_than_matched": mix_dd_wins,
        "cagr_beats_matched": mix_cagr_wins,
        "median_sharpe_diff": med["sharpe_diff"],
        "median_cagr_diff": med["cagr_diff"],
        "median_maxdd_diff": med["dd_diff"],
        "median_maxdd_diff_vs_matched": med["mix_dd_diff"],
        "median_cagr_diff_vs_matched": med["mix_cagr_diff"],
        "median_exposure": med["exposure"],
        "median_beta": med["beta"],
        "median_trades_per_year": med["trades"],
        "median_hit_rate": med["hit"],
        "sharpe_gap_ci_above_zero": ci_pos,
        "sharpe_gap_ci_below_zero": ci_neg,
        "core_sharpe_beats_bnh": core_sharpe_wins,
        "core_cagr_beats_bnh": core_cagr_wins,
        "n_core": len(core),
        "median_upside_capture": med["up"],
        "median_downside_capture": med["down"],
        "spy_protection_dotcom": slow_prot[0] if slow_prot else float("nan"),
        "spy_protection_gfc": slow_prot[1] if len(slow_prot) > 1 else float("nan"),
        "spy_covid_strat_return": float(fast["strat_return"]) if fast is not None else np.nan,
        "spy_covid_bnh_return": float(fast["bench_return"]) if fast is not None else np.nan,
        "tlt_sharpe_pre2020": float(tlt_pre["sharpe"]),
        "tlt_bnh_sharpe_pre2020": float(tlt_pre["bnh_sharpe"]),
        "tlt_maxdd_post2020": float(tlt_post["max_drawdown"]),
        "tlt_bnh_maxdd_post2020": float(tlt_post["bnh_max_drawdown"]),
    }
    for t in tickers:
        results[f"{t}_sharpe"] = float(df.loc[t, "sharpe"])
        results[f"{t}_bnh_sharpe"] = float(df.loc[t, "bnh_sharpe"])
        results[f"{t}_maxdd"] = float(df.loc[t, "max_drawdown"])
        results[f"{t}_bnh_maxdd"] = float(df.loc[t, "bnh_max_drawdown"])

    return Section(
        name=NAME,
        title=TITLE,
        text=text,
        figures=[
            ("Sharpe ratio per asset, rule vs buy & hold.", ctx.fig_path(f"{NAME}_sharpe")),
            (
                "Maximum drawdown per asset: rule, exposure-matched mix and buy & hold.",
                ctx.fig_path(f"{NAME}_maxdd"),
            ),
            (
                "Return give-up vs drawdown reduction, one dot per asset.",
                ctx.fig_path(f"{NAME}_tradeoff"),
            ),
            (
                "Growth of $1 (log scale) for every asset, rule vs buy & hold.",
                ctx.fig_path(f"{NAME}_equity_grid"),
            ),
        ],
        tables=[("EMA50/200 vs buy & hold and the exposure-matched mix, per asset.", display)],
        after=after,
        results=results,
    )

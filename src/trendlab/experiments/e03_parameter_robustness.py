"""Experiment 3: parameter robustness and multiple testing on the EMA-crossover grid.

The textbook 50/200 "golden cross" is one cell of a 62-cell (fast, slow) grid.
If the rule captures a real trend premium, neighbouring cells should look alike
(a *plateau*); if 50/200 only looked good by luck, it would be an isolated
*spike*. Searching the grid and reporting the best cell is a multiple-testing
problem, so the winner is judged with the deflated Sharpe ratio (Bailey &
Lopez de Prado 2014) and the whole search with the probability of backtest
overfitting from combinatorially symmetric cross-validation (Bailey, Borwein,
Lopez de Prado & Zhu 2017).

Two null hypotheses are tested, because they answer different questions:

* **absolute** -- excess returns over T-bills have zero Sharpe (is the rule
  profitable at all?), and
* **relative** -- returns *in excess of buy-and-hold* have zero mean (does the
  timing add anything over simply holding the asset? -- the notebook's claim).

The README's Bitcoin "tuning" story (EMA12/128 first, then 50/200 as an
"improvement") is replayed as two points of the same search.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import metrics, plotting, stats, validation
from ..plotting import _titles, save_figure, theme
from ..report import df_to_markdown
from ..strategies import BuyAndHold, EMACrossover
from .common import Context, Section

NAME = "e03_parameter_robustness"
TITLE = "3. Parameter robustness: is 50/200 a plateau or a lucky spike?"

ASSETS = ("SPY", "QQQ", "BTC-USD")
FAST = [5, 10, 20, 30, 40, 50, 75, 100, 150]
SLOW = [50, 75, 100, 150, 200, 250, 300, 400]
TEXTBOOK = EMACrossover(50, 200)
README_BTC = EMACrossover(12, 128)  # the README's first BTC attempt (not on the grid)
LABEL = {"SPY": "SPY", "QQQ": "QQQ", "BTC-USD": "BTC"}


# --------------------------------------------------------------------------- helpers


def _ann(sr_per_period: float, periods: int) -> float:
    return float(stats.annualize_sharpe(sr_per_period, periods))


def _pc1_share(x: pd.DataFrame) -> float:
    """Share of the cross-configuration correlation explained by the first PC."""
    ev = np.linalg.eigvalsh(np.corrcoef(x.to_numpy().T))
    return float(ev.max() / ev.sum())


def _neighbour_mean(piv: pd.DataFrame, fast: float, slow: float) -> float:
    """Mean Sharpe of the (up to four) grid neighbours of a (fast, slow) cell."""
    i, j = piv.index.get_loc(slow), piv.columns.get_loc(fast)
    vals = [
        piv.iat[a, b]
        for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
        if 0 <= a < piv.shape[0] and 0 <= b < piv.shape[1]
    ]
    return float(np.nanmean(vals))


def _fmt_df(df: pd.DataFrame, pct=(), ints=(), digits: int = 2, prob=()) -> pd.DataFrame:
    """Display copy: percentages, integers, probabilities and fixed decimals as strings."""
    out = df.copy().astype(object)
    for c in df.columns:
        for i in df.index:
            v = df.at[i, c]
            if isinstance(v, str) or v is None:
                continue
            if pd.isna(v):
                out.at[i, c] = "–"
            elif c in pct:
                out.at[i, c] = f"{v:.1%}"
            elif c in ints:
                out.at[i, c] = f"{int(v)}"
            elif c in prob:
                out.at[i, c] = f"{v:.3f}"
            else:
                out.at[i, c] = f"{v:.{digits}f}"
    return out


def _sharpe_se(sr_annual: float, n_bars: int, periods: int) -> float:
    """Asymptotic standard error of an annualised Sharpe estimate (iid returns, Lo 2002)."""
    sr = sr_annual / np.sqrt(periods)
    return float(np.sqrt((1.0 + 0.5 * sr**2) / max(n_bars - 1, 1)) * np.sqrt(periods))


def _save(ctx: Context, name: str, raw: pd.DataFrame, display: pd.DataFrame) -> None:
    """Raw numbers to CSV via ``ctx.save_table``; the Markdown copy gets the display version."""
    ctx.save_table(name, raw)
    (ctx.table_dir / f"{name}.md").write_text(df_to_markdown(display) + "\n")


def _asset_study(ctx: Context, ticker: str, n_blocks: int, n_boot: int) -> dict:
    prices, cfg = ctx.prices(ticker), ctx.config(ticker)
    grid = validation.parameter_grid(FAST, SLOW)
    extra = [README_BTC] if ticker == "BTC-USD" else []
    strategies = grid + extra
    gm = validation.grid_metrics(prices, strategies, cfg, ctx.rf)
    rets = validation.grid_returns(prices, strategies, cfg, ctx.rf)
    names = [s.name for s in grid]
    bh = ctx.backtest(ticker, BuyAndHold(), start=rets.index[0])
    bh_r = bh.returns.reindex(rets.index)
    rf = bh.rf.reindex(rets.index)
    p = bh.periods_per_year

    m = gm.loc[names]
    sh = m["sharpe"]
    tb = TEXTBOOK.name
    best = str(sh.idxmax())
    bh_sharpe = metrics.sharpe(bh_r, rf, p)

    # absolute null: Sharpe of excess-of-T-bill returns is zero
    ts = stats.trial_sharpes(rets[names], rf)
    dsr_best = stats.deflated_sharpe_ratio(rets[best], ts, rf=rf, periods=p)
    psr_tb = stats.probabilistic_sharpe_ratio(rets[tb], 0.0, rf)
    # relative null: returns in excess of buy-and-hold have zero mean
    active = rets.sub(bh_r, axis=0)
    ts_act = stats.trial_sharpes(active[names])
    best_act = str(ts_act.idxmax())
    dsr_act = stats.deflated_sharpe_ratio(active[best_act], ts_act, periods=p)
    psr_act_tb = stats.probabilistic_sharpe_ratio(active[tb], 0.0)

    # risk-adjusted relative test: is 50/200's Sharpe above buy-and-hold's?
    boot = stats.bootstrap_sharpe_difference(rets[tb], bh_r, rf=rf, n_boot=n_boot, periods=p)

    pbo = stats.pbo_cscv(rets[names], n_blocks=n_blocks, rf=rf)
    piv = validation.grid_pivot(m, "sharpe")
    root_p = np.sqrt(p)
    row = {
        "start": rets.index[0].date().isoformat(),
        "end": rets.index[-1].date().isoformat(),
        "years": (len(rets) - 1) / p,
        "n_configs": len(names),
        "bnh_sharpe": bh_sharpe,
        "bnh_psr": stats.probabilistic_sharpe_ratio(bh_r, 0.0, rf),
        "bnh_max_drawdown": metrics.max_drawdown(bh_r),
        "sharpe_se": _sharpe_se(float(sh.median()), len(rets), p),
        "tb_sharpe_gain": boot["diff"],
        "tb_sharpe_gain_ci_low": boot["ci_low"],
        "tb_sharpe_gain_ci_high": boot["ci_high"],
        "tb_sharpe_gain_p": boot["p_value"],
        "tb_sharpe": float(sh[tb]),
        "tb_rank": int((sh > sh[tb]).sum() + 1),
        "tb_percentile": float((sh < sh[tb]).mean()),
        "grid_median_sharpe": float(sh.median()),
        "grid_q25_sharpe": float(sh.quantile(0.25)),
        "grid_q75_sharpe": float(sh.quantile(0.75)),
        "grid_min_sharpe": float(sh.min()),
        "share_beat_bnh_sharpe": float((sh > bh_sharpe).mean()),
        "best_config": best,
        "best_sharpe": float(sh[best]),
        "best_neighbour_sharpe": _neighbour_mean(piv, m.at[best, "fast"], m.at[best, "slow"]),
        "tb_neighbour_sharpe": _neighbour_mean(piv, 50, 200),
        "short_slow_sharpe": float(sh[m["slow"] <= 150].mean()),
        "long_slow_sharpe": float(sh[m["slow"] >= 250].mean()),
        "sr0_annual": dsr_best["sr0_annual"],
        "dsr_best": dsr_best["dsr"],
        "psr_tb": psr_tb,
        "tb_ir_vs_bnh": _ann(ts_act[tb], p),
        "best_ir_config": best_act,
        "best_ir_vs_bnh": _ann(ts_act[best_act], p),
        "ir_sr0_annual": dsr_act["sr0_annual"],
        "dsr_best_vs_bnh": dsr_act["dsr"],
        "psr_tb_vs_bnh": psr_act_tb,
        "share_ir_positive": float((ts_act > 0).mean()),
        "pc1_share_active": _pc1_share(active[names]),
        "pbo": pbo["pbo"],
        "pbo_is_sharpe": float(np.mean(pbo["is_metric"]) * root_p),
        "pbo_oos_sharpe": float(np.mean(pbo["oos_metric"]) * root_p),
        "pbo_degradation": pbo["performance_degradation"],
        "pbo_combinations": pbo["n_combinations"],
    }
    out = {
        "row": row,
        "metrics": gm,
        "names": names,
        "rets": rets,
        "active": active,
        "ts": ts,
        "ts_act": ts_act,
        "bh_r": bh_r,
        "rf": rf,
        "periods": p,
        "pbo": pbo,
    }
    return out


def _btc_story(st: dict) -> pd.DataFrame:
    """README's BTC tuning replayed: 12/128, 50/200 and the grid's best on one window."""
    gm, rets, active = st["metrics"], st["rets"], st["active"]
    ts, ts_act, p, rf = st["ts"], st["ts_act"], st["periods"], st["rf"]
    names = st["names"]
    n_all = len(names) + 1  # the grid plus the README's off-grid first attempt
    grid_sh = gm.loc[names, "sharpe"]
    var_act = list(ts_act.to_numpy()) + [stats.sharpe_per_period(active[README_BTC.name])]
    rows = {}
    picks = [
        ("EMA12/128 (README, 1st try)", README_BTC.name),
        ("EMA50/200 (README, 'improved')", TEXTBOOK.name),
        (f"{st['row']['best_config']} (grid best Sharpe)", st["row"]["best_config"]),
    ]
    for label, key in picks:
        r, a = rets[key], active[key]
        rows[label] = {
            "sharpe": float(gm.at[key, "sharpe"]),
            "cagr": float(gm.at[key, "cagr"]),
            "max_drawdown": float(gm.at[key, "max_drawdown"]),
            "total_return": metrics.total_return(r),
            "grid_percentile": float((grid_sh < gm.at[key, "sharpe"]).mean()),
            "ir_vs_bnh": _ann(stats.sharpe_per_period(a), p),
            "p_beats_bnh_n1": stats.probabilistic_sharpe_ratio(a, 0.0),
            "p_beats_bnh_n2": stats.deflated_sharpe_ratio(a, var_act, n_trials=2, periods=p)["dsr"],
            f"p_beats_bnh_n{n_all}": stats.deflated_sharpe_ratio(
                a, var_act, n_trials=n_all, periods=p
            )["dsr"],
            f"dsr_abs_n{n_all}": stats.deflated_sharpe_ratio(
                r, list(ts.to_numpy()), rf=rf, n_trials=n_all, periods=p
            )["dsr"],
        }
    bh = st["bh_r"]
    rows["Buy & Hold"] = {
        "sharpe": metrics.sharpe(bh, rf, p),
        "cagr": metrics.cagr(bh, p),
        "max_drawdown": metrics.max_drawdown(bh),
        "total_return": metrics.total_return(bh),
    }
    df = pd.DataFrame(rows).T
    df.index.name = "rule"
    return df


# --------------------------------------------------------------------------- figures


def _header_axes(fig, gs, n_cols: int):
    hdr = fig.add_subplot(gs[0, :n_cols])
    hdr.axis("off")
    return hdr


def _plateau_chart(studies: dict, path, title: str, subtitle: str) -> None:
    """Sorted grid Sharpe per asset with buy-and-hold, 50/200, the best and BTC 12/128."""
    with theme() as t:
        n = len(studies)
        fig = plt.figure(figsize=(4.0 * n + 1.0, 4.9))
        gs = fig.add_gridspec(2, n, height_ratios=[0.02, 1.0])
        _titles(_header_axes(fig, gs, n), title, subtitle)
        cats = t["categorical"]
        for k, (ticker, st) in enumerate(studies.items()):
            ax = fig.add_subplot(gs[1, k])
            row = st["row"]
            sh = st["metrics"].loc[st["names"], "sharpe"].sort_values(ascending=False)
            x = np.arange(1, len(sh) + 1)
            q25, q75 = row["grid_q25_sharpe"], row["grid_q75_sharpe"]
            ax.axhspan(q25, q75, color=cats[0], alpha=0.08, lw=0, zorder=0)
            ax.plot(x, sh.to_numpy(), "o", ms=4, color=t["muted"], alpha=0.8, zorder=2)
            bh_line = ax.axhline(
                row["bnh_sharpe"],
                color=t["benchmark"],
                lw=1.4,
                ls="--",
                zorder=1,
                label=f"Buy & hold {row['bnh_sharpe']:.2f}",
            )
            marks = [(TEXTBOOK.name, "EMA50/200", cats[0]), (row["best_config"], "best", cats[1])]
            if README_BTC.name in st["metrics"].index:
                marks.append((README_BTC.name, "EMA12/128 (off-grid)", cats[2]))
            for key, lbl, col in marks:
                v = float(st["metrics"].at[key, "sharpe"])
                pos = float((sh > v).sum() + (0.5 if key not in sh.index else 1))
                ax.plot([pos], [v], "o", ms=9, color=col, mec=t["surface"], mew=1.8, zorder=4)
                text = f"{key.replace('EMA', '')} {v:.2f}" if lbl == "best" else f"{lbl} {v:.2f}"
                # the sorted curve descends, so up-and-right of a point is empty
                right = pos < 0.7 * len(sh)
                ax.annotate(
                    text,
                    xy=(pos, v),
                    xytext=(6, 5) if right else (-6, -5),
                    textcoords="offset points",
                    ha="left" if right else "right",
                    va="bottom" if right else "top",
                    fontsize=8.5,
                    color=t["ink"],
                    zorder=5,
                )
            ax.legend(
                handles=[bh_line, plt.Rectangle((0, 0), 1, 1, color=cats[0], alpha=0.15, lw=0)],
                labels=[bh_line.get_label(), "Grid interquartile range"],
                loc="lower left",
            )
            ax.set_title(f"{LABEL[ticker]} · {row['start'][:4]}–{row['end'][:4]}", loc="left")
            ax.set_xlabel("Rank in grid (1 = best Sharpe)")
            if k == 0:
                ax.set_ylabel("Sharpe ratio (excess of T-bills)")
            ax.annotate(
                f"1 s.e. of a Sharpe estimate ≈ {row['sharpe_se']:.2f}\n"
                f"grid interquartile width = {q75 - q25:.2f}",
                xy=(0.97, 0.97),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=8.5,
                color=t["ink2"],
            )
            ax.set_xlim(-1, len(sh) + 2)
            lo = min(sh.min(), row["bnh_sharpe"])
            hi = max(sh.max(), row["bnh_sharpe"])
            pad = 0.12 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
        save_figure(fig, path)


def _logit_chart(studies: dict, path, title: str, subtitle: str) -> None:
    """CSCV logit distribution of the in-sample winner's out-of-sample rank, per asset."""
    with theme() as t:
        n = len(studies)
        fig = plt.figure(figsize=(4.0 * n + 1.0, 4.2))
        gs = fig.add_gridspec(2, n, height_ratios=[0.02, 1.0])
        _titles(_header_axes(fig, gs, n), title, subtitle)
        allv = np.concatenate([st["pbo"]["logits"] for st in studies.values()])
        bins = np.linspace(np.floor(allv.min()), np.ceil(allv.max()), 31)
        for k, (ticker, st) in enumerate(studies.items()):
            ax = fig.add_subplot(gs[1, k])
            lg = st["pbo"]["logits"]
            below = lg <= 0
            ax.hist(
                lg[below], bins=bins, color=t["categorical"][7], alpha=0.75, label="Below median"
            )
            ax.hist(
                lg[~below], bins=bins, color=t["categorical"][0], alpha=0.75, label="Above median"
            )
            ax.axvline(0, color=t["ink2"], lw=1.0)
            ax.annotate(
                f"PBO = {st['pbo']['pbo']:.0%}",
                xy=(0.03, 0.96),
                xycoords="axes fraction",
                ha="left",
                va="top",
                fontsize=10,
                fontweight="bold",
                color=t["ink"],
            )
            ax.set_title(LABEL[ticker], loc="left")
            ax.set_xlabel("Logit of IS winner's OOS rank")
            if k == 0:
                ax.set_ylabel("CSCV splits")
            if k == n - 1:
                ax.legend(loc="upper right")
        save_figure(fig, path)


# --------------------------------------------------------------------------- run


def run(ctx: Context) -> Section:
    n_blocks = 8 if ctx.quick else 16
    n_boot = 400 if ctx.quick else 2000
    studies = {tk: _asset_study(ctx, tk, n_blocks, n_boot) for tk in ASSETS}
    rows = {tk: st["row"] for tk, st in studies.items()}
    summary = pd.DataFrame(rows).T
    summary.index.name = "asset"

    grid_all = pd.concat(
        {LABEL[tk]: st["metrics"] for tk, st in studies.items()}, names=["asset", "strategy"]
    )
    grid_disp = _fmt_df(
        grid_all,
        pct=("cagr", "max_drawdown", "exposure"),
        ints=("fast", "slow", "n_trades"),
    )
    _save(ctx, f"{NAME}_grid", grid_all, grid_disp)

    btc = _btc_story(studies["BTC-USD"])

    # ---- figures
    figures = []
    for tk, st in studies.items():
        row = st["row"]
        piv = validation.grid_pivot(st["metrics"].loc[st["names"]], "sharpe")
        title = (
            f"{LABEL[tk]}: {row['share_beat_bnh_sharpe']:.0%} of crossovers beat buy-and-hold's "
            f"Sharpe; 50/200 ({row['tb_sharpe']:.2f}) ranks {row['tb_rank']} of {row['n_configs']}"
        )
        sub = (
            f"Sharpe of EMA(fast) > EMA(slow), next-close fills, {ctx.config(tk).costs.bps:g} bp/side, "
            f"{row['start']} to {row['end']}\nColour centred on buy-and-hold Sharpe "
            f"{row['bnh_sharpe']:.2f}: blue beats it, red trails it. Outlined cell = 50/200.\n"
            f"Middle half of the grid: {row['grid_q25_sharpe']:.2f}–{row['grid_q75_sharpe']:.2f}, "
            f"vs 1 standard error of a Sharpe estimate ≈ {row['sharpe_se']:.2f}."
        )
        p = ctx.fig_path(f"{NAME}_sharpe_{LABEL[tk].lower()}")
        plotting.metric_heatmap(
            piv,
            title,
            fmt="{:.2f}",
            center=row["bnh_sharpe"],
            mark=(50, 200),
            subtitle=sub,
            cbar_label="Sharpe ratio",
            xlabel="Fast EMA (bars)",
            ylabel="Slow EMA (bars)",
            path=p,
        )
        plt.close("all")
        figures.append((f"{LABEL[tk]}: Sharpe ratio over the (fast, slow) grid.", p))

    spy = studies["SPY"]
    dd = validation.grid_pivot(spy["metrics"].loc[spy["names"]], "max_drawdown")
    srow = spy["row"]
    p = ctx.fig_path(f"{NAME}_maxdd_spy")
    bh_dd = f"{srow['bnh_max_drawdown']:.0%}".replace("-", "\u2212")
    plotting.metric_heatmap(
        dd,
        f"SPY: every crossover has a shallower drawdown than buy-and-hold's {bh_dd}"
        if np.nanmin(dd.to_numpy()) > srow["bnh_max_drawdown"]
        else f"SPY: max drawdown of each crossover vs buy-and-hold's {bh_dd}",
        fmt=lambda v: f"{v:.0%}",
        center=None,
        mark=(50, 200),
        subtitle=f"Maximum drawdown, {srow['start']} to {srow['end']}; darker = shallower. "
        "Outlined cell = 50/200.",
        cbar_label="Maximum drawdown",
        xlabel="Fast EMA (bars)",
        ylabel="Slow EMA (bars)",
        path=p,
    )
    plt.close("all")
    figures.append(("SPY: maximum drawdown over the grid.", p))

    p = ctx.fig_path(f"{NAME}_plateau")
    beat = [LABEL[t] for t, r in rows.items() if r["share_beat_bnh_sharpe"] >= 0.5]
    _plateau_chart(
        studies,
        p,
        "Sorted grid Sharpe: 50/200 sits inside the pack, not on a spike"
        if all(0.1 <= r["tb_percentile"] <= 0.9 for r in rows.values())
        else "Sorted grid Sharpe: where 50/200 and the in-sample best sit",
        f"All {srow['n_configs']} EMA crossovers per asset, sorted; shaded band = interquartile "
        f"range; dashed = buy-and-hold. Majority of the grid beats B&H Sharpe in: "
        f"{', '.join(beat) if beat else 'none'}.",
    )
    figures.append(("Every configuration's Sharpe, sorted, with the textbook and best cells.", p))

    p = ctx.fig_path(f"{NAME}_pbo_logits")
    _logit_chart(
        studies,
        p,
        f"The in-sample winner ranks below the out-of-sample median in "
        f"{min(r['pbo'] for r in rows.values()):.0%}–{max(r['pbo'] for r in rows.values()):.0%} "
        "of splits",
        f"CSCV with S = {n_blocks} blocks ({srow['pbo_combinations']:,} splits): logit of the "
        "out-of-sample rank of the in-sample winner; mass left of 0 = overfit.",
    )
    figures.append(("Probability of backtest overfitting (CSCV) per asset.", p))

    # ---- display tables
    disp = summary[
        [
            "start",
            "n_configs",
            "bnh_sharpe",
            "tb_sharpe",
            "tb_rank",
            "grid_median_sharpe",
            "grid_q25_sharpe",
            "grid_q75_sharpe",
            "sharpe_se",
            "share_beat_bnh_sharpe",
            "best_config",
            "best_sharpe",
            "sr0_annual",
            "dsr_best",
            "psr_tb",
            "bnh_psr",
        ]
    ]
    disp = _fmt_df(
        disp,
        pct=("share_beat_bnh_sharpe",),
        ints=("n_configs", "tb_rank"),
        prob=("dsr_best", "psr_tb", "bnh_psr"),
    ).rename(
        columns={
            "start": "Start",
            "n_configs": "N",
            "bnh_sharpe": "B&H Sharpe",
            "tb_sharpe": "50/200 Sharpe",
            "tb_rank": "50/200 rank",
            "grid_median_sharpe": "Grid median",
            "grid_q25_sharpe": "Q25",
            "grid_q75_sharpe": "Q75",
            "sharpe_se": "Sharpe s.e.",
            "share_beat_bnh_sharpe": "Beat B&H Sharpe",
            "best_config": "Best",
            "best_sharpe": "Best Sharpe",
            "sr0_annual": "SR0 (N trials)",
            "dsr_best": "DSR best",
            "psr_tb": "PSR 50/200",
            "bnh_psr": "PSR B&H",
        }
    )
    gain_ci = pd.Series(
        {
            tk: f"{r['tb_sharpe_gain_ci_low']:+.2f} to {r['tb_sharpe_gain_ci_high']:+.2f}"
            for tk, r in rows.items()
        }
    )
    disp_rel = _fmt_df(
        summary[
            [
                "tb_sharpe_gain",
                "tb_sharpe_gain_p",
                "tb_ir_vs_bnh",
                "psr_tb_vs_bnh",
                "best_ir_config",
                "best_ir_vs_bnh",
                "ir_sr0_annual",
                "dsr_best_vs_bnh",
                "share_ir_positive",
                "pc1_share_active",
                "pbo",
                "pbo_is_sharpe",
                "pbo_oos_sharpe",
            ]
        ],
        pct=("share_ir_positive", "pc1_share_active", "pbo"),
        prob=("tb_sharpe_gain_p", "psr_tb_vs_bnh", "dsr_best_vs_bnh"),
    )
    disp_rel.insert(1, "95% CI", gain_ci.reindex(disp_rel.index).str.replace("-", "−"))
    disp_rel = disp_rel.rename(
        columns={
            "tb_sharpe_gain": "50/200 Sharpe − B&H Sharpe",
            "tb_sharpe_gain_p": "p (gain > 0)",
            "tb_ir_vs_bnh": "50/200 IR vs B&H",
            "psr_tb_vs_bnh": "P(50/200 beats B&H)",
            "best_ir_config": "Best-IR config",
            "best_ir_vs_bnh": "Best IR",
            "ir_sr0_annual": "IR0 (N trials)",
            "dsr_best_vs_bnh": "Deflated P(best beats B&H)",
            "share_ir_positive": "Configs with IR > 0",
            "pc1_share_active": "PC1 of active returns",
            "pbo": "PBO",
            "pbo_is_sharpe": "IS-best Sharpe (IS)",
            "pbo_oos_sharpe": "IS-best Sharpe (OOS)",
        }
    )
    n_all = len(studies["BTC-USD"]["names"]) + 1
    disp_btc = _fmt_df(
        btc,
        pct=("cagr", "max_drawdown", "total_return", "grid_percentile"),
        prob=("p_beats_bnh_n1", "p_beats_bnh_n2", f"p_beats_bnh_n{n_all}", f"dsr_abs_n{n_all}"),
    ).rename(
        columns={
            "sharpe": "Sharpe",
            "cagr": "CAGR",
            "max_drawdown": "Max DD",
            "total_return": "Total return",
            "grid_percentile": "Grid percentile",
            "ir_vs_bnh": "IR vs B&H",
            "p_beats_bnh_n1": "P(beats B&H), N=1",
            "p_beats_bnh_n2": "N=2",
            f"p_beats_bnh_n{n_all}": f"N={n_all}",
            f"dsr_abs_n{n_all}": f"DSR (Sharpe > 0), N={n_all}",
        }
    )
    tables = [
        ("Grid robustness and the absolute test (Sharpe of excess-of-T-bill returns > 0).", disp),
        (
            "The relative tests: 50/200's Sharpe gain over buy-and-hold (paired stationary "
            "bootstrap, 95% CI); returns in excess of buy-and-hold (IR = annualised mean / "
            "tracking error), with deflation for the search; and CSCV PBO.",
            disp_rel,
        ),
        ("The README's Bitcoin tuning story on one common window.", disp_btc),
    ]
    _save(ctx, f"{NAME}_summary", summary, pd.concat([disp, disp_rel], axis=1))
    _save(ctx, f"{NAME}_btc_story", btc, disp_btc)

    # ---- text
    s, q, b = rows["SPY"], rows["QQQ"], rows["BTC-USD"]
    rv = list(rows.values())
    bt = btc.iloc[0], btc.iloc[1]
    improved = bt[1]["sharpe"] > bt[0]["sharpe"]
    iqr_w = [r["grid_q75_sharpe"] - r["grid_q25_sharpe"] for r in rv]
    rel_iqr = [w / abs(r["grid_median_sharpe"]) for w, r in zip(iqr_w, rv, strict=True)]
    se_ratio = [w / r["sharpe_se"] for w, r in zip(iqr_w, rv, strict=True)]
    plateau_all = max(rel_iqr) < 0.25 and max(se_ratio) < 1.0
    max_pbo = max(r["pbo"] for r in rv)
    abs_pass = all(r["dsr_best"] > 0.95 and r["psr_tb"] > 0.95 for r in rv)
    bnh_pass = all(r["bnh_psr"] > 0.95 for r in rv)
    rel_fail_n1 = all(r["psr_tb_vs_bnh"] < 0.95 for r in rv)
    rel_fail_best = all(r["dsr_best_vs_bnh"] < 0.95 for r in rv)
    oos_at_median = all(r["pbo_oos_sharpe"] <= r["grid_median_sharpe"] + 0.01 for r in rv)
    gain_sig = [LABEL[tk] for tk, r in rows.items() if r["tb_sharpe_gain_ci_low"] > 0]
    pc1 = [r["pc1_share_active"] for r in rv]
    spike = [
        LABEL[tk]
        for tk, r in rows.items()
        if r["best_sharpe"] - r["best_neighbour_sharpe"]
        > 0.5 * (r["best_sharpe"] - r["grid_median_sharpe"])
    ]
    # 50/200 vs its four grid neighbours, measured in standard errors of a Sharpe estimate
    tb_bump = [(r["tb_sharpe"] - r["tb_neighbour_sharpe"]) / r["sharpe_se"] for r in rv]

    spike_txt = "".join(
        f" The sharpest local peak is the cell a search would report, {LABEL[tk]}'s "
        f"{rows[tk]['best_config']} ({rows[tk]['best_sharpe']:.2f} vs neighbours "
        f"{rows[tk]['best_neighbour_sharpe']:.2f})."
        for tk in rows
        if LABEL[tk] in spike[:1]
    )
    tb_txt = (
        f"50/200 differs from its four neighbours' average by "
        f"{s['tb_sharpe'] - s['tb_neighbour_sharpe']:+.2f} / "
        f"{q['tb_sharpe'] - q['tb_neighbour_sharpe']:+.2f} / "
        f"{b['tb_sharpe'] - b['tb_neighbour_sharpe']:+.2f} (at most "
        f"{max(abs(x) for x in tb_bump):.2f} s.e.)"
        + (": neither a spike nor a hole." if max(abs(x) for x in tb_bump) < 0.5 else ".")
    )

    def _vs(a: float, bnh: float) -> str:
        return "level with" if abs(a - bnh) < 0.01 else ("above" if a > bnh else "below")

    def _rank_line(tk: str) -> str:
        r = rows[tk]
        return (
            f"{LABEL[tk]} {r['tb_sharpe']:.2f} (rank {r['tb_rank']}, "
            f"{_vs(r['tb_sharpe'], r['bnh_sharpe'])} B&H {r['bnh_sharpe']:.2f}, grid IQR "
            f"{r['grid_q25_sharpe']:.2f}-{r['grid_q75_sharpe']:.2f})"
        )

    grad = (
        f" BTC tilts towards short slow windows (slow <= 150: {b['short_slow_sharpe']:.2f}; "
        f">= 250: {b['long_slow_sharpe']:.2f})."
        if abs(b["short_slow_sharpe"] - b["long_slow_sharpe"]) > 0.05
        else ""
    )
    ir_neg = bt[0]["ir_vs_bnh"] < 0 and bt[1]["ir_vs_bnh"] < 0
    btc_compound = (
        f"; 12/128 still compounds to more ({bt[0]['total_return']:.0%} vs "
        f"{btc.loc['Buy & Hold', 'total_return']:.0%}) through lower volatility drag, not a "
        "higher average return"
        if bt[0]["total_return"] > btc.loc["Buy & Hold", "total_return"] and bt[0]["ir_vs_bnh"] < 0
        else ""
    )
    if max_pbo > 0.5:
        pbo_txt = (
            "PBO above 50% means the in-sample winner usually lands *below* the median rule out "
            "of sample; among near-ties this is partly mechanical, because CSCV halves are "
            "complements (Bailey et al. 2017). Either way, the ranking of cells is noise."
        )
    else:
        pbo_txt = (
            "The in-sample ranking carries little information out of sample (Bailey et al. 2017)."
        )
    verdict = (
        "The absolute claim (trend timing earns more than T-bills) survives any reasonable N"
        if abs_pass
        else "The absolute claim is not robust on every asset"
    ) + (
        ', but "earns more than buy-and-hold" fails even at N = 1.'
        if rel_fail_n1
        else (
            ', and "earns more than buy-and-hold" fails once the search is deflated.'
            if rel_fail_best
            else "."
        )
    )
    gain_txt = (
        "None of the three gains is distinguishable from zero"
        if not gain_sig
        else f"Only {', '.join(gain_sig)} has a 95% interval above zero"
    )

    # sections 2 and 9 run 50/200 on each asset's full history (200-bar warm-up only);
    # this grid's common window waits for the 400-bar EMA, so its 50/200 Sharpes differ
    full = {}
    for tk in ("SPY", "QQQ"):
        res, _ = ctx.pair(tk, TEXTBOOK)
        full[tk] = {
            "start": res.start.date().isoformat(),
            "sharpe": float(metrics.sharpe(res.returns, res.rf, res.periods_per_year)),
        }
    qqq_skips_bust = full["QQQ"]["start"] < "2002-01-01" and q["start"] > "2002-10-09"
    window_txt = (
        f" The common window starts later than the full histories of sections 2 and 9, "
        f"because it waits for the {SLOW[-1]}-bar EMA (SPY from {s['start']} instead of "
        f"{full['SPY']['start']}, QQQ from {q['start']} instead of {full['QQQ']['start']}), "
        f"so 50/200's Sharpe here differs from theirs: SPY {s['tb_sharpe']:.2f} vs "
        f"{full['SPY']['sharpe']:.2f}, QQQ {q['tb_sharpe']:.2f} vs {full['QQQ']['sharpe']:.2f}"
        + ("; QQQ's window excludes the 2001-02 decline." if qqq_skips_bust else ".")
    )

    def _ci(r: dict) -> str:
        return f"{r['tb_sharpe_gain']:+.2f} [{r['tb_sharpe_gain_ci_low']:+.2f}, {r['tb_sharpe_gain_ci_high']:+.2f}]"

    text = f"""**Question.** Is EMA50/200 a robust choice on a plateau of similar rules, or a lucky cell that does not survive the grid search?

**Setup.** {s["n_configs"]} EMA crossovers (fast {FAST[0]}-{FAST[-1]}, slow {SLOW[0]}-{SLOW[-1]}) on SPY, QQQ and BTC, default settings, one common window per asset after the slowest EMA warms up ({s["start"][:4]} / {q["start"][:4]} / {b["start"][:4]} to {s["end"][:4]}). Nulls: "Sharpe over T-bills = 0" and "mean return minus buy-and-hold's = 0" (PSR/DSR, CSCV PBO with S = {n_blocks}); a paired stationary bootstrap compares 50/200's Sharpe with buy-and-hold's.{window_txt}

**Findings.**

- 50/200 Sharpe: {_rank_line("SPY")}; {_rank_line("QQQ")}; {_rank_line("BTC-USD")}, of {s["n_configs"]}. Each IQR is {min(rel_iqr):.0%}-{max(rel_iqr):.0%} of the median and only {min(se_ratio):.2f}-{max(se_ratio):.2f} of one standard error of a Sharpe estimate ({s["sharpe_se"]:.2f} / {q["sharpe_se"]:.2f} / {b["sharpe_se"]:.2f}){": a plateau" if plateau_all else ""}.{grad} {tb_txt}
- {s["share_beat_bnh_sharpe"]:.0%} / {q["share_beat_bnh_sharpe"]:.0%} / {b["share_beat_bnh_sharpe"]:.0%} of cells beat B&H on Sharpe, but {s["share_ir_positive"]:.0%} / {q["share_ir_positive"]:.0%} / {b["share_ir_positive"]:.0%} earn a higher mean return (50/200 IR {s["tb_ir_vs_bnh"]:.2f} / {q["tb_ir_vs_bnh"]:.2f} / {b["tb_ir_vs_bnh"]:.2f}). 50/200's Sharpe gain over B&H is {_ci(s)} / {_ci(q)} / {_ci(b)} (95% CI). {gain_txt}.
- Absolute null: the best cells ({s["best_config"]}, {q["best_config"]}, {b["best_config"]}) have DSR {s["dsr_best"]:.3f} / {q["dsr_best"]:.3f} / {b["dsr_best"]:.3f}{f", but B&H's own PSR is {s['bnh_psr']:.3f} / {q['bnh_psr']:.3f} / {b['bnh_psr']:.3f}: the test only says the assets went up" if bnh_pass else ""}. Relative null: the deflated P(best-IR cell beats B&H) is {s["dsr_best_vs_bnh"]:.2f} / {q["dsr_best_vs_bnh"]:.2f} / {b["dsr_best_vs_bnh"]:.2f}; for 50/200 with no search (N = 1) it is {s["psr_tb_vs_bnh"]:.2f} / {q["psr_tb_vs_bnh"]:.2f} / {b["psr_tb_vs_bnh"]:.2f}.
- PBO: {s["pbo"]:.0%} / {q["pbo"]:.0%} / {b["pbo"]:.0%}. Across CSCV splits the in-sample winner's Sharpe falls from {s["pbo_is_sharpe"]:.2f} / {q["pbo_is_sharpe"]:.2f} / {b["pbo_is_sharpe"]:.2f} to {s["pbo_oos_sharpe"]:.2f} / {q["pbo_oos_sharpe"]:.2f} / {b["pbo_oos_sharpe"]:.2f} out of sample{", at or below the grid median" if oos_at_median else ""}.
- README BTC story ({b["start"][:4]}-{b["end"][:4]}): EMA12/128 Sharpe {bt[0]["sharpe"]:.2f} (above {bt[0]["grid_percentile"]:.0%} of the grid), EMA50/200 {bt[1]["sharpe"]:.2f} (above {bt[1]["grid_percentile"]:.0%}), so the "improvement" {"holds" if improved else "reverses"}. {"Both" if ir_neg else "Their"} IRs vs B&H {"are negative" if ir_neg else "are"} ({bt[0]["ir_vs_bnh"]:.2f}, {bt[1]["ir_vs_bnh"]:.2f}){btc_compound}. Counting two tries, P(50/200 beats B&H) is {bt[1]["p_beats_bnh_n2"]:.2f}.

**Interpretation.** Crossovers of any length are weighted sums of past returns (Levine & Pedersen 2016), so every cell makes the same bet; the first principal component explains {min(pc1):.0%}-{max(pc1):.0%} of the correlation of their active returns, so N = {s["n_configs"]} overstates the independent trials and the DSR haircut is conservative.{spike_txt} {pbo_txt} Bailey & Lopez de Prado (2014) and Harvey & Liu (2015) explain why the bar must rise with the number of trials; even "N = 1" flatters 50/200, which became textbook by looking good on decades of US equity data (collective data snooping; Sullivan, Timmermann & White 1999), while QQQ and BTC were chosen as famous winners. {verdict} Moving from 12/128 to 50/200 is a step between noisy points on one surface, not skill.
"""

    results = {
        "n_configs": int(s["n_configs"]),
        "n_blocks": int(n_blocks),
    }
    for tk, r in rows.items():
        k = LABEL[tk].lower()
        for key in (
            "tb_sharpe",
            "tb_rank",
            "tb_percentile",
            "bnh_sharpe",
            "grid_median_sharpe",
            "grid_q25_sharpe",
            "grid_q75_sharpe",
            "share_beat_bnh_sharpe",
            "best_sharpe",
            "sr0_annual",
            "dsr_best",
            "psr_tb",
            "tb_ir_vs_bnh",
            "psr_tb_vs_bnh",
            "best_ir_vs_bnh",
            "dsr_best_vs_bnh",
            "share_ir_positive",
            "pbo",
            "pbo_is_sharpe",
            "pbo_oos_sharpe",
            "bnh_psr",
            "sharpe_se",
            "tb_sharpe_gain",
            "tb_sharpe_gain_ci_low",
            "tb_sharpe_gain_ci_high",
            "tb_sharpe_gain_p",
        ):
            v = r[key]
            results[f"{k}_{key}"] = int(v) if key == "tb_rank" else float(v)
        results[f"{k}_best_config"] = str(r["best_config"])
        results[f"{k}_start"] = str(r["start"])
    for tk, f in full.items():
        results[f"{LABEL[tk].lower()}_full_history_start"] = f["start"]
        results[f"{LABEL[tk].lower()}_full_history_tb_sharpe"] = f["sharpe"]
    results["btc_ema12_128_sharpe"] = float(bt[0]["sharpe"])
    results["btc_ema50_200_sharpe"] = float(bt[1]["sharpe"])
    results["btc_ema12_128_grid_percentile"] = float(bt[0]["grid_percentile"])
    results["btc_ema50_200_p_beats_bnh_n2"] = float(bt[1]["p_beats_bnh_n2"])

    after = (
        "PBO and DSR answer different questions: DSR asks whether the *selected* "
        "Sharpe is too high to be the best of N lucky draws; PBO asks whether the "
        "*selection procedure* (pick the in-sample best) tends to pick an "
        "out-of-sample loser. A flat plateau can pass the first and fail the second: "
        "when all configurations are near-ties, ranking them is noise."
    )
    return Section(
        name=NAME,
        title=TITLE,
        text=text,
        figures=figures,
        tables=tables,
        after=after,
        results=results,
    )

"""Experiment 4: walk-forward optimisation versus a fixed, a-priori rule.

Experiment 3 showed that the best in-sample (fast, slow) pair is mostly
selected noise. The practical question is whether *re-optimising* on a rolling
window adds anything a trader could actually have earned. On each fold the
EMA-crossover configuration with the best Sharpe over the previous five years
is frozen and traded for the next year (:func:`trendlab.validation.walk_forward`,
leakage-free with a one-bar embargo for next-close fills). The stitched
out-of-sample (OOS) path is compared with the textbook EMA50/200, fixed before
looking at any data, and with buy-and-hold over exactly the same OOS dates.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

from .. import metrics, plotting, stats, validation
from ..plotting import _date_axis, _titles, save_figure, theme
from ..report import df_to_markdown
from ..strategies import EMACrossover
from .common import Context, Section
from .e09_significance import holm_adjust

NAME = "e04_walk_forward"
TITLE = "4. Walk-forward: does re-optimising the windows beat a fixed textbook rule?"

ASSETS = ("SPY", "QQQ", "BTC-USD", "GLD")
LABEL = {"SPY": "SPY", "QQQ": "QQQ", "BTC-USD": "BTC", "GLD": "GLD"}
FAST = [5, 10, 20, 30, 40, 50, 75, 100, 150]
SLOW = [50, 75, 100, 150, 200, 250, 300, 400]
FAST_QUICK = [10, 20, 50, 100]
SLOW_QUICK = [100, 200, 300]
TRAIN_YEARS, TEST_YEARS = 5, 1
TEXTBOOK = EMACrossover(50, 200)
WF_LABEL, STATIC_LABEL, BNH_LABEL = "Walk-forward", "Static EMA50/200", "Buy & hold"


def _fmt_df(df: pd.DataFrame, pct=(), ints=(), prob=(), digits: int = 2) -> pd.DataFrame:
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


def _save(ctx: Context, name: str, raw: pd.DataFrame, display: pd.DataFrame) -> None:
    """Raw numbers to CSV via ``ctx.save_table``; the Markdown copy gets the display version."""
    ctx.save_table(name, raw)
    (ctx.table_dir / f"{name}.md").write_text(df_to_markdown(display) + "\n")


def _fold_ranks(wf: validation.WalkForwardResult) -> pd.DataFrame:
    """Per fold: OOS Sharpe percentile of the chosen config among all candidates, the
    static 50/200's OOS Sharpe, and whether the walk-forward pick beat it (NaN on ties).

    A candidate that sits in T-bills for the whole test year has zero excess return and
    an undefined Sharpe; it is scored 0, which is what holding cash earns.
    """
    cand, p = wf.candidate_returns, wf.periods_per_year
    out = []
    for row in wf.folds.itertuples():
        seg = cand.loc[row.test_start : row.test_end]
        rf = wf.rf_oos.loc[row.test_start : row.test_end]
        sh = pd.Series({c: metrics.sharpe(seg[c], rf, p) for c in seg.columns}).fillna(0.0)
        mine = sh[row.chosen]
        same = np.allclose(seg[row.chosen].to_numpy(), seg[TEXTBOOK.name].to_numpy())
        out.append(
            {
                "oos_percentile": float((sh < mine).mean() + 0.5 * (sh == mine).mean()),
                "static_oos_sharpe": float(sh[TEXTBOOK.name]),
                "beats_static": np.nan if same else float(mine > sh[TEXTBOOK.name]),
            }
        )
    return pd.DataFrame(out, index=wf.folds.index)


def _p_worse(boot: dict) -> float:
    """One-sided p for H0: ``diff >= 0`` (WF *not* worse), the mirror of ``boot["p_value"]``."""
    d = boot["diff"]
    fin = boot["samples"][np.isfinite(boot["samples"])]
    if len(fin) == 0 or not np.isfinite(d):
        return float("nan")
    return float((1.0 + np.sum(fin - d <= d)) / (len(fin) + 1.0))


def _holm_rows(rows: list[dict]) -> None:
    """Holm-adjust the one-sided p-values (both directions) across every walk-forward test."""
    for key in ("sharpe_diff_p", "sharpe_diff_p_worse"):
        adj = holm_adjust([r[key] for r in rows])
        for r, a in zip(rows, adj, strict=True):
            r[f"{key}_holm"] = float(a)


def _tradeoffs(R: dict, key: str) -> str:
    """Group assets by how a strategy's CAGR and max drawdown compare with buy-and-hold."""
    groups: dict[str, list[str]] = {}
    for tk, r in R.items():
        more = r[f"{key}_cagr"] > r["bnh_cagr"]
        shallower = r[f"{key}_max_drawdown"] > r["bnh_max_drawdown"]
        what = {
            (False, True): "trades return for a shallower drawdown",
            (True, True): "out-earns it with a shallower drawdown",
            (False, False): "earns less with a deeper drawdown",
            (True, False): "earns more with a deeper drawdown",
        }[(more, shallower)]
        groups.setdefault(what, []).append(LABEL[tk])
    parts = [f"{what} ({', '.join(a)})" for what, a in groups.items()]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _study(ctx: Context, ticker: str, grid, anchored: bool, n_boot: int) -> dict:
    prices, cfg = ctx.prices(ticker), ctx.config(ticker)
    wf = validation.walk_forward(
        prices,
        grid,
        cfg,
        ctx.rf,
        train_years=TRAIN_YEARS,
        test_years=TEST_YEARS,
        objective="sharpe",
        anchored=anchored,
    )
    p = wf.periods_per_year
    idx = wf.oos_returns.index
    rf = wf.rf_oos
    series = {
        WF_LABEL: wf.oos_returns,
        STATIC_LABEL: wf.candidate_returns[TEXTBOOK.name].reindex(idx),
        BNH_LABEL: wf.benchmark_oos,
    }
    f = wf.folds
    fr = _fold_ranks(wf)
    pct = fr["oos_percentile"]
    switches = int((f["chosen"] != f["chosen"].shift()).iloc[1:].sum())
    boot = stats.bootstrap_sharpe_difference(
        series[WF_LABEL], series[STATIC_LABEL], rf=rf, n_boot=n_boot, periods=p
    )
    row = {
        "oos_start": idx[0].date().isoformat(),
        "oos_end": idx[-1].date().isoformat(),
        "oos_years": (len(idx) - 1) / p,
        "n_folds": len(f),
        "n_switches": switches,
        "switch_rate": switches / max(len(f) - 1, 1),
        "n_unique": int(f["chosen"].nunique()),
        "tb_chosen": int((f["chosen"] == TEXTBOOK.name).sum()),
        "wfe": wf.walk_forward_efficiency,
        "mean_is_sharpe": float(f["is_objective"].mean()),
        "mean_oos_sharpe_fold": float(f["oos_objective"].mean()),
        "mean_oos_percentile": float(pct.mean()),
        "share_folds_above_median": float((pct > 0.5).mean()),
        "share_folds_beat_static": float(fr["beats_static"].mean()),
        "switch_cost_total": float(f["switch_cost"].sum()),
        "sharpe_diff_vs_static": boot["diff"],
        "sharpe_diff_ci_low": boot["ci_low"],
        "sharpe_diff_ci_high": boot["ci_high"],
        "sharpe_diff_p": boot["p_value"],
        "sharpe_diff_p_worse": _p_worse(boot),
    }
    for lbl, key in ((WF_LABEL, "wf"), (STATIC_LABEL, "static"), (BNH_LABEL, "bnh")):
        r = series[lbl]
        row[f"{key}_sharpe"] = metrics.sharpe(r, rf, p)
        row[f"{key}_cagr"] = metrics.cagr(r, p)
        row[f"{key}_max_drawdown"] = metrics.max_drawdown(r)
    folds = f.join(fr)
    return {"wf": wf, "row": row, "series": series, "folds": folds}


# --------------------------------------------------------------------------- figures


def _signed(v: float) -> str:
    return f"{v:+.2f}".replace("-", "\u2212")


def _static_end_label(ax, eq: pd.Series, others: list[float], t: dict) -> None:
    """Dot + multiple at the static line's end, nudged vertically if it would collide."""
    x, y = eq.index[-1], float(eq.iloc[-1])
    lo, hi = (np.log(v) for v in ax.get_ylim())
    frac = (np.log(y) - lo) / (hi - lo)
    near = [
        (np.log(o) - lo) / (hi - lo)
        for o in others
        if abs((np.log(o) - lo) / (hi - lo) - frac) < 0.05
    ]
    dy = 0 if not near else (-11 if frac <= near[0] else 11)
    color = t["categorical"][1]
    ax.plot([x], [y], "o", ms=7, color=color, mec=t["surface"], mew=2, zorder=5)
    ax.annotate(
        f"{y:.2f}\u00d7",
        xy=(x, y),
        xytext=(7, dy),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=t["ink"],
        annotation_clip=False,
    )


def _wf_chart(st: dict, ticker: str, path) -> None:
    """Library walk-forward chart plus the static 50/200 over the same OOS dates."""
    row = st["row"]
    wf_eq = (1.0 + st["series"][WF_LABEL]).cumprod()
    bh_eq = (1.0 + st["series"][BNH_LABEL]).cumprod()
    st_eq = (1.0 + st["series"][STATIC_LABEL]).cumprod()
    lo, hi = row["sharpe_diff_ci_low"], row["sharpe_diff_ci_high"]
    noise = lo <= 0 <= hi
    holm_sig = min(row["sharpe_diff_p_holm"], row["sharpe_diff_p_worse_holm"]) < 0.05
    if row["wf_sharpe"] > row["static_sharpe"]:
        verb = "edges" if noise or not holm_sig else "beats"
    else:
        verb = "trails"
    if noise:
        gap = ", a gap within noise"
    elif holm_sig:
        gap = ", a significant gap"
    else:
        gap = ", a gap not significant after Holm"
    title = (
        f"{LABEL[ticker]}: yearly re-optimisation {verb} the fixed 50/200 "
        f"(Sharpe {row['wf_sharpe']:.2f} vs {row['static_sharpe']:.2f})" + gap
    )
    sub = (
        f"Each year trades the EMA pair with the best Sharpe over the prior {TRAIN_YEARS} years; "
        f"labels = chosen fast/slow. {row['n_folds']} folds, {row['oos_start']} to "
        f"{row['oos_end']}.\nB&H Sharpe {row['bnh_sharpe']:.2f}. Paired-bootstrap 95% CI of "
        f"the Sharpe gap (walk-forward minus 50/200): {_signed(lo)} to {_signed(hi)}."
    )
    with theme() as t:
        fig = plotting.walk_forward_chart(
            wf_eq,
            bh_eq,
            st["folds"],
            title,
            subtitle=sub,
            strategy_label=f"{WF_LABEL} (out of sample)",
            bnh_label=BNH_LABEL,
        )
        ax = fig.axes[0]
        ax.plot(st_eq.index, st_eq.to_numpy(), color=t["categorical"][1], lw=1.5, zorder=1.5)
        ax.lines[-1].set_label(STATIC_LABEL)
        handles = [ln for ln in ax.lines if ln.get_label() and not ln.get_label().startswith("_")]
        ax.legend(handles=handles, loc="lower right")
        ax.set_ylim(top=ax.get_ylim()[1] * (1.6 if row["n_folds"] > 10 else 1.35))
        _static_end_label(ax, st_eq, [wf_eq.iloc[-1], bh_eq.iloc[-1]], t)
        save_figure(fig, path)


def _param_path_chart(studies: dict, path, title: str, subtitle: str) -> None:
    """Chosen fast and slow window per fold, one panel per asset (log scale)."""
    with theme() as t:
        n = len(studies)
        fig = plt.figure(figsize=(10, 1.9 * n + 1.4))
        gs = fig.add_gridspec(n + 1, 1, height_ratios=[0.02] + [1.0] * n)
        hdr = fig.add_subplot(gs[0, 0])
        hdr.axis("off")
        _titles(hdr, title, subtitle)
        first = None
        cats = t["categorical"]
        ticks = [5, 10, 20, 50, 100, 200, 400]
        for k, (tk, st) in enumerate(studies.items()):
            ax = fig.add_subplot(gs[k + 1, 0], sharex=first)
            first = first or ax
            f = st["folds"]
            x = list(pd.to_datetime(f["test_start"])) + [pd.Timestamp(f["test_end"].iloc[-1])]
            for col, c, lbl in (("fast", cats[0], "Chosen fast"), ("slow", cats[1], "Chosen slow")):
                y = list(f[col].astype(float)) + [float(f[col].iloc[-1])]
                ax.step(x, y, where="post", color=c, lw=2.0, label=lbl)
            for ref in (50, 200):
                ax.axhline(ref, color=t["muted"], lw=0.9, ls=":", zorder=1)
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_ylim(4, 500)
            r = st["row"]
            ax.set_title(
                f"{LABEL[tk]}: {r['n_unique']} distinct pairs in {r['n_folds']} folds, "
                f"{r['n_switches']} switches, 50/200 chosen {r['tb_chosen']}×",
                loc="left",
            )
            ax.set_ylabel("EMA window")
            if k == 0:
                ax.legend(loc="upper left", ncols=2)
            if k < n - 1:
                ax.tick_params(axis="x", labelbottom=False)
        _date_axis(first)
        save_figure(fig, path)


# --------------------------------------------------------------------------- run


def run(ctx: Context) -> Section:
    fast, slow = (FAST_QUICK, SLOW_QUICK) if ctx.quick else (FAST, SLOW)
    grid = validation.parameter_grid(fast, slow)
    n_boot = 300 if ctx.quick else 2000
    studies = {tk: _study(ctx, tk, grid, False, n_boot) for tk in ASSETS}
    anchored = {} if ctx.quick else {tk: _study(ctx, tk, grid, True, n_boot) for tk in ASSETS}
    all_rows = [st["row"] for st in studies.values()] + [st["row"] for st in anchored.values()]
    _holm_rows(all_rows)
    n_tests = len(all_rows)

    rows = {LABEL[tk]: st["row"] for tk, st in studies.items()}
    rows.update({f"{LABEL[tk]} (anchored)": st["row"] for tk, st in anchored.items()})
    summary = pd.DataFrame(rows).T
    summary.index.name = "asset"
    folds_all = pd.concat(
        {LABEL[tk]: st["folds"] for tk, st in studies.items()}, names=["asset", "fold"]
    )
    folds_disp = _fmt_df(
        folds_all.drop(columns=["train_start", "train_end"])
        .assign(
            test_start=lambda d: d["test_start"].dt.date.astype(str),
            test_end=lambda d: d["test_end"].dt.date.astype(str),
            switch_cost=lambda d: d["switch_cost"] * 1e4,
        )
        .rename(columns={"switch_cost": "switch_cost_bp"})
        .reset_index(),
        pct=("is_ann_return", "oos_ann_return", "is_ann_excess", "oos_ann_excess")
        + ("oos_percentile",),
        ints=("fold", "fast", "slow", "beats_static"),
        digits=2,
    )
    _save(ctx, f"{NAME}_folds", folds_all, folds_disp.set_index("asset"))

    # ---- figures
    figures = []
    for tk in ("SPY", "BTC-USD"):
        p = ctx.fig_path(f"{NAME}_equity_{LABEL[tk].lower()}")
        _wf_chart(studies[tk], tk, p)
        figures.append((f"{LABEL[tk]}: stitched walk-forward equity vs static 50/200 and B&H.", p))

    p = ctx.fig_path(f"{NAME}_param_path")
    total_sw = sum(st["row"]["n_switches"] for st in studies.values())
    total_f = sum(st["row"]["n_folds"] - 1 for st in studies.values())
    _param_path_chart(
        studies,
        p,
        f"The 'optimal' windows change at {total_sw} of {total_f} fold boundaries",
        f"Fast and slow EMA chosen on each {TRAIN_YEARS}-year training window "
        f"(rolling, best Sharpe of {len(grid)} pairs); dotted lines at 50 and 200.",
    )
    figures.append(("Parameter path: the chosen (fast, slow) on each fold.", p))

    bars = pd.DataFrame(
        {
            WF_LABEL: {LABEL[tk]: st["row"]["wf_sharpe"] for tk, st in studies.items()},
            STATIC_LABEL: {LABEL[tk]: st["row"]["static_sharpe"] for tk, st in studies.items()},
            BNH_LABEL: {LABEL[tk]: st["row"]["bnh_sharpe"] for tk, st in studies.items()},
        }
    )
    n_beat = int((bars[WF_LABEL] > bars[STATIC_LABEL]).sum())
    n_sig = sum(st["row"]["sharpe_diff_p_holm"] < 0.05 for st in studies.values())
    p = ctx.fig_path(f"{NAME}_oos_sharpe")
    plotting.bar_comparison(
        bars,
        f"Walk-forward beats the fixed 50/200 on {n_beat} of {len(bars)} assets out of sample, "
        f"{'none' if n_sig == 0 else n_sig} significantly",
        subtitle="Sharpe ratio over identical out-of-sample dates (each asset's walk-forward "
        "test windows).\nSignificance: one-sided paired stationary bootstrap of the Sharpe gap, "
        f"Holm-adjusted across {n_tests} walk-forward tests, 5%.",
        percent=False,
        fmt="{:.2f}",
        xlabel="Out-of-sample Sharpe ratio",
        path=p,
    )
    plt.close("all")
    figures.append(("Out-of-sample Sharpe: walk-forward vs static 50/200 vs buy-and-hold.", p))

    # ---- tables
    cols = {
        "oos_start": "OOS start",
        "n_folds": "Folds",
        "wf_sharpe": "WF Sharpe",
        "static_sharpe": "50/200 Sharpe",
        "bnh_sharpe": "B&H Sharpe",
        "wf_cagr": "WF CAGR",
        "static_cagr": "50/200 CAGR",
        "bnh_cagr": "B&H CAGR",
        "wf_max_drawdown": "WF MaxDD",
        "static_max_drawdown": "50/200 MaxDD",
        "bnh_max_drawdown": "B&H MaxDD",
    }
    disp = _fmt_df(
        summary[list(cols)],
        pct=("wf_cagr", "static_cagr", "bnh_cagr", "wf_max_drawdown", "static_max_drawdown")
        + ("bnh_max_drawdown",),
        ints=("n_folds",),
    ).rename(columns=cols)
    cols2 = {
        "wfe": "WFE",
        "mean_is_sharpe": "Mean IS Sharpe",
        "mean_oos_sharpe_fold": "Mean OOS Sharpe (fold)",
        "mean_oos_percentile": "Chosen OOS percentile",
        "share_folds_beat_static": "Folds WF > 50/200",
        "n_switches": "Switches",
        "n_unique": "Distinct pairs",
        "sharpe_diff_vs_static": "Sharpe diff vs 50/200",
        "sharpe_diff_ci_low": "95% CI low",
        "sharpe_diff_ci_high": "95% CI high",
        "sharpe_diff_p": "p (WF > 50/200)",
        "sharpe_diff_p_holm": "Holm p (WF > 50/200)",
        "sharpe_diff_p_worse": "p (WF < 50/200)",
        "sharpe_diff_p_worse_holm": "Holm p (WF < 50/200)",
    }
    disp2 = _fmt_df(
        summary[list(cols2)],
        pct=("mean_oos_percentile", "share_folds_beat_static"),
        ints=("n_switches", "n_unique"),
        prob=(
            "sharpe_diff_p",
            "sharpe_diff_p_holm",
            "sharpe_diff_p_worse",
            "sharpe_diff_p_worse_holm",
        ),
    ).rename(columns=cols2)
    tables = [
        (
            "Out-of-sample performance over identical dates (rolling 5y/1y; anchored as a check).",
            disp,
        ),
        (
            "Selection diagnostics and one-sided paired stationary-bootstrap tests of WF minus "
            f"50/200 in both directions (Holm: adjusted across all {n_tests} walk-forward tests).",
            disp2,
        ),
    ]
    _save(ctx, f"{NAME}_summary", summary, pd.concat([disp, disp2], axis=1))

    # ---- text
    R = {tk: st["row"] for tk, st in studies.items()}
    beat = [LABEL[tk] for tk, r in R.items() if r["wf_sharpe"] > r["static_sharpe"]]
    trail = [LABEL[tk] for tk, r in R.items() if r["wf_sharpe"] <= r["static_sharpe"]]
    sig = [LABEL[tk] for tk, r in R.items() if r["sharpe_diff_p_holm"] < 0.05]
    sig_raw = [
        LABEL[tk] for tk, r in R.items() if r["sharpe_diff_p"] < 0.05 and LABEL[tk] not in sig
    ]
    wfe = {LABEL[tk]: r["wfe"] for tk, r in R.items()}
    pct = [r["mean_oos_percentile"] for r in R.values()]
    fb = [r["share_folds_beat_static"] for r in R.values()]
    mean_diff = float(np.mean([r["sharpe_diff_vs_static"] for r in R.values()]))
    vs_bh = [LABEL[tk] for tk, r in R.items() if r["wf_sharpe"] > r["bnh_sharpe"] + 0.01]
    low_ret = [LABEL[tk] for tk, r in R.items() if r["wf_cagr"] < r["bnh_cagr"]]
    labelled = [(LABEL[tk], st["row"]) for tk, st in studies.items()] + [
        (f"{LABEL[tk]} anchored", st["row"]) for tk, st in anchored.items()
    ]
    worse_adj = [
        f"{lbl} (one-sided p = {r['sharpe_diff_p_worse']:.3f}, Holm {r['sharpe_diff_p_worse_holm']:.3f})"
        for lbl, r in labelled
        if r["sharpe_diff_p_worse_holm"] < 0.05
    ]
    worse_raw = [
        f"{lbl} (one-sided p = {r['sharpe_diff_p_worse']:.3f})"
        for lbl, r in labelled
        if r["sharpe_diff_p_worse"] < 0.05 and r["sharpe_diff_p_worse_holm"] >= 0.05
    ]
    worse_txt = ""
    if worse_adj:
        worse_txt += (
            f"; it is significantly *worse* on {', '.join(worse_adj)}, after Holm across the "
            f"{n_tests} walk-forward tests"
        )
    if worse_raw:
        worse_txt += (
            f"; it is worse at the uncorrected 5% level on {', '.join(worse_raw)}, but not "
            f"significantly after Holm across the {n_tests} walk-forward tests"
        )

    def _line(tk: str) -> str:
        r = R[tk]
        return (
            f"{LABEL[tk]} ({r['oos_start'][:4]}-{r['oos_end'][:4]}, {r['n_folds']} folds): "
            f"WF {r['wf_sharpe']:.2f} / {r['wf_cagr']:.1%} / {r['wf_max_drawdown']:.0%}, "
            f"50/200 {r['static_sharpe']:.2f} / {r['static_cagr']:.1%} / "
            f"{r['static_max_drawdown']:.0%}, B&H {r['bnh_sharpe']:.2f} / {r['bnh_cagr']:.1%} / "
            f"{r['bnh_max_drawdown']:.0%}"
        )

    anch_txt = ""
    if anchored:
        a_beat = [
            LABEL[tk]
            for tk, st in anchored.items()
            if st["row"]["wf_sharpe"] > st["row"]["static_sharpe"]
        ]
        anch_txt = (
            f" An anchored (expanding) training window beats 50/200 on "
            f"{len(a_beat)} of {len(anchored)} ({', '.join(a_beat) if a_beat else 'none'})."
        )
    ci_txt = "; ".join(
        f"{LABEL[tk]} {r['sharpe_diff_ci_low']:+.2f} to {r['sharpe_diff_ci_high']:+.2f}"
        for tk, r in R.items()
    )
    trade_txt = (
        f"Against buy-and-hold, walk-forward {_tradeoffs(R, 'wf')}; the static rule "
        f"{_tradeoffs(R, 'static')}."
    )
    wfe_txt = ", ".join(f"{k} {v:.2f}" if np.isfinite(v) else f"{k} n/a" for k, v in wfe.items())
    verdict = (
        "No asset shows a significant gain from re-optimising"
        if not sig
        else f"Only {', '.join(sig)} shows a gain significant at 5%"
    ) + (
        f" after Holm adjustment across the {n_tests} walk-forward tests"
        + (f" ({', '.join(sig_raw)} only before adjustment)" if sig_raw else "")
    )

    text = f"""**Question.** Does re-optimising the EMA windows every year on the past {TRAIN_YEARS} years beat simply fixing 50/200 in advance, on dates neither has seen?

**Setup.** Rolling walk-forward over the {len(grid)}-pair grid: each fold picks the best-Sharpe pair on {TRAIN_YEARS} years, then trades it for {TEST_YEARS} year, with a one-bar embargo and switching costs at the fold boundary. The static 50/200 and buy-and-hold are measured over exactly the same OOS dates; triples below are Sharpe / CAGR / MaxDD.

**Findings.**

- {_line("SPY")}.
- {_line("QQQ")}.
- {_line("BTC-USD")}.
- {_line("GLD")}.
- Walk-forward beats the fixed rule on {len(beat)} of {len(R)} assets ({", ".join(beat) if beat else "none"}) and trails on {", ".join(trail) if trail else "none"}; the average Sharpe difference is {mean_diff:+.2f}. {verdict} (one-sided paired stationary bootstrap; 95% CIs of the gap: {ci_txt}){worse_txt}.{anch_txt}
- The chosen pair changes at {total_sw} of {total_f} fold boundaries. Its OOS Sharpe ranks, on average, at the {min(pct):.0%}-{max(pct):.0%} percentile of all {len(grid)} candidates in the same test year (50% = a random pick), and it beats 50/200 in {min(fb):.0%}-{max(fb):.0%} of the test years where the two differ. Walk-forward efficiency: {wfe_txt}.

**Interpretation.** Walk-forward analysis (Pardo 2008) removes the look-ahead in the *choice* of parameters, the step the README's Bitcoin re-tuning (12/128, then 50/200) skipped, but it cannot create information that is not there. Experiment 3 showed how little the in-sample ranking of these rules carries out of sample. Here the yearly re-selection {"lands close to the middle of the candidate pack" if 0.35 <= np.mean(pct) <= 0.65 else ("lands in the upper part of the pack" if np.mean(pct) > 0.65 else "lands in the lower part of the pack")} out of sample while hopping between near-equivalent rules. The out-of-sample edge of optimisation over a rule chosen a priori is {("negative on average" if mean_diff <= 0 else "small") + " and statistically indistinguishable from zero" if not sig else "present only on " + ", ".join(sig)}. This is one reason practitioners fix and diversify lookbacks rather than tune them: Hurst, Ooi & Pedersen (2017) average 1-, 3- and 12-month signals across a century of data. Optimisation is a source of selection bias (Bailey et al. 2017), and the fair benchmark for a tuned rule is the untuned one. If anything the test favours the static rule, whose "a priori" status is partly earned ex post: 50/200 became the textbook rule by working on past US equity data (Sullivan, Timmermann & White 1999). Walk-forward beats buy-and-hold's Sharpe by more than 0.01 on {len(vs_bh)} of {len(R)} assets{" (" + ", ".join(vs_bh) + ")" if vs_bh else ""}, yet earns a lower CAGR on {len(low_ret)} ({", ".join(low_ret) if low_ret else "none"}). {trade_txt} BTC has only {R["BTC-USD"]["oos_years"]:.1f} OOS years ({R["BTC-USD"]["n_folds"]} folds) after the slow-EMA warm-up and the {TRAIN_YEARS}-year training window, so its numbers are anecdotes.
"""

    results = {
        "n_configs": len(grid),
        "train_years": TRAIN_YEARS,
        "test_years": TEST_YEARS,
        "n_beat_static": len(beat),
        "n_assets": len(R),
        "total_switches": total_sw,
        "total_boundaries": total_f,
        "mean_sharpe_diff_vs_static": mean_diff,
        "n_walk_forward_tests": n_tests,
    }
    keys = (
        "wf_sharpe",
        "static_sharpe",
        "bnh_sharpe",
        "wf_cagr",
        "static_cagr",
        "bnh_cagr",
        "wf_max_drawdown",
        "static_max_drawdown",
        "bnh_max_drawdown",
        "wfe",
        "n_folds",
        "n_switches",
        "mean_oos_percentile",
        "sharpe_diff_vs_static",
        "sharpe_diff_p",
        "sharpe_diff_p_holm",
        "sharpe_diff_p_worse",
        "sharpe_diff_p_worse_holm",
    )
    for tk, r in R.items():
        for k in keys:
            v = r[k]
            results[f"{LABEL[tk].lower()}_{k}"] = (
                int(v) if k in ("n_folds", "n_switches") else float(v)
            )
        results[f"{LABEL[tk].lower()}_oos_start"] = r["oos_start"]
    for tk, st in anchored.items():
        results[f"{LABEL[tk].lower()}_anchored_wf_sharpe"] = float(st["row"]["wf_sharpe"])
        for k in ("sharpe_diff_p_worse", "sharpe_diff_p_worse_holm"):
            results[f"{LABEL[tk].lower()}_anchored_{k}"] = float(st["row"][k])

    after = (
        "Walk-forward efficiency here is the ratio of mean OOS to mean in-sample excess return "
        "of the chosen configurations. Values near or above 1 do not mean the optimiser is "
        "skilled: when every candidate earns roughly the same trend premium, OOS and IS returns "
        "of *whatever* is chosen are similar, so WFE measures the stability of the premium, not "
        "the value of choosing."
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

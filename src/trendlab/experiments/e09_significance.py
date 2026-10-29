"""Experiment 9: is any edge of EMA 50/200 over buy-and-hold statistically real?

For every asset in the fixed universe the EMA 50/200 rule (default research
settings: next-close fills, per-side costs, T-bill cash) is compared with
buy-and-hold over the same window using

* the probabilistic Sharpe ratio and minimum track record length (Bailey &
  Lopez de Prado 2012) against zero and against buy-and-hold's Sharpe,
* a paired stationary-bootstrap test of the Sharpe difference (Politis &
  Romano 1994; Ledoit & Wolf 2008),
* a random-entry Monte Carlo that keeps the rule's trades but randomises their
  timing,
* Holm (1979) and Benjamini-Hochberg (1995) corrections across the 12 assets,
* Lo's (2002) autocorrelation-adjusted Sharpe ratio,
* a paired stationary-bootstrap test of the *risk* claim: the differences in
  maximum drawdown and in volatility.
"""

from __future__ import annotations

import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import binomtest

from .. import metrics, plotting, stats
from ..strategies import EMACrossover
from ..universe import ASSETS
from .common import Context, Section

NAME = "e09_significance"
TITLE = "9. Is any edge statistically real? Sharpe inference, timing tests and multiple testing"

MEAN_BLOCK = 20.0  # stationary-bootstrap mean block length in bars (~1 trading month)
ALPHA = 0.05
NOTEBOOK_WINDOW = ("2022-01-01", "2024-12-31")  # the original notebook's NVDA sample
_RISK_CHUNK = 250  # bootstrap paths per vectorised drawdown chunk
SENS_BLOCKS = (MEAN_BLOCK, 60.0, 125.0, 250.0)  # drawdown-test block lengths (1, 3, 6, 12 months)


# --------------------------------------------------------------------------- multiple testing


def holm_adjust(p) -> np.ndarray:
    """Holm (1979) step-down family-wise-error adjusted p-values (NaNs ignored)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = len(ok)
    if m == 0:
        return out
    order = ok[np.argsort(p[ok])]
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out[order] = np.minimum(adj, 1.0)
    return out


def bh_adjust(p) -> np.ndarray:
    """Benjamini-Hochberg (1995) false-discovery-rate q-values (NaNs ignored)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = len(ok)
    if m == 0:
        return out
    order = ok[np.argsort(p[ok])]
    raw = p[order] * m / np.arange(1, m + 1)
    out[order] = np.minimum(np.minimum.accumulate(raw[::-1])[::-1], 1.0)
    return out


# --------------------------------------------------------------------------- risk bootstrap


def _mdd_rows(x: np.ndarray) -> np.ndarray:
    """Max drawdown of each row of simple returns (same convention as metrics.max_drawdown)."""
    eq = np.cumprod(1.0 + x, axis=1)
    peak = np.maximum(np.maximum.accumulate(eq, axis=1), 1.0)
    return (eq / peak - 1.0).min(axis=1)


def paired_risk_bootstrap(
    r_a: pd.Series,
    r_b: pd.Series,
    n_boot: int,
    mean_block: float = MEAN_BLOCK,
    seed: int = 0,
    periods: int = 252,
) -> dict:
    """Paired stationary bootstrap of max-drawdown and volatility differences (a - b).

    Both series are resampled with the same index paths (from
    :func:`stats.stationary_bootstrap_indices`), as in
    :func:`stats.bootstrap_sharpe_difference`. ``mdd_diff > 0`` means ``a`` has the
    shallower drawdown; ``vol_diff < 0`` means ``a`` is less volatile.

    The bootstrap distribution of a drawdown difference is skewed and not
    centred on the full-sample estimate (resampling blocks breaks long trends),
    so the one-sided p-value is the *percentile* one, ``P*(diff <= 0)`` for
    drawdown and ``P*(diff >= 0)`` for volatility, consistent with the
    percentile CI. ``mdd_boot_median`` reports how far the resampled
    differences sit from the actual one.
    """
    a = r_a.to_numpy(dtype=float)
    b = r_b.reindex(r_a.index).to_numpy(dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    mdd_a, mdd_b = metrics.max_drawdown(pd.Series(a)), metrics.max_drawdown(pd.Series(b))
    assert abs(_mdd_rows(a[None, :])[0] - mdd_a) < 1e-10  # vectorised == library definition
    scale = math.sqrt(periods)
    vol_a, vol_b = a.std(ddof=1) * scale, b.std(ddof=1) * scale
    d_mdd, d_vol = [], []
    done, k = 0, 0
    while done < n_boot:
        m = min(_RISK_CHUNK, n_boot - done)
        idx = stats.stationary_bootstrap_indices(n, mean_block, m, seed=seed * 10_000 + k)
        xa, xb = a[idx], b[idx]
        d_mdd.append(_mdd_rows(xa) - _mdd_rows(xb))
        d_vol.append((xa.std(axis=1, ddof=1) - xb.std(axis=1, ddof=1)) * scale)
        done += m
        k += 1
    s_mdd, s_vol = np.concatenate(d_mdd), np.concatenate(d_vol)
    diff_mdd, diff_vol = mdd_a - mdd_b, vol_a - vol_b
    lo_m, hi_m = np.quantile(s_mdd, [ALPHA / 2, 1 - ALPHA / 2])
    lo_v, hi_v = np.quantile(s_vol, [ALPHA / 2, 1 - ALPHA / 2])
    return {
        "mdd_a": float(mdd_a),
        "mdd_b": float(mdd_b),
        "mdd_diff": float(diff_mdd),
        "mdd_ci_low": float(lo_m),
        "mdd_ci_high": float(hi_m),
        "mdd_boot_median": float(np.median(s_mdd)),
        # H0: drawdown not shallower (diff <= 0)
        "mdd_p": float((1 + np.sum(s_mdd <= 0.0)) / (len(s_mdd) + 1)),
        # centred (null-shifted) version, as in stats.bootstrap_sharpe_difference
        "mdd_p_centred": float((1 + np.sum(s_mdd - diff_mdd >= diff_mdd)) / (len(s_mdd) + 1)),
        "vol_a": float(vol_a),
        "vol_b": float(vol_b),
        "vol_diff": float(diff_vol),
        "vol_ci_low": float(lo_v),
        "vol_ci_high": float(hi_v),
        # H0: volatility not lower (diff >= 0)
        "vol_p": float((1 + np.sum(s_vol >= 0.0)) / (len(s_vol) + 1)),
        "mdd_samples": s_mdd,
    }


# --------------------------------------------------------------------------- per asset


def _analyse(ctx: Context, ticker: str, seed: int, n_boot: int, n_sims: int) -> tuple[dict, dict]:
    res, bnh = ctx.pair(ticker, EMACrossover(50, 200))
    p = res.periods_per_year
    rs, rb, rf = res.returns, bnh.returns, res.rf
    sr_s = stats.sharpe_per_period(rs, rf)
    sr_b = stats.sharpe_per_period(rb, rf)
    boot = stats.bootstrap_sharpe_difference(
        rs, rb, rf=rf, n_boot=n_boot, mean_block=MEAN_BLOCK, seed=seed, periods=p, alpha=ALPHA
    )
    rnd = stats.random_entry_test_result(res, n_sims=n_sims, seed=seed)
    risk = paired_risk_bootstrap(rs, rb, n_boot, MEAN_BLOCK, seed, p)
    # constant mix with the rule's average exposure (rest in T-bills, no trading costs):
    # how much of the risk reduction is just being invested less of the time?
    e = float(res.exposure.mean())
    mix = e * res.asset_returns + (1.0 - e) * rf
    vs_mix = paired_risk_bootstrap(rs, mix, n_boot, MEAN_BLOCK, seed, p)
    row = {
        "start": res.start.date().isoformat(),
        "years": len(rs) / p,
        "exposure": float(res.exposure.mean()),
        "corr_excess": float(np.corrcoef(rs - rf, rb - rf)[0, 1]),
        "sharpe_strat": float(stats.annualize_sharpe(sr_s, p)),
        "sharpe_bnh": float(stats.annualize_sharpe(sr_b, p)),
        "sharpe_diff": boot["diff"],
        "diff_ci_low": boot["ci_low"],
        "diff_ci_high": boot["ci_high"],
        "p_boot": boot["p_value"],
        "psr_vs_0": stats.probabilistic_sharpe_ratio(rs, 0.0, rf),
        "psr_vs_bnh": stats.probabilistic_sharpe_ratio(rs, sr_b, rf),
        "mintrl_vs_0_years": stats.min_track_record_length(rs, 0.0, ALPHA, rf) / p,
        "mintrl_vs_bnh_years": stats.min_track_record_length(rs, sr_b, ALPHA, rf) / p,
        "random_null_mean": rnd["null_mean"],
        "p_random": rnd["p_value"],
        "n_trades": rnd["n_segments"],
        "lo_sharpe_strat": stats.lo_adjusted_sharpe(rs, rf, p),
        "lo_sharpe_bnh": stats.lo_adjusted_sharpe(rb, rf, p),
        **{k: v for k, v in risk.items() if k != "mdd_samples"},
        "mdd_mix": vs_mix["mdd_b"],
        "vol_mix": vs_mix["vol_b"],
        "mdd_diff_vs_mix": vs_mix["mdd_diff"],
        "mdd_vs_mix_ci_low": vs_mix["mdd_ci_low"],
        "mdd_vs_mix_ci_high": vs_mix["mdd_ci_high"],
        "mdd_vs_mix_p": vs_mix["mdd_p"],
    }
    extra = {"random": rnd, "risk": risk, "res": res, "bnh": bnh}
    return row, extra


def _notebook_window(ctx: Context, n_boot: int) -> dict:
    """The original notebook's sample: NVDA 2022-2024 (indicators warmed up on prior data)."""
    start, end = NOTEBOOK_WINDOW
    res, bnh = ctx.pair("NVDA", EMACrossover(50, 200), start=start, end=end)
    p = res.periods_per_year
    sr_b = stats.sharpe_per_period(bnh.returns, res.rf)
    boot = stats.bootstrap_sharpe_difference(
        res.returns, bnh.returns, rf=res.rf, n_boot=n_boot, mean_block=MEAN_BLOCK, periods=p
    )
    return {
        "years": len(res.returns) / p,
        "total_strat": metrics.total_return(res.returns),
        "total_bnh": metrics.total_return(bnh.returns),
        "sharpe_strat": boot["sharpe_a"],
        "sharpe_bnh": boot["sharpe_b"],
        "diff": boot["diff"],
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "p": boot["p_value"],
        "psr_vs_0": stats.probabilistic_sharpe_ratio(res.returns, 0.0, res.rf),
        "mintrl_vs_bnh_years": stats.min_track_record_length(res.returns, sr_b, ALPHA, res.rf) / p,
    }


# --------------------------------------------------------------------------- figures


def forest_plot(df: pd.DataFrame, path, title: str, subtitle: str) -> None:
    """Two-panel forest plot: Sharpe difference and max-drawdown difference with 95% CIs."""
    panels = [
        ("sharpe_diff", "diff_ci_low", "diff_ci_high", "q_boot_bh", False,
         "Sharpe difference (EMA − B&H, annualised)"),
        ("mdd_diff", "mdd_ci_low", "mdd_ci_high", "q_mdd_bh", True,
         "Max-drawdown difference (EMA − B&H; > 0 = shallower)"),
    ]  # fmt: skip
    with plotting.theme() as t:
        n = len(df)
        fig = plt.figure(figsize=(11, 0.42 * n + 2.6))
        # a zero-height header axis spanning both panels carries the title and subtitle,
        # so the long subtitle does not squeeze the right-hand panel
        gs = fig.add_gridspec(2, 2, height_ratios=[0.001, 1.0])
        header = fig.add_subplot(gs[0, :])
        header.axis("off")
        ax0 = fig.add_subplot(gs[1, 0])
        axes = [ax0, fig.add_subplot(gs[1, 1], sharey=ax0)]
        y = np.arange(n)
        blue = t["categorical"][0]
        for ax, (est, lo, hi, q, pct, xlabel) in zip(axes, panels, strict=True):
            sig = (df[q] < ALPHA).to_numpy()
            ax.hlines(y, df[lo], df[hi], color=blue, lw=1.8, zorder=3)
            ax.scatter(df[est][sig], y[sig], s=46, color=blue, zorder=4, edgecolor=t["surface"])
            ax.scatter(
                df[est][~sig], y[~sig], s=46, facecolor=t["surface"], edgecolor=blue,
                linewidth=1.6, zorder=4,
            )  # fmt: skip
            ax.axvline(0.0, color=t["ink2"], lw=1.0, zorder=2)
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", visible=True)
            ax.tick_params(axis="both", length=0)
            ax.margins(x=0.06)
            ax.set_xlabel(xlabel)
            if pct:
                ax.xaxis.set_major_formatter(
                    plt.FuncFormatter(lambda v, _: f"{v:.0%}".replace("-", "−"))
                )
            else:
                ax.xaxis.set_major_formatter(
                    plt.FuncFormatter(lambda v, _: f"{v:.1f}".replace("-", "−"))
                )
        axes[0].set_yticks(y, list(df.index))
        axes[0].invert_yaxis()
        handles = [
            Line2D([], [], marker="o", ls="", color=blue, mec=t["surface"], ms=7,
                   label=f"BH q < {ALPHA:.2f}"),
            Line2D([], [], marker="o", ls="", mfc=t["surface"], mec=blue, mew=1.6, ms=7,
                   label="Not significant after BH"),
            Line2D([], [], color=blue, lw=1.8, label="95% stationary-bootstrap CI"),
        ]  # fmt: skip
        fig.legend(handles=handles, loc="outside lower center", ncols=3)
        axes[1].tick_params(axis="y", labelleft=False)
        plotting._titles(header, title, subtitle)
        plotting.save_figure(fig, path)


# --------------------------------------------------------------------------- formatting


def _pct(v: float, d: int = 1) -> str:
    return "–" if not np.isfinite(v) else f"{v:.{d}%}"


def _num(v: float, d: int = 2) -> str:
    return "–" if not np.isfinite(v) else f"{v:.{d}f}"


def _p(v: float) -> str:
    if not np.isfinite(v):
        return "–"
    return f"{v:.3f}" if v >= 0.001 else "<0.001"


def _prob(v: float) -> str:
    """A probability near 1 (PSR) without rounding 0.9995 up to a misleading 1.000."""
    if not np.isfinite(v):
        return "–"
    return f"{v:.3f}" if v < 0.9995 else ">0.999"


def _peq(v: float) -> str:
    """'= 0.123' or '< 0.001', for running text."""
    t = _p(v)
    return "< 0.001" if t.startswith("<") else f"= {t}"


def _yrs(v: float) -> str:
    if not np.isfinite(v):
        return "∞"
    return f"{v:,.0f}" if v >= 10 else f"{v:.1f}"


def _ci(lo: float, hi: float, pct: bool = False) -> str:
    f = (lambda v: f"{v:.0%}") if pct else (lambda v: f"{v:.2f}")
    return f"[{f(lo)}, {f(hi)}]"


def _ci_col(df: pd.DataFrame, lo: str, hi: str, pct: bool = False) -> list[str]:
    return [_ci(a, b, pct) for a, b in zip(df[lo], df[hi], strict=True)]


def _display_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sharpe = pd.DataFrame(
        {
            "Years": df["years"].map(lambda v: f"{v:.1f}"),
            "Sharpe EMA": df["sharpe_strat"].map(_num),
            "Sharpe B&H": df["sharpe_bnh"].map(_num),
            "Diff": df["sharpe_diff"].map(_num),
            "95% CI": _ci_col(df, "diff_ci_low", "diff_ci_high"),
            "p (boot)": df["p_boot"].map(_p),
            "p Holm": df["p_boot_holm"].map(_p),
            "q BH": df["q_boot_bh"].map(_p),
            "PSR(SR>0)": df["psr_vs_0"].map(_prob),
            "PSR(SR>B&H)": df["psr_vs_bnh"].map(_prob),
            "MinTRL vs B&H (yrs)": df["mintrl_vs_bnh_years"].map(_yrs),
        }
    )
    timing = pd.DataFrame(
        {
            "Exposure": df["exposure"].map(_pct),
            "Trades": df["n_trades"].map(lambda v: f"{int(v)}"),
            "Sharpe EMA": df["sharpe_strat"].map(_num),
            "Random-timing mean": df["random_null_mean"].map(_num),
            "p (random)": df["p_random"].map(_p),
            "p Holm": df["p_random_holm"].map(_p),
            "q BH": df["q_random_bh"].map(_p),
            "Lo Sharpe EMA": df["lo_sharpe_strat"].map(_num),
            "Lo Sharpe B&H": df["lo_sharpe_bnh"].map(_num),
        }
    )
    risk = pd.DataFrame(
        {
            "MaxDD EMA": df["mdd_a"].map(_pct),
            "MaxDD B&H": df["mdd_b"].map(_pct),
            "Diff vs B&H": df["mdd_diff"].map(_pct),
            "95% CI": _ci_col(df, "mdd_ci_low", "mdd_ci_high", pct=True),
            "p": df["mdd_p"].map(_p),
            "q BH": df["q_mdd_bh"].map(_p),
            "MaxDD mix": df["mdd_mix"].map(_pct),
            "Diff vs mix": df["mdd_diff_vs_mix"].map(_pct),
            "q BH (vs mix)": df["q_mdd_mix_bh"].map(_p),
            "Vol EMA": df["vol_a"].map(_pct),
            "Vol B&H": df["vol_b"].map(_pct),
            "q BH (vol)": df["q_vol_bh"].map(_p),
        }
    )
    for d in (sharpe, timing, risk):
        d.index.name = "Asset"
    return sharpe, timing, risk


# --------------------------------------------------------------------------- run


def run(ctx: Context) -> Section:
    n_boot = 500 if ctx.quick else 5000
    n_sims = 500 if ctx.quick else 5000
    tickers = list(ASSETS)
    rows, extras = {}, {}
    for i, tk in enumerate(tickers):
        rows[tk], extras[tk] = _analyse(ctx, tk, seed=i, n_boot=n_boot, n_sims=n_sims)
    df = pd.DataFrame(rows).T
    df = df.astype({c: float for c in df.columns if c != "start"})
    df["p_boot_holm"] = holm_adjust(df["p_boot"])
    df["q_boot_bh"] = bh_adjust(df["p_boot"])
    df["p_random_holm"] = holm_adjust(df["p_random"])
    df["q_random_bh"] = bh_adjust(df["p_random"])
    df["q_mdd_bh"] = bh_adjust(df["mdd_p"])
    df["q_vol_bh"] = bh_adjust(df["vol_p"])
    df["q_mdd_mix_bh"] = bh_adjust(df["mdd_vs_mix_p"])
    df.index.name = "asset"
    ctx.save_table(f"{NAME}_raw", df)

    sharpe_tab, timing_tab, risk_tab = _display_tables(df)
    ctx.save_table(f"{NAME}_sharpe", sharpe_tab)
    ctx.save_table(f"{NAME}_timing", timing_tab)
    ctx.save_table(f"{NAME}_risk", risk_tab)

    # ---- block-length sensitivity of the drawdown test: max drawdown is a path statistic,
    # so short blocks break up the long bear markets the rule sidesteps
    n_sens = 200 if ctx.quick else 1000
    sens = {}
    for blk in SENS_BLOCKS:
        if blk == MEAN_BLOCK:
            p_pct, p_cen, bmed, nb_used = (
                df["mdd_p"],
                df["mdd_p_centred"],
                df["mdd_boot_median"],
                n_boot,
            )
        else:
            rr = {
                tk: paired_risk_bootstrap(
                    extras[tk]["res"].returns, extras[tk]["bnh"].returns, n_sens, blk, i,
                    extras[tk]["res"].periods_per_year,
                )
                for i, tk in enumerate(tickers)
            }  # fmt: skip
            p_pct = pd.Series({k: v["mdd_p"] for k, v in rr.items()})
            p_cen = pd.Series({k: v["mdd_p_centred"] for k, v in rr.items()})
            bmed = pd.Series({k: v["mdd_boot_median"] for k, v in rr.items()})
            nb_used = n_sens
        sens[blk] = {
            "n_boot": nb_used,
            "median_boot_gap": float(bmed.median()),
            "median_actual_gap": float(df["mdd_diff"].median()),
            "n_sig_bh_percentile": int((bh_adjust(p_pct) < ALPHA).sum()),
            "n_sig_bh_centred": int((bh_adjust(p_cen) < ALPHA).sum()),
            "spy_p_percentile": float(p_pct["SPY"]),
            "spy_p_centred": float(p_cen["SPY"]),
        }
    sens_df = pd.DataFrame(sens).T
    sens_df.index.name = "mean_block_days"
    ctx.save_table(f"{NAME}_mdd_blocks_raw", sens_df)
    sens_tab = pd.DataFrame(
        {
            "Resamples": sens_df["n_boot"].map(lambda v: f"{int(v):,}"),
            "Median resampled gap": sens_df["median_boot_gap"].map(_pct),
            "Median actual gap": sens_df["median_actual_gap"].map(_pct),
            "Sig. after BH (percentile p)": sens_df["n_sig_bh_percentile"].map(
                lambda v: f"{int(v)} of {len(df)}"
            ),
            "Sig. after BH (centred p)": sens_df["n_sig_bh_centred"].map(
                lambda v: f"{int(v)} of {len(df)}"
            ),
            "SPY p (percentile)": sens_df["spy_p_percentile"].map(_p),
            "SPY p (centred)": sens_df["spy_p_centred"].map(_p),
        },
    )
    sens_tab.index = [f"{b:.0f}" for b in sens_df.index]
    sens_tab.index.name = "Mean block (days)"
    ctx.save_table(f"{NAME}_mdd_blocks", sens_tab)

    nb = _notebook_window(ctx, n_boot)
    spy = df.loc["SPY"]
    spy_rnd = extras["SPY"]["random"]

    # marginal (unpaired) SPY drawdown CIs via the library's generic bootstrap
    n_mdd = 300 if ctx.quick else 2000
    spy_res, spy_bnh = extras["SPY"]["res"], extras["SPY"]["bnh"]
    mdd_ci_s = stats.bootstrap_metric_ci(
        spy_res.returns, "max_drawdown", n_boot=n_mdd, mean_block=MEAN_BLOCK
    )
    mdd_ci_b = stats.bootstrap_metric_ci(
        spy_bnh.returns, "max_drawdown", n_boot=n_mdd, mean_block=MEAN_BLOCK
    )

    # ---- headline aggregates
    m = len(df)
    n_pos = int((df["sharpe_diff"] > 0).sum())
    n_raw = int((df["p_boot"] < ALPHA).sum())
    n_holm = int((df["p_boot_holm"] < ALPHA).sum())
    n_bh = int((df["q_boot_bh"] < ALPHA).sum())
    n_ci_excl = int((df["diff_ci_low"] > 0).sum())
    n_rnd_raw = int((df["p_random"] < ALPHA).sum())
    n_rnd_bh = int((df["q_random_bh"] < ALPHA).sum())
    n_psr0 = int((df["psr_vs_0"] > 1 - ALPHA).sum())
    n_psrb = int((df["psr_vs_bnh"] > 1 - ALPHA).sum())
    n_mdd_bh = int((df["q_mdd_bh"] < ALPHA).sum())
    n_mdd_shallower = int((df["mdd_diff"] > 0).sum())
    n_vol_bh = int((df["q_vol_bh"] < ALPHA).sum())
    n_mix_better = int((df["mdd_diff_vs_mix"] > 0).sum())
    n_mix_bh = int((df["q_mdd_mix_bh"] < ALPHA).sum())
    med_mdd_diff = float(df["mdd_diff"].median())
    # share of the drawdown gap vs B&H that a constant exposure-matched mix already delivers
    mix_share = float((1.0 - df["mdd_diff_vs_mix"] / df["mdd_diff"]).median())
    mintrl_b = df["mintrl_vs_bnh_years"]
    fin_mintrl = mintrl_b[np.isfinite(mintrl_b)]
    n_inf_mintrl = int((~np.isfinite(mintrl_b)).sum())
    med_mintrl = float(fin_mintrl.median()) if len(fin_mintrl) else float("inf")
    med_years = float(df["years"].median())
    med_ci_width = float((df["diff_ci_high"] - df["diff_ci_low"]).median())
    med_abs_diff = float(df["sharpe_diff"].abs().median())
    lo_ratio = df["lo_sharpe_strat"] / df["sharpe_strat"]
    lo_ratio_b = df["lo_sharpe_bnh"] / df["sharpe_bnh"]
    med_lo = float(lo_ratio.median())
    med_lo_b = float(lo_ratio_b.median())
    lo_diff = df["lo_sharpe_strat"] - df["lo_sharpe_bnh"]
    n_lo_flip = int((np.sign(lo_diff) != np.sign(df["sharpe_diff"])).sum())
    best = df["sharpe_diff"].idxmax()

    # ---- figures
    forest_path = ctx.fig_path(f"{NAME}_forest")
    forest_title = (
        f"{n_bh} of {m} Sharpe gains survive multiple testing; "
        f"drawdowns are shallower on {n_mdd_shallower} of {m}, "
        f"{'none' if n_mdd_bh == 0 else n_mdd_bh} significantly"
    )
    forest_plot(
        df,
        forest_path,
        forest_title,
        "EMA 50/200 vs buy-and-hold, full history, next-close fills, costs and T-bill cash. "
        f"Paired stationary bootstrap, mean block {MEAN_BLOCK:.0f} days, {n_boot:,} resamples.",
    )
    hist_path = ctx.fig_path(f"{NAME}_spy_random_entry")
    beats = spy["sharpe_strat"] > spy["random_null_mean"]
    fig = plotting.histogram_with_marker(
        spy_rnd["null_sharpes"],
        spy["sharpe_strat"],
        (
            f"SPY: rule timing {'beats' if beats else 'trails'} the random-timing average, "
            f"{'significantly' if spy['p_random'] < ALPHA else 'but not significantly'}"
        ),
        subtitle=(
            f"Same {int(spy['n_trades'])} trades and holding periods, shuffled in time "
            f"({n_sims:,} simulations, same costs). Annualised excess Sharpe."
        ),
        xlabel="Sharpe ratio of randomly timed strategies",
        actual_label="EMA 50/200",
        p_value=spy["p_random"],
        path=None,
    )
    ax = fig.axes[0]
    with plotting.theme() as t:
        dil = math.sqrt(spy["exposure"]) * spy["sharpe_bnh"]
        for x, ls, lab in ((spy["sharpe_bnh"], "--", "B&H"), (dil, ":", "√exposure × B&H")):
            ax.axvline(x, color=t["ink2"], lw=1.4, ls=ls, zorder=3)
            ax.annotate(
                f"{lab} {x:.2f}", xy=(x, 0.97 if ls == "--" else 0.91),
                xycoords=("data", "axes fraction"), xytext=(-5, 0),
                textcoords="offset points", ha="right", va="center", fontsize=9,
                color=t["ink2"],
            )  # fmt: skip
    plotting.save_figure(fig, hist_path)
    plt.close("all")

    # ---- text
    def word(cond: bool, yes: str, no: str) -> str:
        return yes if cond else no

    spy_sig = spy["p_boot"] < ALPHA
    if n_raw == 0:
        sig_clause = "none is significant at 5% even before correction"
    else:
        sig_clause = (
            f"only {n_raw} is significant at 5% before correction and "
            f"{n_holm} (Holm) / {n_bh} (BH) after"
        )
    if med_lo > 1 and med_lo_b > 1:
        lo_clause = "higher because short-lag autocorrelation of daily returns is negative"
    elif med_lo < 1 and med_lo_b < 1:
        lo_clause = "lower because short-lag autocorrelation of daily returns is positive"
    else:
        lo_clause = "moving the two legs in opposite directions"
    flip_clause = (
        "no sign of the difference flips"
        if n_lo_flip == 0
        else f"the sign of the difference flips on {n_lo_flip} asset(s)"
    )
    risk_robust = n_mdd_bh >= m / 2
    n_mdd_centred_bh = int((bh_adjust(df["mdd_p_centred"]) < ALPHA).sum())
    sign_p = float(binomtest(n_mdd_shallower, m, 0.5, alternative="greater").pvalue)
    lit = (
        "the long-only, single-asset counterpart of Faber's (2007) moving-average timing "
        "result and of the crisis behaviour Hurst, Ooi & Pedersen (2017) document for "
        "diversified trend following"
    )
    if risk_robust:
        risk_text = (
            "What does survive is the risk profile: stepping aside during prolonged declines "
            f"cuts drawdowns, {lit}."
        )
    else:
        risk_text = (
            "The risk claim holds in *direction*, not per-asset significance: shallower "
            f"drawdowns on {n_mdd_shallower} of {m} assets (sign-test p {_peq(sign_p)} if "
            f"assets were independent, which they are not) are {lit}, but a max drawdown "
            "rests on one or two episodes, so single-asset tests rarely reject."
        )
    # random timing at exposure e: mean scales by e, volatility by ~sqrt(e)
    null_ratio = float(
        (df["random_null_mean"] / (np.sqrt(df["exposure"]) * df["sharpe_bnh"])).median()
    )
    n_beat_null = int((df["sharpe_strat"] > df["random_null_mean"]).sum())
    med_corr = float(df["corr_excess"].median())
    se_diff = med_ci_width / (2 * 1.959964)
    # block-length sensitivity of the drawdown test
    blk_str = "/".join(f"{b:.0f}" for b in SENS_BLOCKS)
    pct_counts = "/".join(str(sens[b]["n_sig_bh_percentile"]) for b in SENS_BLOCKS)
    cen_counts = [sens[b]["n_sig_bh_centred"] for b in SENS_BLOCKS]
    gap_first = sens[SENS_BLOCKS[0]]["median_boot_gap"]
    gap_last = sens[SENS_BLOCKS[-1]]["median_boot_gap"]
    if cen_counts[-1] < cen_counts[0]:
        cen_txt = (
            f"but lengthening the blocks from {SENS_BLOCKS[0]:.0f} to {SENS_BLOCKS[-1]:.0f} "
            f"days moves the median resampled gap from {_pct(gap_first, 0)} to "
            f"{_pct(gap_last, 0)} (actual {_pct(med_mdd_diff, 0)}) and cuts the centred "
            f"test's rejections from {cen_counts[0]} to {cen_counts[-1]}, while the percentile "
            f"test gives {pct_counts} rejections across {blk_str}-day blocks. The centred "
            "test's apparent significance therefore mostly reflects the resampling bias, "
            "which is why we report the percentile version."
        )
    else:
        cen_txt = (
            f"and across {blk_str}-day blocks the centred test gives "
            f"{'/'.join(map(str, cen_counts))} rejections against {pct_counts} for the "
            "percentile test; we report the more conservative percentile version."
        )
    psr0_txt = "> 0.999" if spy["psr_vs_0"] >= 0.9995 else f"= {spy['psr_vs_0']:.3f}"
    yrs_word = word(np.isfinite(med_mintrl) and med_mintrl >= 100, "centuries", "years")
    text = f"""**Question.** After accounting for sampling noise, the rule's trade structure and testing {m} assets at once, is there reliable evidence that EMA 50/200 beats buy-and-hold?

**Setup.** Full history per asset (median {med_years:.0f} years), default research settings. Paired stationary bootstrap ({MEAN_BLOCK:.0f}-day mean blocks, {n_boot:,} resamples) for Sharpe and drawdown differences; random-entry Monte Carlo ({n_sims:,} shuffles of the rule's own trades) for timing; Holm and Benjamini-Hochberg (BH) corrections across the {m} assets.

**Findings.**

- The rule has the higher Sharpe on {n_pos} of {m} assets, but {sig_clause}. The median 95% CI is {med_ci_width:.2f} Sharpe units wide against a median absolute difference of {med_abs_diff:.2f}. The largest gap ({best}, {df.loc[best, "sharpe_diff"]:+.2f}) has p {_peq(df.loc[best, "p_boot"])}, BH q {_peq(df.loc[best, "q_boot_bh"])}.
- SPY: Sharpe {spy["sharpe_strat"]:.2f} vs {spy["sharpe_bnh"]:.2f}, difference CI {_ci(spy["diff_ci_low"], spy["diff_ci_high"])}, p {_peq(spy["p_boot"])} ({word(spy_sig, "significant", "not significant")}). PSR: P(SR > 0) {psr0_txt}, but P(SR > B&H) = {spy["psr_vs_bnh"]:.2f}; MinTRL to confirm the gap is {_yrs(spy["mintrl_vs_bnh_years"])} years (median finite {_yrs(med_mintrl)}; infinite on the {n_inf_mintrl} assets where the rule trails).
- Timing: the rule beats the random-timing mean on {n_beat_null} of {m} assets, but random entry is rejected at 5% on only {n_rnd_raw}, {n_rnd_bh} after BH (SPY {spy["sharpe_strat"]:.2f} vs {spy["random_null_mean"]:.2f}, p {_peq(spy["p_random"])}).
- Lo (2002) Sharpes are a median {med_lo:.2f}x (rule) and {med_lo_b:.2f}x (B&H) the naive ones, {lo_clause}; {flip_clause}.
- Risk: max drawdown is shallower on {n_mdd_shallower} of {m} assets (SPY {_pct(spy["mdd_a"], 0)} vs {_pct(spy["mdd_b"], 0)}, difference CI {_ci(spy["mdd_ci_low"], spy["mdd_ci_high"], True)}), significant after BH on {n_mdd_bh} ({pct_counts} with {blk_str}-day blocks). Volatility is significantly lower on {n_vol_bh} of {m}. Against a static mix with the same average exposure, drawdowns are shallower on {n_mix_better} of {m}, {n_mix_bh} significantly.
- The notebook's own sample (NVDA {NOTEBOOK_WINDOW[0][:4]}-{NOTEBOOK_WINDOW[1][:4]}, {nb["years"]:.1f} years): Sharpe difference {nb["diff"]:+.2f}, CI {_ci(nb["ci_low"], nb["ci_high"])}, p {_peq(nb["p"])}, MinTRL {_yrs(nb["mintrl_vs_bnh_years"])} years.

**Interpretation.** Even with the two return streams {med_corr:.2f} correlated, the paired bootstrap leaves a standard error of about {se_diff:.2f} on the Sharpe difference, so gaps of 0.1-0.2 are noise; MinTRL (Bailey & Lopez de Prado 2012) converts that into {yrs_word}. Quoting the best of {m} assets is the data-snooping trap of Sullivan, Timmermann & White (1999), and Holm/BH are the minimum haircut (Harvey & Liu 2015). Random in-and-out timing at exposure e earns about √e times B&H's Sharpe (median ratio {null_ratio:.2f}), because the mean scales with e and volatility with √e. Beating that null is a lower bar than beating B&H, and it is cleared significantly on {word(n_rnd_bh == 0, "no asset", f"only {n_rnd_bh} asset(s)")} after correction. {risk_text} Lower volatility is close to mechanical, and an exposure-matched static mix delivers a median {_pct(mix_share, 0)} of the drawdown gap with no timing at all. The defensible claim is "similar risk-adjusted return, smaller drawdowns", not "alpha".
"""
    if n_holm == n_bh == n_rnd_bh == n_mdd_bh == 0:
        dep_note = "With no BH rejections in the Sharpe, timing or drawdown tests, the choice between Holm and BH does not change any conclusion here."
    else:
        dep_note = "Where Holm and BH disagree, Holm is the safe reading."
    after = (
        "**Method notes.** p-values are one-sided (H0: the rule is no better). Holm controls "
        f"the probability of any false rejection across the {m} assets under any dependence. "
        "Benjamini-Hochberg controls the expected share of false discoveries and is valid "
        "under independence or positive regression dependence (Benjamini & Yekutieli 2001). "
        "Equity tests are positively dependent, but bond and equity tests need not be. "
        f"{dep_note} PSR against B&H treats B&H's Sharpe as a known constant, so it overstates "
        "confidence relative to the paired bootstrap, which resamples both legs together. The "
        "Sharpe p-value centres the bootstrap distribution on zero. The drawdown distribution "
        "is skewed and sits below the full-sample estimate, which is why the forest plot's "
        "drawdown points sit near the top of their CIs. The drawdown and volatility p-values "
        "are therefore percentile ones (the bootstrap share of differences on the wrong side "
        f"of zero), consistent with the CIs. A centred test would be significant after BH on "
        f"{n_mdd_centred_bh} of {m} assets (SPY p {_peq(spy['mdd_p_centred'])}), "
        f"{cen_txt} The block-length rows use {n_sens:,} resamples. Max drawdown is a single-path extreme, effectively "
        "one or two independent episodes per asset, so treat its p-values as indicative. The "
        "exposure-matched mix holds the rule's average exposure in the asset and the rest in "
        "T-bills, rebalanced daily without costs. For SPY the unpaired bootstrap 95% intervals "
        f"for max drawdown (`bootstrap_metric_ci`, {n_mdd:,} resamples) are "
        f"{_ci(mdd_ci_s['ci_low'], mdd_ci_s['ci_high'], True)} for the rule and "
        f"{_ci(mdd_ci_b['ci_low'], mdd_ci_b['ci_high'], True)} for B&H. They "
        f"{word(mdd_ci_s['ci_low'] > mdd_ci_b['ci_high'], 'do not overlap', 'overlap')}. "
        "Comparing marginal intervals ignores the strong correlation between the two return "
        "streams, which is why the paired test is the right one. The universe is not free of "
        "hindsight: NVDA is included because the notebook picked it after its run-up, and "
        "Bitcoin is an extreme survivor. That inflates both legs, one reason to test "
        "differences rather than levels."
    )

    results = {
        "n_assets": m,
        "n_boot": n_boot,
        "n_sims": n_sims,
        "n_sharpe_diff_positive": n_pos,
        "n_sharpe_sig_raw": n_raw,
        "n_sharpe_sig_holm": n_holm,
        "n_sharpe_sig_bh": n_bh,
        "n_sharpe_ci_excludes_zero": n_ci_excl,
        "n_psr_vs_0_above_95": n_psr0,
        "n_psr_vs_bnh_above_95": n_psrb,
        "n_random_entry_sig_raw": n_rnd_raw,
        "n_random_entry_sig_bh": n_rnd_bh,
        "n_mdd_shallower": n_mdd_shallower,
        "n_mdd_sig_bh": n_mdd_bh,
        "n_vol_sig_bh": n_vol_bh,
        "median_mintrl_vs_bnh_years": med_mintrl,
        "median_ci_width": med_ci_width,
        "median_lo_ratio_strat": med_lo,
        "median_lo_ratio_bnh": med_lo_b,
        "best_asset": str(best),
        "best_sharpe_diff": float(df.loc[best, "sharpe_diff"]),
        "best_q_bh": float(df.loc[best, "q_boot_bh"]),
        "spy_sharpe_strat": float(spy["sharpe_strat"]),
        "spy_sharpe_bnh": float(spy["sharpe_bnh"]),
        "spy_sharpe_diff": float(spy["sharpe_diff"]),
        "spy_diff_ci_low": float(spy["diff_ci_low"]),
        "spy_diff_ci_high": float(spy["diff_ci_high"]),
        "spy_p_boot": float(spy["p_boot"]),
        "spy_psr_vs_0": float(spy["psr_vs_0"]),
        "spy_psr_vs_bnh": float(spy["psr_vs_bnh"]),
        "spy_mintrl_vs_bnh_years": float(spy["mintrl_vs_bnh_years"]),
        "spy_p_random": float(spy["p_random"]),
        "spy_random_null_mean": float(spy["random_null_mean"]),
        "spy_lo_sharpe_strat": float(spy["lo_sharpe_strat"]),
        "spy_mdd_strat": float(spy["mdd_a"]),
        "spy_mdd_bnh": float(spy["mdd_b"]),
        "spy_mdd_diff": float(spy["mdd_diff"]),
        "spy_mdd_diff_ci_low": float(spy["mdd_ci_low"]),
        "spy_mdd_diff_ci_high": float(spy["mdd_ci_high"]),
        "spy_mdd_p": float(spy["mdd_p"]),
        "spy_mdd_p_centred": float(spy["mdd_p_centred"]),
        "n_mdd_sig_bh_centred": n_mdd_centred_bh,
        **{
            f"mdd_block{b:.0f}_n_sig_bh_{kind}": int(sens[b][f"n_sig_bh_{kind}"])
            for b in SENS_BLOCKS
            for kind in ("percentile", "centred")
        },
        "n_beat_random_mean": n_beat_null,
        "median_random_null_over_sqrt_exposure_bnh": null_ratio,
        "median_corr_strat_bnh": med_corr,
        "median_se_sharpe_diff": se_diff,
        "mdd_sign_test_p": sign_p,
        "n_mdd_shallower_than_mix": n_mix_better,
        "n_mdd_vs_mix_sig_bh": n_mix_bh,
        "median_mdd_gap_share_from_exposure": mix_share,
        "notebook_nvda_sharpe_diff": nb["diff"],
        "notebook_nvda_ci_low": nb["ci_low"],
        "notebook_nvda_ci_high": nb["ci_high"],
        "notebook_nvda_p": nb["p"],
        "notebook_nvda_mintrl_vs_bnh_years": nb["mintrl_vs_bnh_years"],
    }

    return Section(
        name=NAME,
        title=TITLE,
        text=text,
        figures=[
            (
                "Paired-bootstrap 95% CIs for the Sharpe and max-drawdown differences, "
                "EMA 50/200 minus buy-and-hold. Filled markers survive Benjamini-Hochberg at 5%. "
                "Drawdown points sit near the top of their CIs because resampling breaks up the "
                "long bear markets that create the gap.",
                forest_path,
            ),
            (
                "SPY random-entry null: Sharpe ratios of the rule's own trades placed at random "
                "times, with the actual rule, buy-and-hold and the sqrt(exposure)-diluted "
                "buy-and-hold Sharpe marked.",
                hist_path,
            ),
        ],
        tables=[
            (
                "Sharpe inference: bootstrap, multiple-testing corrections, PSR and MinTRL",
                sharpe_tab,
            ),
            (
                "Timing (random-entry test) and Lo (2002) autocorrelation-adjusted Sharpe",
                timing_tab,
            ),
            (
                "Risk claim: paired-bootstrap tests of max-drawdown and volatility reduction",
                risk_tab,
            ),
            (
                "Drawdown test vs bootstrap block length: percentile vs centred p-values",
                sens_tab,
            ),
        ],
        after=after,
        results=results,
    )


if __name__ == "__main__":  # pragma: no cover
    t0 = time.time()
    s = run(Context())
    print(s.text)
    print(s.results)
    print(f"{time.time() - t0:.1f}s")

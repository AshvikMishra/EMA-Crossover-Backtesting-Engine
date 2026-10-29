"""Experiment 10: execution and trading-cost realism.

Four checks on whether the backtest's frictions are plausible and whether the
conclusions depend on them:

1. **Spreads.** Corwin & Schultz (2012) high-low spread estimates per asset,
   next to the one-tick floor (one cent over the last price) and the cost
   assumed in the research defaults.
2. **Cost sweep.** Sharpe and CAGR of EMA50/200 at 0-100 bp per side for five
   representative assets (``costs.cost_sensitivity``).
3. **Break-even cost.** The per-side cost at which the rule's edge disappears:
   arithmetic excess return over T-bills and over buy-and-hold
   (``costs.breakeven_from_result``, exact), and the cost at which the rule's
   CAGR / Sharpe falls to buy-and-hold's (root-found on full backtests).
4. **Execution timing.** Same-close vs next-open vs next-close fills, i.e. how
   much the optimistic "trade on the close you just observed" convention of the
   original notebook is worth -- for the slow 50/200 rule and, for contrast, a
   fast 5/20 rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .. import costs, metrics, plotting
from ..engine import CostModel, run_backtest
from ..plotting import _minus
from ..report import df_to_markdown
from ..strategies import BuyAndHold, EMACrossover
from ..universe import ASSETS
from .common import Context, Section, costs_for

NAME = "e10_costs"
TITLE = "10. Execution and cost realism: does friction change the verdict?"

STRATEGY = EMACrossover(50, 200)
FAST = EMACrossover(5, 20)
SWEEP_ASSETS = ["SPY", "QQQ", "GLD", "BTC-USD", "NVDA"]
BPS_GRID = (0, 2, 5, 10, 20, 50, 100)
BPS_GRID_QUICK = (0, 5, 20, 100)
EXECUTIONS = ("same_close", "next_open", "next_close")
EXEC_LABELS = {"same_close": "Same close", "next_open": "Next open", "next_close": "Next close"}
MAX_BPS = 2000.0  # root search ceiling: 20% of notional per side
RECENT_YEARS = 5
TICK = 0.01  # US equity/ETF minimum price increment, USD
NB_WINDOW = ("2022-01-01", "2024-12-31")  # the original notebook's NVDA sample


def _save(ctx: Context, name: str, raw: pd.DataFrame, display: pd.DataFrame) -> None:
    """Raw numbers to CSV; the formatted copy replaces the Markdown version."""
    ctx.save_table(name, raw)
    (ctx.table_dir / f"{name}.md").write_text(df_to_markdown(display) + "\n")


def _bps(v: float, digits: int = 1) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:,.{digits}f}"


# --------------------------------------------------------------------------- 1. spreads


def _spreads(ctx: Context, tickers: list[str]) -> pd.DataFrame:
    rows = {}
    for t in tickers:
        p = ctx.prices(t)
        recent = p.loc[p.index[-1] - pd.DateOffset(years=RECENT_YEARS) :]
        crypto = ASSETS[t]["class"] == "Crypto"
        rows[t] = {
            "class": ASSETS[t]["class"],
            "cs_robust_bps": costs.median_spread_bps(p),
            "cs_daily_clip_bps": costs.median_spread_bps(p, clip_daily=True),
            "cs_daily_clip_recent_bps": costs.median_spread_bps(recent, clip_daily=True),
            "tick_floor_bps": np.nan if crypto else TICK / float(p["Close"].iloc[-1]) * 1e4,
            "assumed_per_side_bps": costs_for(t).bps,
            "assumed_half_spread_bps": costs_for(t).half_spread_bps,
        }
    return pd.DataFrame(rows).T


def _spreads_display(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out.index.name = "Asset"
    out["Class"] = df["class"]
    out["CS spread, robust (bp)"] = df["cs_robust_bps"].map(_bps)
    out["CS spread, daily-clipped (bp)"] = df["cs_daily_clip_bps"].map(_bps)
    out[f"Daily-clipped, last {RECENT_YEARS}y (bp)"] = df["cs_daily_clip_recent_bps"].map(_bps)
    out["One-tick spread (bp)"] = df["tick_floor_bps"].map(lambda v: _bps(v, 2))
    out["Assumed half-spread (bp)"] = df["assumed_half_spread_bps"].map(_bps)
    out["Assumed cost per side (bp)"] = df["assumed_per_side_bps"].map(_bps)
    return out


# --------------------------------------------------------------------------- 2-3. sweep, break-even


def _solve_bps(f) -> float:
    """Smallest per-side cost where the decreasing gap ``f(bps)`` hits zero.

    Returns a negative sentinel (-1) when there is no edge even at zero cost,
    and +inf when the edge survives ``MAX_BPS``.
    """
    lo = f(0.0)
    if not np.isfinite(lo):
        return float("nan")
    if lo <= 0:
        return -1.0
    if f(MAX_BPS) > 0:
        return float("inf")
    return float(brentq(f, 0.0, MAX_BPS, xtol=0.05))


def _breakeven(ctx: Context, ticker: str) -> dict:
    prices = ctx.prices(ticker)
    target = STRATEGY.target_position(prices)
    zero = ctx.backtest(ticker, STRATEGY, costs=CostModel.zero())
    bnh = ctx.backtest(ticker, BuyAndHold(), start=zero.start)
    p = zero.periods_per_year
    bnh_cagr = metrics.cagr(bnh.returns, p)
    bnh_sharpe = metrics.sharpe(bnh.returns, bnh.rf, p)

    cache: dict[float, tuple[float, float]] = {}

    def at(bps: float) -> tuple[float, float]:
        if bps not in cache:
            cfg = ctx.config(ticker, costs=CostModel.flat(bps))
            r = run_backtest(prices, STRATEGY, cfg, ctx.rf, target=target)
            cache[bps] = (metrics.cagr(r.returns, p), metrics.sharpe(r.returns, r.rf, p))
        return cache[bps]

    vs_cash = costs.breakeven_from_result(zero)
    vs_bnh = costs.breakeven_from_result(zero, benchmark=bnh.returns)
    default = ctx.backtest(ticker, STRATEGY)
    return {
        "class": ASSETS[ticker]["class"],
        "turnover_per_year": vs_cash["turnover_per_year"],
        "round_trips_per_year": vs_cash["turnover_per_year"] / 2.0,
        "default_bps": costs_for(ticker).bps,
        "default_drag_bps_per_year": float(default.costs.mean() * p * 1e4),
        "edge_vs_cash": vs_cash["edge_annual"],
        "breakeven_vs_cash_bps": vs_cash["breakeven_bps"],
        "edge_vs_bnh": vs_bnh["edge_annual"],
        "breakeven_vs_bnh_bps": vs_bnh["breakeven_bps"],
        "breakeven_cagr_bps": _solve_bps(lambda b: at(b)[0] - bnh_cagr),
        "breakeven_sharpe_bps": _solve_bps(lambda b: at(b)[1] - bnh_sharpe),
    }


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _be_fmt(v: float) -> str:
    if v is None or np.isnan(v):
        return "–"
    if v < 0:
        return "no edge"
    if np.isinf(v):
        return f">{MAX_BPS:,.0f}"
    return f"{v:,.0f}"


def _breakeven_display(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out.index.name = "Asset"
    out["Class"] = df["class"]
    out["Round trips/yr"] = df["round_trips_per_year"].map(lambda v: f"{v:.2f}")
    out["Cost drag at default (bp/yr)"] = df["default_drag_bps_per_year"].map(lambda v: _bps(v))
    out["Edge vs cash"] = df["edge_vs_cash"].map(lambda v: f"{v:.1%}")
    out["Break-even vs cash (bp)"] = df["breakeven_vs_cash_bps"].map(_be_fmt)
    out["Edge vs B&H"] = df["edge_vs_bnh"].map(lambda v: f"{v:.1%}")
    out["Break-even vs B&H (bp)"] = df["breakeven_vs_bnh_bps"].map(_be_fmt)
    out["CAGR = B&H at (bp)"] = df["breakeven_cagr_bps"].map(_be_fmt)
    out["Sharpe = B&H at (bp)"] = df["breakeven_sharpe_bps"].map(_be_fmt)
    return out


# --------------------------------------------------------------------------- 4. execution timing


def _execution(ctx: Context, ticker: str, strategy) -> dict:
    out = {}
    for ex in EXECUTIONS:
        r = ctx.backtest(ticker, strategy, execution=ex)
        p = r.periods_per_year
        out[f"sharpe_{ex}"] = metrics.sharpe(r.returns, r.rf, p)
        out[f"cagr_{ex}"] = metrics.cagr(r.returns, p)
    return out


# --------------------------------------------------------------------------- run


def pick(cond: bool, yes: str, no: str) -> str:
    return yes if cond else no


def run(ctx: Context) -> Section:
    tickers = list(ASSETS)
    grid = BPS_GRID_QUICK if ctx.quick else BPS_GRID

    # 1. spreads
    spreads = _spreads(ctx, tickers)
    spreads.index.name = "ticker"
    spreads_disp = _spreads_display(spreads)
    _save(ctx, f"{NAME}_spreads", spreads, spreads_disp)
    etf = spreads[spreads["class"] != "Crypto"]
    n_robust_zero = int((etf["cs_robust_bps"] < 0.05).sum())
    med_clip = float(etf["cs_daily_clip_bps"].median())
    med_clip_recent = float(etf["cs_daily_clip_recent_bps"].median())
    med_tick = float(etf["tick_floor_bps"].median())
    max_tick = float(etf["tick_floor_bps"].max())
    max_tick_name = str(etf["tick_floor_bps"].idxmax())
    btc_clip = float(spreads.loc["BTC-USD", "cs_daily_clip_bps"])
    btc_robust = float(spreads.loc["BTC-USD", "cs_robust_bps"])

    # 2. cost sweep
    sweeps = {}
    for t in SWEEP_ASSETS:
        kw = {"execution": "next_close", "end": ctx.end}
        sw = costs.cost_sensitivity(ctx.prices(t), STRATEGY, grid, ctx.rf, kw)
        s0 = ctx.backtest(t, STRATEGY, costs=CostModel.zero())
        bnh = ctx.backtest(t, BuyAndHold(), start=s0.start)
        sw["bnh_cagr"] = metrics.cagr(bnh.returns, bnh.periods_per_year)
        sw["bnh_sharpe"] = metrics.sharpe(bnh.returns, bnh.rf, bnh.periods_per_year)
        sweeps[t] = sw
    sweep_long = pd.concat(sweeps, names=["ticker", "bps"])
    sharpe_wide = pd.DataFrame({t: sw["sharpe"] for t, sw in sweeps.items()})
    gap_wide = pd.DataFrame({t: sw["cagr"] - sw["bnh_cagr"] for t, sw in sweeps.items()})
    sweep_disp = pd.DataFrame(
        {
            **{f"{t} Sharpe": sharpe_wide[t].map(lambda v: f"{v:.2f}") for t in SWEEP_ASSETS},
            **{f"{t} CAGR": sweeps[t]["cagr"].map(lambda v: f"{v:.1%}") for t in SWEEP_ASSETS},
        }
    )
    sweep_disp.index = [f"{b:g}" for b in sweep_disp.index]
    sweep_disp.index.name = "bp per side"
    _save(ctx, f"{NAME}_sweep", sweep_long, sweep_disp)
    hi_bps = float(max(grid))
    sharpe_drop = sharpe_wide.iloc[0] - sharpe_wide.loc[hi_bps]
    ref_bps = 20.0 if 20.0 in sharpe_wide.index else float(sorted(grid)[-2])
    drop_ref = sharpe_wide.iloc[0] - sharpe_wide.loc[ref_bps]

    plotting.cost_sensitivity_line(
        sharpe_wide,
        f"Sharpe falls by at most {sharpe_drop.max():.2f} between 0 and {hi_bps:g} bp per side",
        subtitle="EMA50/200, full history per asset, next-close fills, flat cost per side",
        ylabel="Sharpe ratio",
        path=ctx.fig_path(f"{NAME}_sharpe_vs_bps"),
    )
    default_bps = {t: costs_for(t).bps for t in SWEEP_ASSETS}
    gap_default = pd.Series(
        {t: float(np.interp(default_bps[t], gap_wide.index, gap_wide[t])) for t in SWEEP_ASSETS}
    )
    above = [t for t in SWEEP_ASSETS if gap_default[t] > 0]
    gap_title = (
        f"Only {_join(above)} beat{'s' if len(above) == 1 else ''} buy & hold on CAGR; "
        "costs tilt the lines but cross zero for none"
        if above and all((gap_wide[t] > 0).all() for t in above)
        else f"{len(above)} of {len(SWEEP_ASSETS)} beat buy & hold on CAGR at default cost; "
        "costs only tilt the lines"
    )
    plotting.cost_sensitivity_line(
        gap_wide,
        gap_title,
        subtitle="Rule CAGR minus buy-and-hold CAGR (B&H at default cost) over the same window; "
        "0 = parity",
        ylabel="CAGR minus B&H CAGR",
        percent=True,
        path=ctx.fig_path(f"{NAME}_cagr_gap_vs_bps"),
    )

    # 3. break-even
    be = pd.DataFrame({t: _breakeven(ctx, t) for t in tickers}).T
    num = [c for c in be.columns if c != "class"]
    be[num] = be[num].astype(float)
    be.index.name = "ticker"
    be_disp = _breakeven_display(be)
    _save(ctx, f"{NAME}_breakeven", be, be_disp)
    med_rt = float(be["round_trips_per_year"].median())
    med_drag = float(be["default_drag_bps_per_year"].median())
    max_drag = float(be["default_drag_bps_per_year"].max())
    max_drag_name = str(be["default_drag_bps_per_year"].idxmax())
    be_cash = be["breakeven_vs_cash_bps"]
    min_be_cash = float(be_cash.min())
    min_be_cash_name = str(be_cash.idxmin())
    med_be_cash = float(be_cash.median())
    has_cagr_edge = be["breakeven_cagr_bps"] >= 0
    n_cagr_edge = int(has_cagr_edge.sum())
    has_sharpe_edge = be["breakeven_sharpe_bps"] >= 0
    n_sharpe_edge = int(has_sharpe_edge.sum())
    sharpe_be_vals = be.loc[has_sharpe_edge, "breakeven_sharpe_bps"]
    med_sharpe_be = float(sharpe_be_vals.median()) if len(sharpe_be_vals) else float("nan")
    min_sharpe_be = float(sharpe_be_vals.min()) if len(sharpe_be_vals) else float("nan")
    min_sharpe_be_name = str(sharpe_be_vals.idxmin()) if len(sharpe_be_vals) else "–"

    # 4. execution timing
    ex = pd.DataFrame({t: _execution(ctx, t, STRATEGY) for t in tickers}).T
    ex_fast = pd.DataFrame({t: _execution(ctx, t, FAST) for t in tickers}).T
    ex.index.name = ex_fast.index.name = "ticker"
    ex_all = pd.concat({STRATEGY.name: ex, FAST.name: ex_fast}, names=["rule", "ticker"])
    prem = ex["sharpe_same_close"] - ex["sharpe_next_close"]
    prem_open = ex["sharpe_next_open"] - ex["sharpe_next_close"]
    prem_cagr = ex["cagr_same_close"] - ex["cagr_next_close"]
    prem_fast = ex_fast["sharpe_same_close"] - ex_fast["sharpe_next_close"]
    prem_fast_cagr = ex_fast["cagr_same_close"] - ex_fast["cagr_next_close"]
    ex_disp = pd.DataFrame(index=ex.index)
    ex_disp.index.name = "Asset"
    for e in EXECUTIONS:
        ex_disp[f"Sharpe, {EXEC_LABELS[e].lower()}"] = ex[f"sharpe_{e}"].map(lambda v: f"{v:.2f}")
    ex_disp["Same-close premium (Sharpe)"] = prem.map(lambda v: f"{v:+.3f}")
    ex_disp["Same-close premium (CAGR)"] = prem_cagr.map(lambda v: f"{v * 1e4:+.0f} bp")
    ex_disp[f"{FAST.name} premium (Sharpe)"] = prem_fast.map(lambda v: f"{v:+.2f}")
    _save(ctx, f"{NAME}_execution", ex_all, ex_disp)

    lab_sc, lab_no = f"Same close, {STRATEGY.name}", f"Next open, {STRATEGY.name}"
    lab_fast = f"Same close, {FAST.name}"
    bars = pd.DataFrame({lab_sc: prem, lab_no: prem_open, lab_fast: prem_fast}).astype(float)
    n_pos_slow = int((prem > 0).sum())
    plotting.bar_comparison(
        bars,
        f"Same-close fills raise the {STRATEGY.name} Sharpe in "
        f"{pick(n_pos_slow <= len(prem) / 2, 'only ', '')}{n_pos_slow} of "
        f"{len(prem)} markets, by at most {prem.max():+.3f}",
        subtitle="Sharpe minus the next-close Sharpe, default costs, full history per asset; "
        f"the fast {FAST.name} rule shown for contrast",
        percent=False,
        fmt=lambda v: _minus(f"{v:+.3f}"),
        xlabel="Sharpe difference vs next-close fills",
        colors={
            lab_sc: plotting.STRATEGY_COLOR,
            lab_no: plotting.CATEGORICAL[2],
            lab_fast: plotting.CATEGORICAL[1],
        },
        path=ctx.fig_path(f"{NAME}_execution"),
    )

    # ---- text
    n = len(tickers)
    assumed = costs_for("SPY").bps
    assumed_half = costs_for("SPY").half_spread_bps
    half_tick_ok = assumed_half >= max_tick / 2.0
    spread_txt = (
        f"The robust Corwin-Schultz estimate (window averages clipped at zero) is "
        f"essentially zero for {n_robust_zero} of {len(etf)} exchange-traded assets; the "
        f"paper's baseline (two-day estimates clipped at zero) gives an upward-biased median "
        f"{med_clip:.0f} bp ({med_clip_recent:.0f} bp over the last {RECENT_YEARS} years). "
        f"The one-tick spread at the latest price is a median "
        f"{med_tick:.2f} bp (widest: {max_tick_name}, {max_tick:.2f} bp), so the assumed "
        f"{assumed_half:g} bp half-spread "
        + pick(
            half_tick_ok,
            f"is at least half a tick for every US-listed asset at today's prices, and the "
            f"{assumed:g} bp all-in cost per side is generous for those that quote at one tick. "
            f"That is not a safe bound for the early sample: before decimalisation (2001) US "
            f"shares were quoted in fractions of a dollar and relative spreads were wider.",
            f"is below half a tick for {max_tick_name}, so the cost assumption is not "
            f"conservative there.",
        )
        + f" Bitcoin: {btc_robust:.0f} bp robust, {btc_clip:.0f} bp daily-clipped."
    )
    edge_items = [
        f"{t} up to {_be_fmt(v)} bp" for t, v in be.loc[has_cagr_edge, "breakeven_cagr_bps"].items()
    ]
    be_txt = (
        f"Measured against T-bills, the arithmetic edge disappears only at {min_be_cash:,.0f} bp "
        f"per side for the weakest asset ({min_be_cash_name}; median {med_be_cash:,.0f} bp). "
        f"Against buy-and-hold the rule's CAGR is higher even at zero cost in "
        + pick(n_cagr_edge <= n / 2, "only ", "")
        + f"{n_cagr_edge} of {n} assets"
        + (f" ({'; '.join(edge_items)})" if edge_items else "")
        + f"; its Sharpe is higher in {n_sharpe_edge} of {n}"
        + (
            f", an advantage that survives to a median {med_sharpe_be:,.0f} bp per side "
            f"(lowest: {min_sharpe_be_name}, {min_sharpe_be:,.0f} bp)"
            if n_sharpe_edge
            else ""
        )
        + "."
    )
    # assets where a plausible cost (up to 4x the default) flips the comparison with B&H
    plausible = 4 * be["default_bps"]
    fragile = [
        t
        for t in tickers
        if 0 <= be.loc[t, "breakeven_cagr_bps"] <= plausible[t]
        or 0 <= be.loc[t, "breakeven_sharpe_bps"] <= plausible[t]
    ]
    abs_slow = float(prem.abs().mean())
    abs_fast = float(prem_fast.abs().mean())
    fast_more = abs_fast > 2 * abs_slow
    autocorr = float(np.median([ctx.prices(t)["Close"].pct_change().autocorr(1) for t in tickers]))
    ac_txt = pick(
        autocorr < 0,
        "consistent with daily returns that are, if anything, mildly mean-reverting",
        "even though daily returns are, if anything, mildly trending",
    )
    exec_txt = (
        f"Same-close instead of next-close fills changes the 50/200 Sharpe by a median "
        f"{prem.median():+.3f} (range {prem.min():+.3f} to {prem.max():+.3f}; CAGR "
        f"{prem_cagr.median() * 1e4:+.0f} bp a year); next-open fills by "
        f"{prem_open.median():+.3f}. For a fast {FAST.name} rule the mean absolute "
        f"same-close effect is {abs_fast:.3f} Sharpe vs {abs_slow:.3f} for 50/200"
        + pick(fast_more, ", several times larger", ", of similar size")
        + f", but its median is {prem_fast.median():+.3f}: "
        + pick(
            abs(prem_fast.median()) < 0.05,
            "noise around zero rather than a systematic gain.",
            "a systematic " + pick(prem_fast.median() > 0, "gain.", "loss."),
        )
    )
    # the notebook's own sample: NVDA 2022-2024, indicators warmed up on prior history
    nb = {
        e: ctx.backtest("NVDA", STRATEGY, execution=e, start=NB_WINDOW[0], end=NB_WINDOW[1])
        for e in ("same_close", "next_close")
    }
    nb_tr = {e: metrics.total_return(r.returns) for e, r in nb.items()}
    nb_sh = {e: metrics.sharpe(r.returns, r.rf, r.periods_per_year) for e, r in nb.items()}
    exec_txt += (
        f" On the notebook's own NVDA {NB_WINDOW[0][:4]}-{NB_WINDOW[1][:4]} window, same-close "
        f"fills give {nb_tr['same_close']:+.0%} total return vs {nb_tr['next_close']:+.0%} "
        f"for next-close (Sharpe {nb_sh['same_close']:.2f} vs {nb_sh['next_close']:.2f})."
    )
    fragile_txt = (
        f" The exception is {_join(fragile)}, where the comparison with buy-and-hold is "
        f"close enough that a cost within 4x the default flips it."
        if fragile
        else ""
    )
    small_exec = abs(prem.median()) < 0.02
    switches = 2 * med_rt
    nb_helped = nb_tr["same_close"] > nb_tr["next_close"]
    exec_interp = pick(
        small_exec,
        f"barely matters for a 50/200 rule: with about {switches:.1f} position changes a "
        f"year, the one-day difference touches roughly {10 * switches:.0f} days per decade",
        "materially changes the 50/200 results",
    ) + pick(
        nb_helped,
        ", although on the notebook's short NVDA window it did flatter the result",
        ", and on the notebook's own NVDA window it did not flatter the result either",
    )

    text = f"""**Question.** Are the default frictions ({assumed:g} bp per side, {costs_for("BTC-USD").bps:g} bp for Bitcoin, next-close fills) realistic, and would different costs or fill timing change any conclusion?

**Setup.** Spreads are estimated from daily highs and lows with Corwin & Schultz (2012); EMA50/200 is re-run with flat costs of {", ".join(f"{b:g}" for b in grid)} bp per side; break-even costs are solved exactly (arithmetic edge / turnover) and by root-finding full backtests; and every asset is re-run with same-close, next-open and next-close fills.

**Findings.**

- {spread_txt}
- The rule trades a median {med_rt:.2f} round trips a year, so the default costs drag returns by a median {med_drag:.0f} bp a year (most: {max_drag_name}, {max_drag:.0f} bp).
- Raising costs from 0 to {ref_bps:g} bp per side lowers Sharpe by at most {drop_ref.max():.3f} across {", ".join(SWEEP_ASSETS)}; even {hi_bps:g} bp lowers it by at most {sharpe_drop.max():.2f}.
- {be_txt}
- {exec_txt}

**Interpretation.** A slow trend rule is a low-turnover strategy: with {pick(med_rt < 1, "under one round trip", f"about {med_rt:.1f} round trips")} a year, costs scale with a handful of trades, not with the holding period, so realistic friction is not what makes it {pick(n_cagr_edge > n / 2, "beat", "trail")} buy-and-hold on CAGR in most markets; the verdict is set by the signal.{fragile_txt} This matches the finding that trend-following returns survive estimated trading costs (Hurst, Ooi & Pedersen 2017), and a 50/200 rule trades far less than their 1-12 month signals. The daily-bar spread estimates are bounds, not measurements: when the spread is a cent and daily volatility is a percent, the Corwin & Schultz (2012) estimator either collapses to zero or, clipped day by day, overstates the spread many times over. The same-close fill convention {exec_interp}. Even for a fast rule its sign is unstable, {ac_txt} (median lag-1 autocorrelation {autocorr:+.3f}); whatever inflated the notebook's numbers (section 1), fill timing is at most a small part of it on full histories.
"""

    after = (
        "Costs are proportional and symmetric; market impact is ignored, which is reasonable "
        "for retail size in these ETFs but not for large orders in thinner products such as "
        "DBC or early Bitcoin. Crypto fees vary widely by venue and account tier. The "
        f"break-even search is capped at {MAX_BPS:,.0f} bp per side."
    )

    results = {
        "etf_cs_robust_zero_count": n_robust_zero,
        "etf_cs_daily_clip_median_bps": med_clip,
        "etf_cs_daily_clip_recent_median_bps": med_clip_recent,
        "etf_tick_floor_median_bps": med_tick,
        "btc_cs_daily_clip_bps": btc_clip,
        "median_round_trips_per_year": med_rt,
        "median_cost_drag_bps_per_year": med_drag,
        "max_sharpe_drop_at_20bps": float(drop_ref.max()),
        "max_sharpe_drop_at_max_bps": float(sharpe_drop.max()),
        "min_breakeven_vs_cash_bps": min_be_cash,
        "median_breakeven_vs_cash_bps": med_be_cash,
        "n_cagr_edge_vs_bnh": n_cagr_edge,
        "n_sharpe_edge_vs_bnh": n_sharpe_edge,
        "median_breakeven_sharpe_bps": med_sharpe_be,
        "median_same_close_premium_sharpe": float(prem.median()),
        "median_same_close_premium_cagr": float(prem_cagr.median()),
        "median_next_open_premium_sharpe": float(prem_open.median()),
        "median_fast_same_close_premium_sharpe": float(prem_fast.median()),
        "median_fast_same_close_premium_cagr": float(prem_fast_cagr.median()),
    }
    for t in tickers:
        results[f"{t}_breakeven_vs_cash_bps"] = float(be.loc[t, "breakeven_vs_cash_bps"])
        v = float(be.loc[t, "breakeven_cagr_bps"])
        results[f"{t}_breakeven_cagr_bps"] = v if v >= 0 else "no edge"

    return Section(
        name=NAME,
        title=TITLE,
        text=text,
        figures=[
            ("Sharpe ratio vs cost per side.", ctx.fig_path(f"{NAME}_sharpe_vs_bps")),
            ("CAGR gap to buy & hold vs cost per side.", ctx.fig_path(f"{NAME}_cagr_gap_vs_bps")),
            ("Sharpe ratio by fill convention.", ctx.fig_path(f"{NAME}_execution")),
        ],
        tables=[
            ("Estimated bid-ask spreads vs assumed costs.", spreads_disp),
            ("Cost sweep: Sharpe and CAGR by cost per side.", sweep_disp),
            (
                "Break-even costs per side (bp); 'no edge' = no advantage even at zero cost.",
                be_disp,
            ),
            ("Fill-timing sensitivity.", ex_disp),
        ],
        after=after,
        results=results,
    )
